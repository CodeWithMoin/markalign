"""The hot path: grade one essay against a rubric + cached profile.

Output is validated against GradingResult, so span-grounding is enforced by the
schema, not by hope. The abstention gate is a separate, tunable step.
"""
from __future__ import annotations

from .schemas import (Essay, Rubric, TeacherProfile, GradingResult,
                      TraitJudgement, Span)
from . import model_adapter as ma


SYSTEM = """
You are grading a student essay AS ONE SPECIFIC TEACHER, not as a generic assessor.

Your task is to predict the mark THIS teacher would actually give the essay.

You are given this teacher's inferred marking standard — what they reward, penalise, prioritise, and weigh. Treat that standard as the authority for grading. Do not replace it with a generic rubric, your own preferences, or your personal judgment of what makes an essay good.

If this teacher is harsh, grade harshly. If they are lenient, grade leniently. If they consistently reward or penalise something that you would personally judge differently, follow the teacher's pattern.

The question is not:
"What mark does this essay deserve?"

The question is:
"What mark would THIS teacher give this essay?"

## Evidence rule — critical

Every trait score MUST be supported by one or more spans that are EXACT, VERBATIM substrings of the student's essay.

Supporting spans must:
- appear exactly in the essay;
- preserve the student's original wording;
- preserve spelling, grammar, punctuation, and capitalisation;
- contain no corrections or paraphrasing;
- never be invented or reconstructed.

Before using a span, verify that the exact sequence of words appears in the essay.

If you cannot find a genuine supporting span for a trait, do not fabricate one. The score may still be assigned if the teacher's standard clearly supports it, but the lack of direct textual evidence should reduce your confidence.

Do not use evidence from the teacher's inferred standard as if it were evidence from the student's essay.

## Grading principle

Judge each trait according to how THIS teacher appears to distinguish stronger work from weaker work.

Do not award points simply because a feature is generally considered good.

Do not penalise something merely because you personally dislike it.

Pay particular attention to the teacher's demonstrated preferences, including:
- what they consistently reward;
- what they consistently penalise;
- what they appear to weigh heavily;
- what they seem to care about less;
- and what separates essays receiving different marks.

When the evidence is ambiguous, prefer the interpretation that is most consistent with the teacher's observed marking behavior.

## Confidence

Confidence measures how certain you are that the assigned mark matches what THIS teacher would actually give the essay.

Use approximately:
- 0.8–1.0 when the essay clearly fits the teacher's established marking patterns.
- 0.5–0.7 when the essay is borderline or the evidence is mixed.
- below 0.5 when the essay is genuinely ambiguous against the teacher's standard.

Do not inflate confidence merely because you can justify a score.

A high confidence score means:
"I am confident this is the mark THIS teacher would give."

It does NOT mean:
"I think this is objectively a good essay."

Your final judgment should therefore reflect the teacher's demonstrated marking behavior, not an abstract definition of essay quality.
"""


def _snap_quote(quote: str, text: str) -> str | None:
    """Enforce verbatim evidence with code, not hope. Models silently 'fix' student
    spelling inside quotes ('I new' -> 'I knew') — a small hallucination wearing
    evidence's clothes. If the quote isn't an exact substring, snap it to the
    closest real passage; if nothing close exists, it's fabricated -> None."""
    q = quote.strip()
    if not q:
        return None
    if q in text:
        return q
    from difflib import SequenceMatcher
    n = len(q)
    best, best_r = None, 0.0
    step = max(1, n // 4)
    for i in range(0, max(1, len(text) - n + 1), step):
        cand = text[i:i + n]
        r = SequenceMatcher(None, q.lower(), cand.lower()).ratio()
        if r > best_r:
            best_r, best = r, i
    if best is None or best_r < 0.75:
        return None
    # expand the best window to word boundaries so the snap reads cleanly
    lo, hi = best, min(len(text), best + n)
    while lo > 0 and text[lo - 1] not in " \n":
        lo -= 1
    while hi < len(text) and text[hi] not in " \n":
        hi += 1
    snapped = text[lo:hi].strip()
    # drop over-captured trailing words while doing so reads closer to the quote
    ql = q.lower()
    while " " in snapped:
        shorter = snapped.rsplit(" ", 1)[0]
        if SequenceMatcher(None, ql, shorter.lower()).ratio() <= \
           SequenceMatcher(None, ql, snapped.lower()).ratio():
            break
        snapped = shorter
    return snapped


def _anchor_block(rubric: Rubric, anchors) -> str:
    """A few of the teacher's own scored essays, so the model anchors the ABSOLUTE
    scale (what a 6 vs a 9 looks like) instead of guessing where the scale sits.
    The abstract profile says *what* is rewarded; these say *how it maps to a number*."""
    if not anchors:
        return ""
    lines = []
    for a in anchors:
        traits = (" · traits " + ", ".join(f"{k} {v}" for k, v in a.trait_scores.items())
                  if a.trait_scores else "")
        lines.append(f"  --- this teacher scored the essay below {a.holistic_score}"
                     f"/{rubric.holistic_max}{traits} ---\n  {a.text[:500].strip()}")
    return ("<calibration_examples>\n"
            "How THIS teacher actually scored real essays — anchor your absolute scale to these:\n"
            + "\n".join(lines) + "\n</calibration_examples>\n\n")


def _user_prompt(essay: Essay, rubric: Rubric, profile: TeacherProfile, anchors=None) -> str:
    traits = "\n".join(f"  - {t.name} (score {t.min_score}-{t.max_score}): {t.description}"
                       for t in rubric.traits)
    rewards = "\n".join(f"    - {p.pattern} (evidence: {p.evidence}) [{p.strength}]"
                        for p in profile.rewards)
    penalizes = "\n".join(f"    - {p.pattern} (evidence: {p.evidence}) [{p.strength}]"
                          for p in profile.penalizes)
    return (
        f"<task>\n{rubric.prompt}\n</task>\n\n"
        f"<teacher_standard>\n"
        f"  rewards:\n{rewards}\n"
        f"  penalises:\n{penalizes}\n"
        f"  severity: {profile.severity}\n"
        f"  trait emphasis: {profile.trait_emphasis}\n"
        f"  notes: {profile.notes}\n"
        f"</teacher_standard>\n\n"
        f"<traits>\n{traits}\n</traits>\n\n"
        f"<holistic_range>{rubric.holistic_min} to {rubric.holistic_max}</holistic_range>\n\n"
        + _anchor_block(rubric, anchors) +
        f"<essay>\n{essay.text}\n</essay>\n\n"
        "Grade the essay above as this teacher would. Score each trait within its range, "
        "then give a holistic score within the holistic range. Return ONLY this JSON, no prose:\n"
        "{\n"
        '  "trait_judgements": [\n'
        '    {"trait": <trait name>, "score": <int in range>,\n'
        '     "spans": [{"quote": <exact verbatim substring of the essay>, "reason": <why it supports this score>}]}\n'
        "  ],\n"
        '  "holistic_score": <int in the holistic range>,\n'
        '  "confidence": <float 0-1, per the confidence rule>,\n'
        '  "summary": <one sentence on how this essay meets or misses THIS teacher\'s bar>\n'
        "}\n"
        "Every 'quote' must appear character-for-character in the essay above."
    )


def _checklist_prompt(essay: Essay, rubric: Rubric, profile: TeacherProfile, anchors=None) -> str:
    """Binary decomposition: instead of one wobbly 0-N rating per trait, answer the
    teacher-derived yes/no ladder and count the yeses. LLM judges are steadier on
    binary decisions than on ordinal scales (scoring-bias literature), and every
    'yes' must point at a span — evidence is forced, not optional."""
    blocks = []
    for t in rubric.traits:
        qs = (profile.checklist or {}).get(t.name, [])
        blocks.append(f"  {t.name}:\n" + "\n".join(f"    {i+1}. {q}" for i, q in enumerate(qs)))
    rewards = "\n".join(f"    - {p.pattern} [{p.strength}]" for p in profile.rewards)
    penalizes = "\n".join(f"    - {p.pattern} [{p.strength}]" for p in profile.penalizes)
    return (
        f"<task>\n{rubric.prompt}\n</task>\n\n"
        f"<teacher_standard>\n  rewards:\n{rewards}\n  penalises:\n{penalizes}\n"
        f"  severity: {profile.severity}\n  notes: {profile.notes}\n</teacher_standard>\n\n"
        f"<checklist>\nThis teacher's bar, decomposed into yes/no questions per trait "
        f"(ordered easiest to hardest):\n" + "\n".join(blocks) + "\n</checklist>\n\n"
        + _anchor_block(rubric, anchors) +
        f"<essay>\n{essay.text}\n</essay>\n\n"
        "Answer EVERY checklist question yes or no, judging by THIS teacher's bar (from "
        "their standard and scored examples), not a generic one. For each answer give a "
        "verbatim quote from the essay: for 'yes', the evidence that clears the bar; for "
        "'no', the closest attempt that falls short (or the opening words if nothing "
        "relevant exists). Return ONLY this JSON:\n"
        "{\n"
        '  "checklist_answers": [\n'
        '    {"trait": <trait name>, "answers": [\n'
        '       {"q": <question number>, "yes": true|false,\n'
        '        "quote": <exact verbatim substring of the essay>, "reason": <one short sentence>}\n'
        "    ]}\n"
        "  ],\n"
        '  "confidence": <float 0-1: how sure you are these answers match this teacher>,\n'
        '  "summary": <one sentence on how the essay meets or misses this teacher\'s bar>\n'
        "}\n"
        "Every 'quote' must appear character-for-character in the essay above."
    )


def _grade_checklist(essay: Essay, rubric: Rubric, profile: TeacherProfile,
                     anchors=None) -> GradingResult:
    raw = ma.complete_json(SYSTEM, _checklist_prompt(essay, rubric, profile, anchors))
    by_trait = {a["trait"]: a["answers"] for a in raw["checklist_answers"]}
    judgements = []
    for t in rubric.traits:
        answers = by_trait.get(t.name, [])
        score = max(t.min_score, min(t.max_score, sum(1 for a in answers if a.get("yes"))))
        # spans: evidence behind the yes answers; if none cleared, the closest-miss quotes
        picked = [a for a in answers if a.get("yes")] or answers
        spans = []
        for a in picked:
            q = _snap_quote(a.get("quote", ""), essay.text)
            if q:
                spans.append(Span(quote=q, reason=a["reason"]))
            elif a.get("quote"):
                spans.append(Span(quote=a["quote"],
                                  reason="[unverified quote — not found in the essay] " + a["reason"]))
        if not spans:  # schema requires grounding; fall back to the essay opening
            spans = [Span(quote=essay.text[:60], reason="no checklist bar cleared for this trait")]
        judgements.append(TraitJudgement(trait=t.name, score=score, spans=spans))
    total = sum(j.score for j in judgements)
    holistic = max(rubric.holistic_min, min(rubric.holistic_max, total))
    return GradingResult(
        essay_id=essay.essay_id, grader_id=profile.grader_id,
        trait_judgements=judgements, holistic_score=holistic,
        confidence=float(raw["confidence"]), summary=raw["summary"],
    )


def grade(essay: Essay, rubric: Rubric, profile: TeacherProfile,
          *, anchors=None, mock: bool = False) -> GradingResult:
    if profile.checklist and not mock:
        return _grade_checklist(essay, rubric, profile, anchors)
    raw = ma.complete_json(
        SYSTEM, _user_prompt(essay, rubric, profile, anchors),
        mock=mock, mock_kind="grade",
        ctx={"essay_id": essay.essay_id, "text": essay.text,
             "traits": [t.name for t in rubric.traits],
             "trait_min": min(t.min_score for t in rubric.traits),
             "trait_max": max(t.max_score for t in rubric.traits),
             "holistic_min": rubric.holistic_min,
             "holistic_max": rubric.holistic_max},
    )
    judgements = []
    for tj in raw["trait_judgements"]:
        spans = []
        for s in tj["spans"]:
            q = _snap_quote(s["quote"], essay.text)
            spans.append(Span(quote=q or s["quote"],
                              reason=s["reason"] if q else
                              "[unverified quote — not found in the essay] " + s["reason"]))
        judgements.append(TraitJudgement(trait=tj["trait"], score=tj["score"], spans=spans))
    # Match metric to label construction: when the human holistic IS the trait sum
    # (set 7), build the model's holistic the same way instead of a free-form guess.
    if rubric.holistic_from_traits and judgements:
        total = sum(j.score for j in judgements)
        holistic = max(rubric.holistic_min, min(rubric.holistic_max, total))
    else:
        holistic = raw["holistic_score"]
    return GradingResult(
        essay_id=essay.essay_id,
        grader_id=profile.grader_id,
        trait_judgements=judgements,
        holistic_score=holistic,
        confidence=float(raw["confidence"]),
        summary=raw["summary"],
    )

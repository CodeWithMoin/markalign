"""Standard inference: turn a grader's marked essays into an inspectable profile.

Runs once per grader, then cached. It is the only step that 'learns' a teacher's
standard, kept isolated so you can ablate it (grade with profile / without / with
the WRONG grader's profile — that last one is the personalization test).
"""
from __future__ import annotations

from .schemas import GradedExample, Rubric, TeacherProfile
from . import model_adapter as ma

SYSTEM = """
You are an assessment analyst reverse-engineering the marking behavior of one specific teacher.

You are given multiple essays this teacher has scored, including both high- and low-scoring examples. Your job is to identify what actually moves THIS teacher's marks — not what generally makes an essay "good."

## Method

Treat the teacher's scores as the ground truth.

Compare the highest-scoring essays with the lowest-scoring essays. Look for features that are consistently more present, more developed, or handled differently in the high-scoring essays, and consistently absent, weaker, or handled differently in the low-scoring essays.

A real marking pattern must appear across several essays. Do not infer a preference from a single example.

If two essays share a quality but received meaningfully different marks, that quality alone is NOT a useful explanation for the difference. Keep looking for what separates them.

Also distinguish between simply having a feature and how that feature is executed. If both high- and low-scoring essays use evidence, for example, investigate whether the teacher rewards a particular kind of evidence use: greater specificity, closer explanation, stronger connection to the argument, better placement, or something else.

## What to report

Prioritize features that actually discriminate between marks.

For each claimed marking pattern, explain:

- What the high-scoring essays do.
- What the low-scoring essays do differently or fail to do.
- Why this difference appears to matter to THIS teacher.
- How strong the evidence is across the essays.

Be concrete and grader-specific.

Avoid generic observations such as:
- "rewards clear structure"
- "uses good evidence"
- "has strong analysis"
- "writes persuasively"

Instead, identify the observable behavior that appears to affect the mark.

For example:
- "Rewards an explicitly stated counterargument even when the rebuttal is brief."
- "Marks evidence more highly when the writer immediately explains how it proves the specific claim, rather than leaving the quotation to speak for itself."

## Evidence and uncertainty

Separate strong patterns from weak or uncertain ones.

If a feature appears in both high- and low-scoring essays, do not call it a marking preference unless there is evidence that the teacher rewards a particular way of using it.

If the evidence is inconsistent, sparse, or cannot distinguish between competing explanations, say so explicitly.

Do not infer what the teacher should reward. Do not fill gaps with general essay-writing advice.

Your goal is not to describe good writing.

Your goal is to reverse-engineer the teacher's actual marking behavior from the essays and scores provided.
"""




def _checklist_spec(rubric: Rubric) -> str:
    """Ask for a per-trait binary decomposition: max_score yes/no questions per
    trait, ordered easiest→hardest, so a trait score becomes a count of yeses.
    The questions must encode THIS teacher's bar (from the examples), not a
    generic rubric — that is what keeps the checklist personalized."""
    per_trait = ", ".join(f'"{t.name}": [<exactly {t.max_score} questions>]'
                          for t in rubric.traits)
    return (
        ',\n  "checklist": {' + per_trait + "}\n"
        "  // For each trait, write EXACTLY max_score yes/no questions about an essay,\n"
        "  // ordered as a ladder: question 1 is the bar almost every essay this teacher\n"
        "  // passed clears; the last question only their top-scored essays clear.\n"
        "  // Derive each bar from what actually separated scores in the examples above\n"
        "  // (e.g. if essays with a named, specific event scored higher on ideas, a\n"
        "  // question is 'Does the essay describe one specific, concrete event?').\n"
        "  // A trait score will be computed as the COUNT of yes answers, so the ladder\n"
        "  // must be cumulative: an essay that clears question 3 also clears 1 and 2.\n"
    )


def _score_distribution(rubric: Rubric, examples: list[GradedExample]) -> str:
    """Computed facts about how this teacher actually uses the scale — counted by
    code, not left for the model to notice across 25 documents. The key fact for
    ladder design: which scores the teacher never/rarely gives (the real floor)."""
    lines = ["\nHOW THIS TEACHER USES THE SCALE (counted over the essays above):"]
    for t in rubric.traits:
        scores = [e.trait_scores[t.name] for e in examples if t.name in e.trait_scores]
        if not scores:
            continue
        counts = {v: scores.count(v) for v in range(t.min_score, t.max_score + 1)}
        used = ", ".join(f"{v}: {c}x" for v, c in counts.items())
        floor = next((v for v in sorted(counts) if counts[v]), t.min_score)
        lines.append(f"  - {t.name}: {used}"
                     + (f"  (never goes below {floor} in this sample)" if floor > t.min_score else ""))
    lines.append(
        "Calibrate any per-trait ladder to this: if the teacher never or almost never "
        "uses the bottom score, the FIRST question must be a bar nearly every essay "
        "clears (their de-facto floor), and the ladder's spread should mirror where "
        "their scores actually cluster.")
    return "\n".join(lines) + "\n"


def _user_prompt(rubric: Rubric, examples: list[GradedExample],
                 checklist: bool = False) -> str:
    traits = "\n".join(f"  - {t.name}: {t.description}" for t in rubric.traits)
    lines = [f"TASK the students responded to:\n{rubric.prompt}",
             f"\nTRAITS this teacher weighs:\n{traits}",
             f"\nHolistic score range: {rubric.holistic_min} (lowest) to {rubric.holistic_max} (highest)",
             "\nEssays this teacher marked, shown WITH their score so you can read the "
             "gradient. Study why the high ones scored high and the low ones scored low:"]
    # Sort high→low so the model reads the standard as a gradient, not a shuffle.
    for ex in sorted(examples, key=lambda e: e.holistic_score, reverse=True):
        lines.append(f"\n--- essay {ex.essay_id} · this teacher gave it {ex.holistic_score}"
                     f"/{rubric.holistic_max} ---")
        lines.append(ex.text[:1200])
        if ex.note:
            lines.append(f"[teacher's own note] {ex.note}")
    lines.append(_score_distribution(rubric, examples))
    lines.append(
        f"\nNow summarise THIS teacher's standard as JSON. Each reward/penalty is a "
        "discriminating pattern WITH its evidence and how strong that evidence is:\n"
        "{\n"
        '  "rewards": [\n'
        '    {"pattern": <the behavior that lifts the mark for THIS teacher>,\n'
        '     "evidence": <what the high-scoring essays do that the low ones do not>,\n'
        '     "strength": "strong" | "moderate" | "weak"}\n'
        "  ],\n"
        '  "penalizes": [\n'
        '    {"pattern": <the behavior that costs marks>,\n'
        '     "evidence": <what the low-scoring essays do that the high ones avoid>,\n'
        '     "strength": "strong" | "moderate" | "weak"}\n'
        "  ],\n"
        '  "severity": "lenient" | "balanced" | "harsh",   // judged against how they used the score range\n'
        '  "trait_emphasis": {trait_name: "high" | "medium" | "low"},  // which traits drive their score\n'
        '  "notes": str             // 1-2 sentences a human grader would recognise as true of them;\n'
        "                           // flag here if the sample was too small to be sure\n"
        + (_checklist_spec(rubric) if checklist else "") +
        "}\n"
        "Every pattern must be visible across multiple essays above; mark thin evidence "
        "as \"weak\" rather than dropping it. When citing an essay in evidence, refer to "
        "it by its score (e.g. \"the 12-scoring essay\", \"the essays scoring 4 and below\"), "
        "never by its id — ids are internal and meaningless to a reader. Output only the JSON."
    )
    return "\n".join(lines)


def build_profile(grader_id: str, rubric: Rubric,
                  examples: list[GradedExample], *, mock: bool = False,
                  checklist: bool = False) -> TeacherProfile:
    raw = ma.complete_json(
        SYSTEM, _user_prompt(rubric, examples, checklist=checklist and not mock),
        mock=mock, mock_kind="profile",
        ctx={"grader_id": grader_id,
             "traits": [t.name for t in rubric.traits],
             "n": len(examples)},
    )
    return TeacherProfile(
        grader_id=grader_id,
        set_id=rubric.set_id,
        rewards=raw["rewards"],
        penalizes=raw["penalizes"],
        severity=raw["severity"],
        trait_emphasis=raw["trait_emphasis"],
        notes=raw["notes"],
        n_examples=len(examples),
        checklist=raw.get("checklist"),
    )

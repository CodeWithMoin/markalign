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




def _user_prompt(rubric: Rubric, examples: list[GradedExample]) -> str:
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
        "}\n"
        "Every pattern must be visible across multiple essays above; mark thin evidence "
        "as \"weak\" rather than dropping it. Output only the JSON."
    )
    return "\n".join(lines)


def build_profile(grader_id: str, rubric: Rubric,
                  examples: list[GradedExample], *, mock: bool = False) -> TeacherProfile:
    raw = ma.complete_json(
        SYSTEM, _user_prompt(rubric, examples),
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
    )

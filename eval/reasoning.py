"""Reasoning-agreement: the part ASAP CANNOT verify for you.

Score-agreement asks "same mark as the human?". This asks the harder question:
"right mark for the right reason, or lucky?" — and splits wrong answers into
systematic (fixable) vs random (the ceiling).

There is no reasoning ground truth in ASAP, so this rests on sampled human/model
adjudication. Its credibility depends on documenting the labelling criteria, not
on the dataset. Keep the sample small and honest (~60-100), same budget shape as
the preprint adjudications.
"""
from __future__ import annotations

from collections import Counter

from backend.schemas import GradingResult, Adjudication
from backend import model_adapter as ma


SYSTEM = (
    "You audit a single essay grade against the human score. Decide whether the "
    "predicted score matches, and whether the model's cited spans actually justify "
    "its judgement. Label the case with one of the fixed categories. Be strict: "
    "a correct score with irrelevant spans is 'lucky', not 'grounded'."
)

LABELS = [
    "correct_score_grounded",
    "correct_score_ungrounded",
    "wrong_systematic",
    "wrong_random",
    "correct_abstention",
]


def adjudicate(result: GradingResult, human_holistic: int, essay_text: str,
               *, mock: bool = False) -> Adjudication:
    short = len(essay_text.split()) < 120
    raw = ma.complete_json(
        SYSTEM,
        f"Human holistic: {human_holistic}\n"
        f"Predicted holistic: {result.holistic_score}\n"
        f"Abstained: {result.abstained}\n"
        f"Cited spans: {[s.quote for tj in result.trait_judgements for s in tj.spans]}\n"
        f"Essay length (words): {len(essay_text.split())}\n"
        f"Return JSON: {{label: one of {LABELS}, rationale: str}}.",
        mock=mock, mock_kind="adjudicate",
        ctx={"human": human_holistic, "pred": result.holistic_score,
             "short": short, "grounded": bool(result.trait_judgements)},
    )
    return Adjudication(
        essay_id=result.essay_id,
        grader_id=result.grader_id,
        human_holistic=human_holistic,
        predicted_holistic=result.holistic_score,
        label=raw["label"],
        rationale=raw["rationale"],
    )


def summarize(adjs: list[Adjudication]) -> dict:
    c = Counter(a.label for a in adjs)
    total = len(adjs) or 1
    wrong = c["wrong_systematic"] + c["wrong_random"]
    return {
        "n": len(adjs),
        "counts": dict(c),
        "share": {k: round(c[k] / total, 3) for k in LABELS},
        # The headline: of the wrong grades, how many are fixable vs the ceiling?
        "of_wrong_share_systematic": round(c["wrong_systematic"] / wrong, 3) if wrong else None,
        "lucky_rate": round(c["correct_score_ungrounded"] / total, 3),
    }

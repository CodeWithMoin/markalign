"""Load essays into the common schema and split each rater into a distinct grader.

The rater1/rater2 split is the trick: two humans marked the same essays to a
shared rubric, so treating them as two 'graders' gives a real personalization
test AND real labels without annotating anything. The gap between them is small
BY CONSTRUCTION (they were trained to agree) — so a small personalization effect
is the expected, honest result.

Works with the bundled synthetic sample out of the box. To use real ASAP-AES,
convert its TSV to the same records shape (see README).
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.schemas import Rubric, Trait, GradedExample, Essay


def load_rubric(path: str | Path) -> Rubric:
    d = json.loads(Path(path).read_text())
    return Rubric(
        set_id=d["set_id"], prompt=d["prompt"],
        traits=[Trait(**t) for t in d["traits"]],
        holistic_min=d["holistic_min"], holistic_max=d["holistic_max"],
        holistic_from_traits=d.get("holistic_from_traits", False),
    )


def load_records(path: str | Path) -> list[dict]:
    """Each record: essay_id, text, rater1_holistic, rater2_holistic,
    rater1_traits, rater2_traits (trait dicts)."""
    return json.loads(Path(path).read_text())


def graded_examples(records: list[dict], rater: str) -> list[GradedExample]:
    return [
        GradedExample(
            essay_id=r["essay_id"], text=r["text"],
            trait_scores=r[f"{rater}_traits"],
            holistic_score=r[f"{rater}_holistic"],
        )
        for r in records
    ]


def essays(records: list[dict], set_id: str) -> list[Essay]:
    return [Essay(essay_id=r["essay_id"], set_id=set_id, text=r["text"])
            for r in records]

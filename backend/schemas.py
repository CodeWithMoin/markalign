"""Typed contracts for the whole system.

The GradingResult schema is the load-bearing design decision: span-grounding and
confidence are enforced here, so an ungrounded rationale is a validation failure
rather than prose we hoped the model produced.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------- Inputs ----------

class Trait(BaseModel):
    name: str                      # e.g. "content", "organization"
    description: str               # what this trait rewards
    min_score: int = 1
    max_score: int = 6


class Rubric(BaseModel):
    set_id: str                    # ASAP essay-set id, or a school/task id
    prompt: str                    # the essay task the student responded to
    traits: list[Trait]
    holistic_min: int = 2
    holistic_max: int = 12
    # When the human holistic is DEFINED as the sum of the trait scores (e.g. ASAP
    # set 7), construct the model's holistic the same way — sum its trait judgements
    # instead of trusting a free-form holistic. Matches metric to label construction.
    holistic_from_traits: bool = False


class GradedExample(BaseModel):
    """One essay this grader already marked — the calibration signal."""
    essay_id: str
    text: str
    trait_scores: dict[str, int]   # trait name -> score
    holistic_score: int
    note: Optional[str] = None     # optional teacher rationale, if present


class Essay(BaseModel):
    essay_id: str
    set_id: str
    text: str


# ---------- Learned artifact ----------

class MarkingPattern(BaseModel):
    """One discriminating behavior in a grader's marking, with its evidence.
    Structured (not a bare string) so the contrastive analysis is auditable."""
    pattern: str                   # the marking behavior that moves the score
    evidence: str                  # what high- vs low-scoring essays do differently
    strength: Literal["strong", "moderate", "weak"]


class TeacherProfile(BaseModel):
    """Inspectable representation of a grader's standard. JSON, not an embedding —
    legibility is the whole point."""
    grader_id: str
    set_id: str
    rewards: list[MarkingPattern]  # what this grader consistently rewards
    penalizes: list[MarkingPattern]  # what this grader consistently penalizes
    severity: Literal["lenient", "balanced", "harsh"]
    trait_emphasis: dict[str, str] # trait name -> "high" | "medium" | "low"
    notes: str                     # free-text summary a human can read
    n_examples: int


# ---------- Output ----------

class Span(BaseModel):
    """A pointer into the essay that justifies a judgement."""
    quote: str                     # verbatim substring of the essay
    reason: str                    # why this span supports the score


class TraitJudgement(BaseModel):
    trait: str
    score: int
    spans: list[Span]              # must be non-empty — enforced below


class GradingResult(BaseModel):
    essay_id: str
    grader_id: str
    trait_judgements: list[TraitJudgement]
    holistic_score: int
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    summary: str

    @field_validator("trait_judgements")
    @classmethod
    def _traits_must_be_grounded(cls, v: list[TraitJudgement]):
        for tj in v:
            if not tj.spans:
                raise ValueError(
                    f"trait '{tj.trait}' has a score but no grounding spans"
                )
        return v


# ---------- Eval ----------

ReasoningLabel = Literal[
    "correct_score_grounded",     # right mark, reasoning points at real evidence
    "correct_score_ungrounded",   # right mark, wrong/absent reasons ("lucky")
    "wrong_systematic",           # wrong mark with a fixable pattern (e.g. length bias)
    "wrong_random",               # wrong mark, no pattern (the ceiling)
    "correct_abstention",         # said "unsure", was genuinely a hard case
]


class Adjudication(BaseModel):
    essay_id: str
    grader_id: str
    human_holistic: int
    predicted_holistic: int
    label: ReasoningLabel
    rationale: str

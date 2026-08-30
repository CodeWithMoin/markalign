"""Confidence gate. Kept separate so the abstention policy is tunable without
touching grading logic. Maps directly to the product constraint that teachers
keep judgement on the hard cases."""
from __future__ import annotations

from .schemas import GradingResult


def apply_gate(result: GradingResult, threshold: float = 0.6) -> GradingResult:
    """Flag a grade for human review when the model isn't confident enough."""
    if result.confidence < threshold:
        result.abstained = True
    return result

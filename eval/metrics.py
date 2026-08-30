"""Score-agreement metrics. Model-free and deterministic, so they can be re-run
a hundred times over frozen grading results without re-grading.

Quadratic Weighted Kappa (QWK) is the ASAP standard: it rewards being close, not
just exactly right, which is what you want for ordinal scores.
"""
from __future__ import annotations

from collections import defaultdict


def quadratic_weighted_kappa(y_true: list[int], y_pred: list[int],
                             lo: int | None = None, hi: int | None = None) -> float:
    assert len(y_true) == len(y_pred) and y_true
    lo = lo if lo is not None else min(min(y_true), min(y_pred))
    hi = hi if hi is not None else max(max(y_true), max(y_pred))
    n = hi - lo + 1
    if n <= 1:
        return 1.0

    O = [[0] * n for _ in range(n)]
    for a, b in zip(y_true, y_pred):
        O[a - lo][b - lo] += 1

    hist_t = [sum(O[i]) for i in range(n)]
    hist_p = [sum(O[i][j] for i in range(n)) for j in range(n)]
    total = len(y_true)

    num = den = 0.0
    for i in range(n):
        for j in range(n):
            w = ((i - j) ** 2) / ((n - 1) ** 2)
            e = hist_t[i] * hist_p[j] / total
            num += w * O[i][j]
            den += w * e
    return 1.0 - (num / den) if den else 1.0


def exact_and_adjacent(y_true: list[int], y_pred: list[int]) -> dict:
    exact = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    adj = sum(abs(a - b) <= 1 for a, b in zip(y_true, y_pred)) / len(y_true)
    return {"exact_agreement": round(exact, 3), "within_1": round(adj, 3)}


def calibration_by_confidence(rows: list[dict], bins: int = 5) -> list[dict]:
    """Are high-confidence grades actually more accurate? rows need keys:
    confidence, human, pred."""
    buckets = defaultdict(list)
    for r in rows:
        b = min(bins - 1, int(r["confidence"] * bins))
        buckets[b].append(abs(r["human"] - r["pred"]))
    out = []
    for b in range(bins):
        errs = buckets.get(b, [])
        out.append({
            "confidence_band": f"{b/bins:.1f}-{(b+1)/bins:.1f}",
            "n": len(errs),
            "mean_abs_error": round(sum(errs) / len(errs), 2) if errs else None,
        })
    return out


def confusion(y_true: list[int], y_pred: list[int], lo: int, hi: int) -> dict:
    m = {t: {p: 0 for p in range(lo, hi + 1)} for t in range(lo, hi + 1)}
    for a, b in zip(y_true, y_pred):
        m[a][b] += 1
    return m

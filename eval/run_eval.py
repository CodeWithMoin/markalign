"""Run the full evaluation and print the numbers that make the claims falsifiable.

  python -m eval.run_eval --mock                 # no API key, proves the pipeline
  python -m eval.run_eval                         # real grading (needs API key)
  python -m eval.run_eval --mock --adjudicate 40  # + reasoning taxonomy on a sample

What it reports, mapped to the three claims:
  1. "grades like the teacher"  -> QWK vs held-out human scores
  2. "personalizes to a grader" -> QWK on rater1 with rater1's profile vs rater2's
  3. "right for the right reason"-> reasoning taxonomy (sampled adjudication)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # keep the harness runnable without the optional dep
    def tqdm(it, **_):
        return it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.profile_builder import build_profile          # noqa: E402
from backend.grading_engine import grade                   # noqa: E402
from backend.abstention import apply_gate                  # noqa: E402
from eval import dataset, metrics, reasoning               # noqa: E402


def _split(records, k):
    return records[:k], records[k:]


def _pick_anchors(examples, n=4):
    """A few calibration essays spanning the score range, to anchor the grader's
    absolute scale in the grading prompt (not just the abstract profile)."""
    if len(examples) <= n:
        return examples
    ordered = sorted(examples, key=lambda e: e.holistic_score)
    step = (len(ordered) - 1) / (n - 1)
    return [ordered[round(i * step)] for i in range(n)]


def _grade_all(essays, rubric, profile, mock, threshold, anchors=None, workers=6):
    """Grade essays concurrently — each grade is an independent API call, so a
    thread pool cuts a real run from minutes to seconds. Failures are skipped so
    one bad model response can't sink a paid run."""
    def one(e):
        try:
            return apply_gate(grade(e, rubric, profile, anchors=anchors, mock=mock), threshold)
        except Exception as ex:
            print(f"  skip {e.essay_id}: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
            return None

    n = len(essays)
    if mock:  # deterministic + instant; no need for threads
        return [r for r in (one(e) for e in essays) if r is not None]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(one, essays), total=n,
                            desc="grading", unit="essay", file=sys.stderr))
    return [r for r in results if r is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the deterministic fake grader")
    ap.add_argument("--calib", type=int, default=8, help="essays used to infer the profile")
    ap.add_argument("--threshold", type=float, default=0.6, help="abstention confidence cutoff")
    ap.add_argument("--adjudicate", type=int, default=0, help="sample size for reasoning taxonomy")
    ap.add_argument("--max-test", type=int, default=0, dest="max_test",
                    help="cap held-out essays graded (0 = all) — controls cost/time of a real run")
    ap.add_argument("--data", default="data/sample_essays.json")
    ap.add_argument("--rubric", default="data/rubric.json")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    rubric = dataset.load_rubric(ROOT / args.rubric)
    records = dataset.load_records(ROOT / args.data)
    calib_rec, test_rec = _split(records, args.calib)
    if args.max_test:
        test_rec = test_rec[:args.max_test]

    # --- Profiles for both raters (from calibration split only) ---
    prof1 = build_profile("rater1", rubric, dataset.graded_examples(calib_rec, "rater1"), mock=args.mock)
    prof2 = build_profile("rater2", rubric, dataset.graded_examples(calib_rec, "rater2"), mock=args.mock)

    test_essays = dataset.essays(test_rec, rubric.set_id)
    human1 = {r["essay_id"]: r["rater1_holistic"] for r in test_rec}
    human2 = {r["essay_id"]: r["rater2_holistic"] for r in test_rec}

    # score anchors for rater1, from the calibration split only (test stays held out)
    anchors1 = _pick_anchors(dataset.graded_examples(calib_rec, "rater1"))

    # --- Claim 1: grade like rater1, measure vs rater1's held-out marks ---
    graded = _grade_all(test_essays, rubric, prof1, args.mock, args.threshold, anchors=anchors1)
    scored = [g for g in graded if not g.abstained]

    yt = [human1[g.essay_id] for g in scored]
    yp = [g.holistic_score for g in scored]
    qwk1 = metrics.quadratic_weighted_kappa(yt, yp, rubric.holistic_min, rubric.holistic_max)

    # --- Claim 2: personalization test — same grades, scored against the OTHER
    #     rater. If the profile is doing anything, it fits rater1 better. ---
    yp_same = yp
    yt_other = [human2[g.essay_id] for g in scored]
    qwk_same = metrics.quadratic_weighted_kappa(yt, yp_same, rubric.holistic_min, rubric.holistic_max)
    qwk_cross = metrics.quadratic_weighted_kappa(yt_other, yp_same, rubric.holistic_min, rubric.holistic_max)

    # calibration + confusion
    calib_rows = [{"confidence": g.confidence, "human": human1[g.essay_id],
                   "pred": g.holistic_score} for g in scored]

    report = {
        "config": {"mock": args.mock, "calib": args.calib, "threshold": args.threshold,
                   "n_test": len(test_essays), "n_scored": len(scored),
                   "n_abstained": len(graded) - len(scored)},
        "claim_1_grades_like_teacher": {
            "qwk_vs_rater1": round(qwk1, 3),
            **metrics.exact_and_adjacent(yt, yp),
        },
        "claim_2_personalization": {
            "qwk_fit_own_rater": round(qwk_same, 3),
            "qwk_fit_other_rater": round(qwk_cross, 3),
            "gap": round(qwk_same - qwk_cross, 3),
            "note": "gap is small BY CONSTRUCTION — the two raters share a rubric.",
        },
        "calibration_by_confidence": metrics.calibration_by_confidence(calib_rows),
        "confusion_vs_rater1": metrics.confusion(yt, yp, rubric.holistic_min, rubric.holistic_max),
    }

    # --- Claim 3: reasoning taxonomy on a sample ---
    if args.adjudicate:
        sample = scored[:args.adjudicate]
        text_by_id = {r["essay_id"]: r["text"] for r in test_rec}
        adjs = [reasoning.adjudicate(g, human1[g.essay_id], text_by_id[g.essay_id], mock=args.mock)
                for g in sample]
        report["claim_3_reasoning_agreement"] = reasoning.summarize(adjs)

    Path(ROOT / args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

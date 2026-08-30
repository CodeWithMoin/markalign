"""Fit a scale calibration on the CALIBRATION split and apply it to the test grades.

    python -m eval.calibrate --mock --data data/asap_set7_essays.json \
                             --rubric data/asap_set7_rubric.json --calib 25

The model can agree with the teacher's ORDERING while sitting on a different
scale (systematic offset/compression). That is a fixable error, so measure it:
fit `human ≈ a·model + b` on the calibration essays, then apply the map to the
confusion matrix already in --report. Re-grades only the calibration essays;
the held-out test grades are reused as-is.

Leakage note: the anchor essays are shown to the grader inside the prompt, so
its grades on them are not free predictions — they are excluded from the fit.
The test split is never fitted on, only transformed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.grading_engine import grade                  # noqa: E402
from backend.profile_builder import build_profile         # noqa: E402
from eval import dataset                                  # noqa: E402
from eval.metrics import quadratic_weighted_kappa         # noqa: E402
from eval.run_eval import _pick_anchors                   # noqa: E402


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env if present. Never prints values."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/asap_set7_essays.json")
    ap.add_argument("--rubric", default="data/asap_set7_rubric.json")
    ap.add_argument("--report", default="eval_report.json", help="run whose test grades to calibrate")
    ap.add_argument("--calib", type=int, default=25, help="calibration split size (must match the run)")
    ap.add_argument("--out", default="calibration_result.json")
    ap.add_argument("--mock", action="store_true", help="deterministic fake grader, no API calls")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")

    rubric = dataset.load_rubric(ROOT / args.rubric)
    calib_rec = dataset.load_records(ROOT / args.data)[:args.calib]
    examples = dataset.graded_examples(calib_rec, "rater1")
    profile = build_profile("rater1", rubric, examples, mock=args.mock)
    anchors = _pick_anchors(examples)
    anchor_ids = {a.essay_id for a in anchors}
    human = {r["essay_id"]: r["rater1_holistic"] for r in calib_rec}

    def one(e):
        try:
            return e.essay_id, grade(e, rubric, profile, anchors=anchors, mock=args.mock).holistic_score
        except Exception as ex:  # one bad response shouldn't sink the fit
            print(f"  skip {e.essay_id}: {type(ex).__name__}: {ex}", file=sys.stderr)
            return None

    essays = dataset.essays(calib_rec, rubric.set_id)
    print(f"grading {len(essays)} calibration essays…", flush=True)
    with ThreadPoolExecutor(max_workers=1 if args.mock else 6) as pool:
        pairs = [r for r in pool.map(one, essays) if r]

    fit = [(human[eid], pred) for eid, pred in pairs if eid not in anchor_ids]
    if len(fit) < 2:
        sys.exit("not enough non-anchor calibration grades to fit")
    ys = [h for h, _ in fit]
    xs = [m for _, m in fit]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 1.0
    b = my - a * mx
    print(f"fit on {len(fit)} calib essays (anchors excluded): human ~ {a:.2f}*model + {b:.2f}")

    def cal(m):
        return max(rubric.holistic_min, min(rubric.holistic_max, round(a * m + b)))

    cm = json.loads((ROOT / args.report).read_text())["confusion_vs_rater1"]
    H, M = [], []
    for h, row in cm.items():
        for m, cnt in row.items():
            H += [int(h)] * cnt
            M += [int(m)] * cnt
    Mc = [cal(m) for m in M]
    lo, hi = rubric.holistic_min, rubric.holistic_max
    out = {
        "a": a, "b": b,
        "raw_qwk": quadratic_weighted_kappa(H, M, lo, hi),
        "calibrated_qwk": quadratic_weighted_kappa(H, Mc, lo, hi),
        "mae_raw": sum(abs(h - m) for h, m in zip(H, M)) / len(H),
        "mae_cal": sum(abs(h - m) for h, m in zip(H, Mc)) / len(H),
    }
    print(f"\nTEST (n={len(H)}):")
    print(f"  raw QWK:        {out['raw_qwk']:.3f}   (mean abs err {out['mae_raw']:.2f})")
    print(f"  calibrated QWK: {out['calibrated_qwk']:.3f}   (mean abs err {out['mae_cal']:.2f})")
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

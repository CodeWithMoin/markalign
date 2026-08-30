"""Convert real ASAP-AES into this repo's records shape — one essay set at a time.

    python data/convert_asap.py training_set_rel3.tsv --set 1

Writes data/asap_set1_essays.json + data/asap_set1_rubric.json, then point the
eval at them:

    python -m eval.run_eval --data data/asap_set1_essays.json \
                            --rubric data/asap_set1_rubric.json

Why per-rater domain1 scores: ASAP gives each essay two pre-resolution human
raters (rater1_domain1, rater2_domain1). This repo treats those two humans as
two "graders" — that's the personalization test with no new annotation. We use
their INDIVIDUAL scores (not the resolved sum), so the holistic range is the
per-rater range, listed below.

Sets 1-6: one clean holistic score per rater, no per-rater trait breakdown, so
trait_scores is left empty (the headline QWK metric is holistic and unaffected).

Set 7 (recommended): the per-rater holistic (rater{1,2}_domain1, 0-12) is exactly
the sum of FOUR real per-rater trait scores (ideas / organization / style /
conventions, each 0-3). Both the holistic split AND real trait ground truth come
through — that's the richest set for this repo. Set 8's domain1 is not a clean
trait sum (weighted, 0-30), so it's skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Per-set: individual pre-resolution rater range for domain1, plus a short rubric.
# Ranges from the ASAP-AES data description (score of ONE rater, not the resolved sum).
SETS = {
    1: {"lo": 1, "hi": 6, "genre": "persuasive/argumentative",
        "prompt": "Write a letter to your local newspaper stating your opinion on the "
                  "effects computers have on people."},
    2: {"lo": 1, "hi": 6, "genre": "persuasive/argumentative",
        "prompt": "Write a persuasive essay to a newspaper reflecting your views on "
                  "censorship in libraries."},
    3: {"lo": 0, "hi": 3, "genre": "source-dependent response",
        "prompt": "Explain how the setting affects the cyclist, using details from the "
                  "essay 'ROUGH ROAD AHEAD'."},
    4: {"lo": 0, "hi": 3, "genre": "source-dependent response",
        "prompt": "Explain the ending of the story 'Winter Hibiscus', using details "
                  "from the passage."},
    5: {"lo": 0, "hi": 4, "genre": "source-dependent response",
        "prompt": "Describe the mood created by the author in the memoir, supporting "
                  "your answer with relevant details."},
    6: {"lo": 0, "hi": 4, "genre": "source-dependent response",
        "prompt": "Describe the obstacles the builders of the Empire State Building "
                  "faced in allowing dirigibles to dock, using the excerpt."},
    7: {"lo": 0, "hi": 12, "genre": "narrative",
        "prompt": "Write a story about a time when you were patient, or a time when "
                  "someone you know was patient.",
        # per-rater holistic (domain1) == sum of these four traits (each 0-3)
        "traits": [
            {"name": "ideas", "description": "Development, focus and detail of the story", "min_score": 0, "max_score": 3},
            {"name": "organization", "description": "Structure, sequencing and coherence", "min_score": 0, "max_score": 3},
            {"name": "style", "description": "Word choice, sentence fluency and voice", "min_score": 0, "max_score": 3},
            {"name": "conventions", "description": "Grammar, usage, spelling and mechanics", "min_score": 0, "max_score": 3},
        ],
        # trait name -> ASAP trait column index (rater{who}_trait{N})
        "trait_cols": {"ideas": 1, "organization": 2, "style": 3, "conventions": 4},
        # the human holistic IS the trait sum here, so the grader builds its holistic the same way
        "holistic_from_traits": True},
}

# Generic traits for sets 1-6 (no per-rater trait ground truth in those sets).
DEFAULT_TRAITS = [
    {"name": "content", "description": "Relevance, depth and quality of ideas and evidence", "min_score": 1, "max_score": 6},
    {"name": "organization", "description": "Structure, sequencing and coherence", "min_score": 1, "max_score": 6},
    {"name": "conventions", "description": "Grammar, spelling and mechanics", "min_score": 1, "max_score": 6},
]


def convert(tsv_path: Path, essay_set: int, limit: int | None) -> tuple[list[dict], dict]:
    if essay_set not in SETS:
        raise SystemExit(f"set {essay_set} not supported (use 1-7); set 8 domain1 isn't a clean trait sum")
    spec = SETS[essay_set]
    traits = spec.get("traits", DEFAULT_TRAITS)
    trait_cols = spec.get("trait_cols")  # None for sets 1-6

    def rater_traits(row, who):
        if not trait_cols:
            return {}
        return {name: int(row[f"{who}_trait{col}"]) for name, col in trait_cols.items()}

    records = []
    # ASAP ships as latin-1 (windows-1252) tab-separated, not utf-8.
    with tsv_path.open(encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["essay_set"]) != essay_set:
                continue
            r1, r2 = row.get("rater1_domain1"), row.get("rater2_domain1")
            if not (row.get("essay") and r1 and r2):
                continue  # skip rows missing text or either rater
            records.append({
                "essay_id": f"s{essay_set}_{row['essay_id']}",
                "text": row["essay"].strip(),
                "rater1_holistic": int(r1),
                "rater2_holistic": int(r2),
                "rater1_traits": rater_traits(row, "rater1"),
                "rater2_traits": rater_traits(row, "rater2"),
            })
            if limit and len(records) >= limit:
                break

    if not records:
        raise SystemExit(f"no rows found for set {essay_set} in {tsv_path}")

    rubric = {
        "set_id": f"asap-set-{essay_set}",
        "prompt": spec["prompt"],
        "traits": traits,
        "holistic_min": spec["lo"],
        "holistic_max": spec["hi"],
        "holistic_from_traits": spec.get("holistic_from_traits", False),
    }
    return records, rubric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path, help="path to training_set_rel3.tsv")
    ap.add_argument("--set", type=int, default=7, dest="essay_set", help="ASAP essay set (1-7; 7 has real traits)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of essays (0 = all)")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    args = ap.parse_args()

    records, rubric = convert(args.tsv, args.essay_set, args.limit or None)
    base = args.outdir / f"asap_set{args.essay_set}"
    (base.with_name(base.name + "_essays.json")).write_text(json.dumps(records, indent=2))
    (base.with_name(base.name + "_rubric.json")).write_text(json.dumps(rubric, indent=2))

    scores = [r["rater1_holistic"] for r in records]
    print(f"set {args.essay_set}: {len(records)} essays, "
          f"rater1 range {min(scores)}-{max(scores)} (rubric {rubric['holistic_min']}-{rubric['holistic_max']})")
    print(f"wrote {base.name}_essays.json and {base.name}_rubric.json")


def _selfcheck():
    """Tiny end-to-end check on a synthetic 2-row TSV — no real data needed."""
    import io, csv as _csv, tempfile, os
    rows = [
        {"essay_id": "1", "essay_set": "1", "essay": "hello world essay",
         "rater1_domain1": "4", "rater2_domain1": "5"},
        {"essay_id": "2", "essay_set": "2", "essay": "other set, ignore",
         "rater1_domain1": "3", "rater2_domain1": "3"},
    ]
    fd, path = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w", encoding="latin-1", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    recs, rub = convert(Path(path), 1, None)
    os.unlink(path)
    assert len(recs) == 1, recs                      # set filter works
    assert recs[0]["rater1_holistic"] == 4
    assert recs[0]["essay_id"] == "s1_1"
    assert rub["holistic_min"] == 1 and rub["holistic_max"] == 6
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()

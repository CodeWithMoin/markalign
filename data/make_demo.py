"""Rebuild data/demo.json — the preloaded landing fixture — from ASAP set 7.

    python data/make_demo.py

Picks ~9 real teacher-marked essays spanning the 0-12 holistic range (the
calibration standard) and 4 readable samples at distinct scores, cleans ASAP's
@ANONYMIZED tokens into plain English, and orders the samples by how well the
mock grader already agrees with the teacher, so the landing essay looks sane
with no API key. Committed so the fixture is reproducible, not magic.

Needs data/asap_set7_essays.json (Kaggle-gated, gitignored — see README).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.grading_engine import grade                 # noqa: E402
from backend.profile_builder import build_profile        # noqa: E402
from backend.schemas import Essay, GradedExample, Rubric, Trait  # noqa: E402

# ASAP replaces named entities with @TOKEN placeholders; left in, they read as
# noise to a human AND to the grader. Map them to neutral plain English.
SUBS = [
    (r"@CAPS\d+", ""), (r"@PERSON\d+", "a classmate"), (r"@LOCATION\d+", "a nearby city"),
    (r"@ORGANIZATION\d+", "a company"), (r"@NUM\d+", "many"), (r"@PERCENT\d+", "most"),
    (r"@DATE\d+", "one day"), (r"@MONTH\d+", "that month"), (r"@TIME\d+", "midday"),
    (r"@CITY\d+", "the city"), (r"@STATE\d+", "the state"), (r"@[A-Z]+\d+", ""),
]


# The ASAP TSV is Windows-1252 read as latin-1: curly quotes/dashes arrive as
# invisible control bytes (\x92 in "don\x92t" renders as "don t"). Map them back.
CP1252 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
          "\x96": "-", "\x97": "-", "\x85": "..."}


def clean(text: str) -> str:
    for bad, good in CP1252.items():
        text = text.replace(bad, good)
    for pat, rep in SUBS:
        text = re.sub(pat, rep, text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)      # ASAP tokenization: "turn ?" -> "turn?"
    text = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d|'m)\b", r"\1", text)  # "mom 's" -> "mom's"
    return re.sub(r"\s+", " ", text).strip()


def n_tokens(text: str) -> int:
    return len(re.findall(r"@[A-Z]+\d+", text))


def main():
    rubric_d = json.loads((ROOT / "data" / "asap_set7_rubric.json").read_text())
    records = json.loads((ROOT / "data" / "asap_set7_essays.json").read_text())

    # samples: readable length, distinct scores, least-anonymized first
    pool = sorted(
        (r for r in records if 140 <= len(clean(r["text"]).split()) <= 330),
        key=lambda r: n_tokens(r["text"]),
    )
    samples, seen = [], set()
    for r in pool:
        if r["rater1_holistic"] in seen:
            continue
        seen.add(r["rater1_holistic"])
        samples.append(r)
        if len(samples) == 4:
            break

    # calibration: ~9 essays spanning the full 0-12 range, samples excluded
    sample_ids = {r["essay_id"] for r in samples}
    calib = []
    for score in range(2, 13, 1):
        pick = next((r for r in pool if r["essay_id"] not in sample_ids
                     and r["rater1_holistic"] == score
                     and r["essay_id"] not in {c["essay_id"] for c in calib}), None)
        if pick:
            calib.append(pick)
    if len(calib) > 9:  # thin evenly, keeping both ends of the scale
        calib = [calib[round(i * (len(calib) - 1) / 8)] for i in range(9)]

    # order samples by mock agreement so the default landing essay looks sane
    rubric = Rubric(set_id=rubric_d["set_id"], prompt=rubric_d["prompt"],
                    traits=[Trait(**t) for t in rubric_d["traits"]],
                    holistic_min=rubric_d["holistic_min"], holistic_max=rubric_d["holistic_max"],
                    holistic_from_traits=rubric_d.get("holistic_from_traits", False))
    examples = [GradedExample(essay_id=f"cal{i}", text=clean(c["text"]),
                              trait_scores=c["rater1_traits"], holistic_score=c["rater1_holistic"])
                for i, c in enumerate(calib)]
    prof = build_profile("teacher-A", rubric, examples, mock=True)

    def rank(r):
        g = grade(Essay(essay_id=r["essay_id"], set_id=rubric.set_id, text=clean(r["text"])),
                  rubric, prof, mock=True)
        return (abs(g.holistic_score - r["rater1_holistic"]), -g.confidence)

    samples.sort(key=rank)

    demo = {
        "rubric": rubric_d,
        "calibration": [{"text": e.text, "holistic_score": e.holistic_score,
                         "trait_scores": e.trait_scores} for e in examples],
        "samples": [{"essay_id": r["essay_id"], "text": clean(r["text"]),
                     "human_holistic": r["rater1_holistic"]} for r in samples],
    }
    out = ROOT / "data" / "demo.json"
    out.write_text(json.dumps(demo, indent=2))
    print(f"wrote {out}: {len(demo['calibration'])} calibration, {len(demo['samples'])} samples "
          f"(scores {[s['human_holistic'] for s in demo['samples']]})")


if __name__ == "__main__":
    main()

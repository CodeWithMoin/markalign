"""FastAPI service. Serves the grading endpoint and the single-page frontend as
one deployable unit (one service, one deploy → one live link).

Set EDEXIA_MOCK=1 to run the demo with no API key (deterministic fake grader).
Set ANTHROPIC_API_KEY to grade for real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .schemas import Rubric, Trait, GradedExample, Essay
from .profile_builder import build_profile
from .grading_engine import grade
from .abstention import apply_gate

ROOT = Path(__file__).resolve().parent.parent
MOCK = os.environ.get("EDEXIA_MOCK", "0") == "1"

app = FastAPI(title="Edexia Alignment Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# tiny in-process cache: grader_id -> profile
_PROFILES: dict = {}


class CalibExample(BaseModel):
    text: str
    holistic_score: int
    trait_scores: dict[str, int]


class GradeRequest(BaseModel):
    grader_id: str
    rubric: Rubric
    calibration: list[CalibExample]
    essay_text: str
    threshold: float = 0.6


@app.get("/health")
def health():
    return {"ok": True, "mock": MOCK}


@app.post("/grade")
def grade_endpoint(req: GradeRequest):
    if not req.calibration:
        raise HTTPException(400, "need at least one calibration example")

    examples = [
        GradedExample(essay_id=f"c{i}", text=c.text,
                      trait_scores=c.trait_scores, holistic_score=c.holistic_score)
        for i, c in enumerate(req.calibration)
    ]
    # rebuild profile if calibration changed
    key = (req.grader_id, len(examples))
    if key not in _PROFILES:
        _PROFILES[key] = build_profile(req.grader_id, req.rubric, examples, mock=MOCK)
    profile = _PROFILES[key]

    essay = Essay(essay_id="live", set_id=req.rubric.set_id, text=req.essay_text)
    result = apply_gate(grade(essay, req.rubric, profile, mock=MOCK), req.threshold)

    return {"profile": profile.model_dump(), "result": result.model_dump()}


# ---- Preloaded demo: a real teacher + real ASAP essays, so a visitor sees a
#      graded essay with evidence on arrival — no data entry. ----
DEMO = json.loads((ROOT / "data" / "demo.json").read_text())
_rb = DEMO["rubric"]
DEMO_RUBRIC = Rubric(set_id=_rb["set_id"], prompt=_rb["prompt"],
                     traits=[Trait(**t) for t in _rb["traits"]],
                     holistic_min=_rb["holistic_min"], holistic_max=_rb["holistic_max"],
                     holistic_from_traits=_rb.get("holistic_from_traits", False))
_DEMO_EXAMPLES = [GradedExample(essay_id=f"cal{i}", text=c["text"],
                  trait_scores=c.get("trait_scores", {}), holistic_score=c["holistic_score"])
                  for i, c in enumerate(DEMO["calibration"])]
# a few calibration essays spanning the range, to anchor the grader's absolute scale
_DEMO_ANCHORS = sorted(_DEMO_EXAMPLES, key=lambda e: e.holistic_score)
_DEMO_ANCHORS = [_DEMO_ANCHORS[0], _DEMO_ANCHORS[len(_DEMO_ANCHORS) // 2], _DEMO_ANCHORS[-1]]
_demo_cache: dict = {}


def _demo_profile():
    """Learn the teacher's standard once from the fixture's calibration set, cached."""
    if "profile" not in _demo_cache:
        _demo_cache["profile"] = build_profile("teacher-A", DEMO_RUBRIC, _DEMO_EXAMPLES, mock=MOCK)
    return _demo_cache["profile"]


class OwnGrade(BaseModel):
    essay_text: str
    essay_id: str | None = None
    threshold: float = 0.6


@app.get("/demo")
def demo():
    """The preloaded teacher standard + real sample essays for the landing view."""
    prof = _demo_profile()
    return {
        "profile": prof.model_dump(),
        "threshold": 0.6,
        "holistic_max": DEMO_RUBRIC.holistic_max,
        "trait_max": max(t.max_score for t in DEMO_RUBRIC.traits),
        "prompt": DEMO_RUBRIC.prompt,
        "samples": [{"essay_id": s["essay_id"], "text": s["text"],
                     "human_holistic": s["human_holistic"]} for s in DEMO["samples"]],
    }


@app.post("/grade_own")
def grade_own(req: OwnGrade):
    """Grade any essay (a sample or a pasted one) against the preloaded teacher."""
    prof = _demo_profile()
    essay = Essay(essay_id=req.essay_id or "live", set_id=DEMO_RUBRIC.set_id, text=req.essay_text)
    result = apply_gate(grade(essay, DEMO_RUBRIC, prof, anchors=_DEMO_ANCHORS, mock=MOCK), req.threshold)
    human = next((s["human_holistic"] for s in DEMO["samples"] if s["essay_id"] == req.essay_id), None)
    return {"result": result.model_dump(), "human_holistic": human}


@app.get("/eval")
def eval_summary():
    """Latest frozen eval report, if one has been generated. Powers the
    'how aligned is it' scorecard so the substance is on the page, not just
    the terminal. Run `python -m eval.run_eval` to (re)generate it."""
    live = ROOT / "eval_report.json"
    snap = ROOT / "results" / "eval_report_real.json"
    report = source = None
    if live.exists():
        d = json.loads(live.read_text())
        if d.get("config", {}).get("mock") is False:   # a mock run must never pose as the headline
            report, source = d, "live"
    if report is None and snap.exists():
        report, source = json.loads(snap.read_text()), "snapshot"
    if report is None:
        return {"available": False}

    out = {"available": True, "source": source, **report}
    cal = ROOT / "calibration_result.json" if source == "live" else ROOT / "results" / "calibration_real.json"
    if not cal.exists():
        cal = ROOT / "results" / "calibration_real.json"
    if cal.exists():
        out["calibration"] = json.loads(cal.read_text())
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "frontend" / "index.html").read_text()

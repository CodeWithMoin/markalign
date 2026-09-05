"""FastAPI service. Serves the grading endpoint and the single-page frontend as
one deployable unit (one service, one deploy → one live link).

Set EDEXIA_MOCK=1 to run the demo with no API key (deterministic fake grader).
Set ANTHROPIC_API_KEY to grade for real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .schemas import Rubric, Trait, GradedExample, Essay
from .profile_builder import build_profile
from .grading_engine import grade
from .abstention import apply_gate

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env if present (provider url/key/model), so
    `make demo-real` works without exporting anything. setdefault → shell wins.
    Never prints values."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
MOCK = os.environ.get("EDEXIA_MOCK", "0") == "1"

app = FastAPI(title="Edexia Alignment Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# tiny in-process cache: grader_id -> profile
_PROFILES: dict = {}

# ---- usage guards: a public link must not be an open tap on the API key.
#      Cached sample grades cost nothing and are never limited; these gates only
#      cover grades that reach the model. In-process state suits the one-worker
#      deploy; a multi-worker deploy would move this to a shared store. ----
MAX_ESSAY_CHARS = 10_000                     # ~3x the longest real ASAP set-7 essay
RATE_LIMIT = 6                               # live grades per IP...
RATE_WINDOW = 10 * 60                        # ...per 10 minutes
DAILY_CAP = int(os.environ.get("EDEXIA_DAILY_CAP", "200"))  # hard wallet ceiling
_hits: dict[str, list[float]] = {}
_day = {"date": "", "n": 0}


# ---- optional sign-in gate (Supabase + Google OAuth): the browser does the
#      OAuth dance with supabase-js; the server only checks the bearer token
#      against Supabase's auth API. Unset SUPABASE_URL → gate off (local dev,
#      mock mode) and everything behaves as before. ----
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


def _verify_user(req: Request) -> str | None:
    """Return the signed-in user's id, or raise 401. None when the gate is off."""
    if MOCK or not SUPABASE_URL:
        return None
    token = (req.headers.get("authorization") or "").removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "sign in with Google to grade your own essay — "
                                 "the preloaded sample essays need no account")
    import urllib.request as _ur
    r = _ur.Request(f"{SUPABASE_URL}/auth/v1/user",
                    headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY})
    try:
        with _ur.urlopen(r, timeout=10) as resp:
            user = json.loads(resp.read())
        if not user.get("id"):
            raise ValueError("no user id")
        return user["id"]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "your sign-in expired — sign in again to grade your own essay")


def _guard_live_grade(req: Request, text: str, key: str | None = None) -> None:
    """Raise 4xx before any model call: size, per-user/IP rate, global daily budget."""
    if len(text) > MAX_ESSAY_CHARS:
        raise HTTPException(413, f"essay too long ({len(text)} chars; limit {MAX_ESSAY_CHARS})")
    if MOCK:
        return                               # mock grading is free — no limits
    ip = key or (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "?")).split(",")[0].strip()
    now = time.time()
    recent = [t for t in _hits.get(ip, []) if now - t < RATE_WINDOW]
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(429, "rate limit: a few live grades per 10 minutes — "
                                 "the preloaded sample essays are unlimited")
    today = time.strftime("%Y-%m-%d")
    if _day["date"] != today:
        _day.update(date=today, n=0)
    if _day["n"] >= DAILY_CAP:
        raise HTTPException(429, "daily live-grading budget reached — try tomorrow, "
                                 "or explore the preloaded graded essays")
    recent.append(now)
    _hits[ip] = recent
    _day["n"] += 1


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
def grade_endpoint(req: GradeRequest, request: Request):
    if not req.calibration:
        raise HTTPException(400, "need at least one calibration example")
    _guard_live_grade(request, req.essay_text)

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


_PROFILE_PATH = ROOT / "data" / "demo_profile.json"
_GRADES_PATH = ROOT / "data" / "demo_grades.json"
# sample grades, frozen to disk like the profile: a click on a preloaded essay
# never spends a model call, and the grade never drifts between restarts.
_GRADES: dict = (json.loads(_GRADES_PATH.read_text())
                 if not MOCK and _GRADES_PATH.exists() else {})


def _demo_profile():
    """The teacher's standard, learned ONCE and frozen to disk. Rebuilding on every
    restart re-rolls the model and the demo's grades drift — a vetted profile is a
    versioned artifact, not a dice throw. Delete data/demo_profile.json to relearn."""
    if "profile" not in _demo_cache:
        from .schemas import TeacherProfile
        if not MOCK and _PROFILE_PATH.exists():
            _demo_cache["profile"] = TeacherProfile(**json.loads(_PROFILE_PATH.read_text()))
        else:
            # checklist=True: per-trait yes/no ladders; the grading engine counts
            # yeses — the format that scored QWK 0.656 in the eval.
            prof = build_profile("teacher-A", DEMO_RUBRIC, _DEMO_EXAMPLES,
                                 mock=MOCK, checklist=True)
            if not MOCK:
                _PROFILE_PATH.write_text(json.dumps(prof.model_dump(), indent=2))
            _demo_cache["profile"] = prof
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
def grade_own(req: OwnGrade, request: Request):
    """Grade any essay (a sample or a pasted one) against the preloaded teacher.
    Sample essays are graded once and cached — a reload or shared link shows the
    same grade instantly instead of re-rolling the model."""
    human = next((s["human_holistic"] for s in DEMO["samples"] if s["essay_id"] == req.essay_id), None)
    is_sample = req.essay_id is not None and human is not None
    if is_sample and req.essay_id in _GRADES:
        return {"result": _GRADES[req.essay_id], "human_holistic": human}
    # live grading costs money: require sign-in (when configured) and rate-limit
    # per user rather than per IP; frozen samples above stay open to everyone
    user = _verify_user(request)
    _guard_live_grade(request, req.essay_text, key=user)
    prof = _demo_profile()
    essay = Essay(essay_id=req.essay_id or "live", set_id=DEMO_RUBRIC.set_id, text=req.essay_text)
    try:
        result = apply_gate(grade(essay, DEMO_RUBRIC, prof, anchors=_DEMO_ANCHORS, mock=MOCK), req.threshold)
    except HTTPException:
        raise
    except Exception as ex:   # model/provider failure → clean JSON error, not a raw 500
        raise HTTPException(502, f"live grading is unavailable right now ({type(ex).__name__}) — "
                                 "the preloaded sample essays still work")
    out = result.model_dump()
    if is_sample and not MOCK:
        _GRADES[req.essay_id] = out
        _GRADES_PATH.write_text(json.dumps(_GRADES, indent=2))
    return {"result": out, "human_holistic": human}


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
    # Attach calibration ONLY when it actually improves QWK for the served report.
    # (On the current luna headline it hurts — 0.478 < 0.585 — so it's omitted.)
    cal = ROOT / "calibration_result.json" if source == "live" else ROOT / "results" / "calibration_luna.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        raw_qwk = report.get("claim_1_grades_like_teacher", {}).get("qwk_vs_rater1")
        if raw_qwk is not None and c.get("calibrated_qwk", 0) > raw_qwk:
            out["calibration"] = c
    return out


@app.get("/llms.txt")
def llms_txt():
    """Machine-readable site summary (llms.txt convention) — so an AI asked about
    this link answers with the measured numbers instead of guessing."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse((ROOT / "llms.txt").read_text())


@app.get("/", response_class=HTMLResponse)
def index():
    # stamp the task into the HTML so the banner renders in the first paint
    # instead of popping in after the /demo fetch; auth config is public by design
    # (Supabase anon keys are meant for browsers) and empty when the gate is off
    return ((ROOT / "frontend" / "index.html").read_text()
            .replace("{{TASK}}", DEMO_RUBRIC.prompt)
            .replace("{{SB_URL}}", "" if MOCK else SUPABASE_URL)
            .replace("{{SB_KEY}}", "" if MOCK else SUPABASE_ANON_KEY))

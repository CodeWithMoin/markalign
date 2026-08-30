"""One thin seam over the model call. Everything above it is provider-agnostic.

Two modes:
  - real:  calls the Anthropic API. Needs ANTHROPIC_API_KEY.
  - mock:  a deterministic fake grader. No key needed. Lets the eval harness run
           end-to-end so the QWK pipeline is provable before you spend a cent,
           and lets CI stay green.

The mock is intentionally *biased* (slightly harsh on short essays) so the
reasoning-agreement taxonomy has a real "wrong_systematic" pattern to find.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional


def _model() -> str:
    # Read at CALL time, not import time — otherwise a .env loaded after this
    # module is imported (e.g. eval/calibrate.py) is silently ignored.
    return os.environ.get("EDEXIA_MODEL", "claude-sonnet-4-5")


def _stable_unit(*parts: str) -> float:
    """Deterministic pseudo-random in [0,1) from the inputs — same essay always
    grades the same way in mock mode, so eval numbers are reproducible."""
    h = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _openai_compatible(system: str, user: str) -> dict:
    """Generic OpenAI-compatible chat endpoint (NVIDIA NIM, OpenAI, DeepSeek,
    local Ollama — same protocol, different base_url). Selected by setting
    EDEXIA_PROVIDER_URL (+ EDEXIA_PROVIDER_KEY, EDEXIA_MODEL) in the env.
    stdlib-only so it adds no dependency."""
    import urllib.request

    # Stream (SSE): reasoning models can think for a long time, and a streamed
    # response keeps the socket alive so there's no idle read-timeout.
    url = os.environ["EDEXIA_PROVIDER_URL"]
    payload = {
        "model": _model(),
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": True,
    }
    if "openai.com" in url:                 # gpt-5.x: reasoning model, different rules
        payload["max_completion_tokens"] = 16000   # reasoning tokens + JSON
        payload["reasoning_effort"] = "low"         # grading needs little; keeps it fast
    else:                                    # NVIDIA NIM / DeepSeek / Ollama / etc.
        payload["max_tokens"] = 8192
        payload["temperature"] = 0
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        os.environ["EDEXIA_PROVIDER_URL"], data=body,
        headers={"Authorization": f"Bearer {os.environ.get('EDEXIA_PROVIDER_KEY', '')}",
                 "Content-Type": "application/json", "Accept": "text/event-stream"})
    chunks = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            if not choices:                  # keepalive / role-only chunks have empty choices
                continue
            piece = choices[0].get("delta", {}).get("content")  # ignore reasoning_content
            if piece:
                chunks.append(piece)
    text = "".join(chunks)
    # Reasoning models may inline <think>…</think>; models may fence the JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def complete_json(system: str, user: str, *, mock: bool = False,
                  mock_kind: str = "grade", ctx: Optional[dict] = None) -> dict:
    """Return a parsed JSON object from the model (or the mock)."""
    if mock:
        return _mock(mock_kind, ctx or {})

    if os.environ.get("EDEXIA_PROVIDER_URL"):      # any OpenAI-compatible provider
        return _openai_compatible(system, user)

    # Default real path: Anthropic. Imported lazily so mock mode needs no dependency.
    import anthropic

    # Identity-linked keys require naming the workspace the request bills to.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": ws} if ws else None)
    msg = client.messages.create(
        model=_model(),
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


# ---------- Mock implementations ----------

def _mock(kind: str, ctx: dict) -> dict:
    if kind == "profile":
        sev = ["lenient", "balanced", "harsh"][int(_stable_unit(ctx.get("grader_id", "")) * 3)]
        return {
            "rewards": [
                {"pattern": "specific evidence tied to the claim", "strength": "strong",
                 "evidence": "[mock] high essays explain how evidence proves the point; low ones just cite"},
                {"pattern": "controlled structure", "strength": "moderate",
                 "evidence": "[mock] high essays signpost; low ones jump between ideas"},
            ],
            "penalizes": [
                {"pattern": "vague generalisation", "strength": "strong",
                 "evidence": "[mock] low essays assert without support"},
            ],
            "severity": sev,
            "trait_emphasis": {t: "high" if i == 0 else "medium"
                               for i, t in enumerate(ctx.get("traits", []))},
            "notes": f"[mock] inferred from {ctx.get('n', 0)} examples; leans {sev}.",
        }

    if kind == "grade":
        text = ctx.get("text", "")
        traits = ctx.get("traits", [])
        lo = ctx.get("holistic_min", 2)
        hi = ctx.get("holistic_max", 12)
        # Baseline from a stable hash of the essay...
        base = _stable_unit(ctx.get("essay_id", ""), text[:80])
        holistic = round(lo + base * (hi - lo))
        # ...with a deliberate, discoverable bias: dock short essays.
        if len(text.split()) < 120:
            holistic = max(lo, holistic - 2)
        # Cite a DISTINCT clause per trait so the marked-up essay shows several
        # real highlights, not the same words three times.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        sentences = sentences or [text[:80] or "the response"]
        tj = []
        for i, t in enumerate(traits):
            tlo, thi = ctx.get("trait_min", 1), ctx.get("trait_max", 6)
            ts = max(tlo, min(thi, round(tlo + base * (thi - tlo))))
            s = sentences[i % len(sentences)]
            clause = s if len(s.split()) <= 26 else " ".join(s.split()[:22])  # keep whole short sentences
            tj.append({
                "trait": t,
                "score": ts,
                "spans": [{"quote": clause,
                           "reason": f"Where this response shows its {t} level."}],
            })
        return {
            "trait_judgements": tj,
            "holistic_score": holistic,
            "confidence": round(0.5 + base * 0.4, 2),
            "summary": "[mock] deterministic grade for pipeline testing.",
        }

    if kind == "adjudicate":
        # Label the disagreement. Mock keys off the true gap + length so the
        # taxonomy is populated meaningfully.
        gap = abs(ctx.get("human", 0) - ctx.get("pred", 0))
        short = ctx.get("short", False)
        if gap == 0:
            label = "correct_score_grounded" if ctx.get("grounded", True) else "correct_score_ungrounded"
        elif short:
            label = "wrong_systematic"
        else:
            label = "wrong_random"
        return {"label": label, "rationale": f"[mock] gap={gap}, short={short}"}

    raise ValueError(f"unknown mock kind: {kind}")

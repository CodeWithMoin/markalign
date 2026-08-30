# Alignment — measuring whether an AI grader agrees for the *right reason*

The hard problem in AI assessment isn't producing a score. Anything can output a
number. The hard problem is telling a grader that **works** from one that
**looks like it works** — whether it matched the teacher for the right reason, or
just got lucky. That's the core risk for anyone putting AI in front of student
work, and it's what this system measures.

A grader is included, but it is **scaffold, not the point** — it learns one
teacher's marking standard, grades a new essay, and shows the exact spans behind
every score, so that the measurement has something real to measure. The
measurement is the deliverable:

- **Score-agreement** — did it match the teacher's held-out marks? (QWK)
- **Reasoning-agreement** — was a *right* score right for the right reason, and
  when it was *wrong*, was the error a **fixable systematic pattern** or **random
  noise**?

Separating those two questions — and separating systematic bias from random
noise inside the second — is the same instinct as telling a real signal from an
artifact. It's the judgment the rest of the repo is built to make legible.

> Stack mirrors a typical AI-native assessment backend: **Python + FastAPI**,
> Pydantic-typed contracts, a model adapter, and a model-free eval harness.

---

## Run it in 30 seconds (no API key)

```bash
pip install -r requirements.txt
EDEXIA_MOCK=1 uvicorn backend.app:app --reload
# open http://localhost:8000
```

Mock mode uses a deterministic fake grader so the whole thing runs with no key.
It carries a **deliberate bias** (harsh on short essays) so the eval has a real
systematic error to catch. Grade the sample essay, then shorten it and watch the
score drop and the confidence gate abstain.

### Grade for real

```bash
export ANTHROPIC_API_KEY=sk-...
uvicorn backend.app:app --reload          # EDEXIA_MOCK unset → real grading
```

---

## The eval (where the judgment shows)

```bash
# prove the pipeline, no key (deterministic fake grader)
python -m eval.run_eval --mock --data data/asap_set7_sample.json --rubric data/asap_set7_rubric.json

# real numbers on real ASAP set 7 (needs ANTHROPIC_API_KEY; --max-test caps cost)
python -m eval.run_eval --data data/asap_set7_essays.json --rubric data/asap_set7_rubric.json \
                        --calib 25 --max-test 120

python -m eval.run_eval --mock --adjudicate 40   # + reasoning taxonomy (method demo)
```

It reports three things, each mapped to a claim:

| Claim | How it's checked | Metric |
|---|---|---|
| Grades like the teacher | vs held-out human marks | **QWK** (the ASAP standard) |
| Personalizes to a grader | fit own rater vs the *other* rater | QWK gap |
| Right for the right reason | sampled hand/model adjudication | reasoning taxonomy |

The reasoning taxonomy splits wrong grades into **systematic** (a fixable
pattern, e.g. length bias) vs **random** (the ceiling) — the same disentangling
instinct as separating a real signal from an artifact.

---

## Results (real, on ASAP set 7)

Real run: `claude-sonnet-4-5`, 120 held-out essays, a 25-essay calibration set.
**Read every number against the human ceiling, not against 1.0:** two trained
raters on this set agree at **QWK 0.72**. That's the bar.

| | QWK vs the teacher | mean error (0–12 scale) |
|---|---|---|
| Grader as-is | **0.40** | 2.9 pts |
| + scale calibration | **0.47** (ceiling ~0.65) | 1.7 pts |
| Human vs human (ceiling) | 0.72 | — |

**The number isn't the point — the diagnosis is.** A 0.40 looks mediocre, so the
eval asks *why*. The confusion matrix shows the grader **ranks essays correctly**
(its mean score rises monotonically with the human's) but scores **systematically
harsh and compressed** — it never uses the top of the scale. That error is
**systematic, not random**, so it's fixable: a scale transform fit on the
calibration set (applied to held-out test) nearly halves the error. This is the
whole thesis in one run — *telling a grader that works from one that only looks
like it, then locating the fixable part.*

**Grader-quality probe (directional, n=10):** a stronger model (Opus) grading the
same essays **blind** with a few anchors reached **QWK 0.74** (≈ the human
ceiling), mean error 1.5. Small sample — treat as a signal, not a headline — but
strong evidence the gap above is **model + calibration, not a task ceiling**.

**Personalization:** own-rater vs other-rater gap **+0.02** — small *by
construction* (the two raters share a rubric), which is the honest expected result.

> Why it matters here: moving beyond standardised senior English into per-department,
> Years 7–10 assessment means learning **each grader's** standard from a handful of
> examples — and expanding *without losing trust*. This measures exactly that: can it
> learn one grader's standard, and can you tell when it's off *and why*.

---

## Architecture

```
ingestion → normalized store
          → profile builder (per grader, cached)  ─┐
                                                    ▼
   essay + rubric ─────────────► grading engine → GradingResult (typed, span-grounded)
                                                    │
                                          abstention gate
                                     ┌──────────────┴──────────────┐
                                     ▼                             ▼
                              served result                  eval harness
                                                        (QWK · calibration ·
                                                         reasoning taxonomy)
```

Three seams carry the design:

- **Profile is a cacheable artifact, not a runtime step** → ablate personalization by swapping one input (own profile / none / the wrong grader's).
- **`GradingResult` is the contract** → span-grounding and confidence are schema validation, not prose you hoped for. An ungrounded trait score fails to construct.
- **Metrics layer is model-free** (except sampled adjudication) → reproducible over frozen results; score-agreement and reasoning-agreement never entangled.

```
backend/   schemas · model_adapter · profile_builder · grading_engine · abstention · app
eval/      dataset · metrics · reasoning · run_eval
data/      convert_asap.py · real ASAP set 7 (two raters, real traits) + synthetic fallback
frontend/  single dependency-free HTML file (served by the API)
```

**On the frontend stack:** it's a single dependency-free HTML file on purpose, so
the demo runs as one FastAPI service with no build step and no second deploy
target. Production would be **Next.js + TypeScript + Tailwind** to match your
stack — that's overhead a one-service demo doesn't need, and the `/grade` JSON
contract is the clean seam to swap it later.

---

## Real ASAP-AES data (wired, not hypothetical)

`data/convert_asap.py` converts the public ASAP-AES benchmark into the records
shape the eval expects (`essay_id, text, rater1_holistic, rater2_holistic,
rater1_traits, rater2_traits`):

```bash
python data/convert_asap.py training_set_rel3.tsv --set 7
python -m eval.run_eval --data data/asap_set7_essays.json \
                        --rubric data/asap_set7_rubric.json --calib 25 --max-test 120
```

**Set 7** is used because its per-rater holistic (0–12) is exactly the sum of four
real per-rater trait scores (ideas / organization / style / conventions) — so both
the two-rater personalization split **and** real trait ground truth come through.
The two pre-resolution raters become two "graders": a real personalization test
with no new annotation. Human inter-rater agreement on this set is **QWK ≈ 0.72** —
that's the ceiling any grader is measured against, not 1.0.

The dataset is Kaggle-gated, so `training_set_rel3.tsv` isn't committed; a small
converted sample is checked in so the pipeline is provably real offline.

---

## Honest limits (read before believing any number)

- **Real ASAP ≠ your distribution.** Score-agreement numbers are real (ASAP set 7, held-out marks), but ASAP isn't VCE/HSC/IB. The claim is *the method transfers*, never *it works on a specific curriculum's distribution*. Read the human ceiling with it: two trained raters agree at QWK ≈ 0.72, so a grader is measured against 0.72, not 1.0.
- **The two raters share a rubric**, so the personalization gap is small **by construction**. A small effect is the expected, honest result; a large one would be suspicious.
- **Confidence is anti-calibrated, so the abstention gate is currently useless.** On the real run the model never reported confidence below the 0.6 threshold (0/120 abstentions), and its more-confident grades were *worse*: mean absolute error 3.13 in the 0.8–1.0 band vs 2.25 in the 0.6–0.8 band. The gate works, the signal driving it does not. Calibrating confidence — not just scores — is the next piece of work.
- **Reasoning-agreement is a method, not a validated result.** There is no reasoning ground truth in ASAP, so the taxonomy rests on hand-adjudication against documented labelling criteria — its credibility is in those criteria, not in a number. Score-agreement is the number you can trust; reasoning-agreement is the framework for the harder question.

---

## Deploy (one service, one link)

Render: push to GitHub, "New Web Service", point at the repo. `render.yaml` is
included; it starts in mock mode so the link is live immediately. Add
`ANTHROPIC_API_KEY` and set `EDEXIA_MOCK=0` to grade for real.

Or anywhere that runs a Procfile:
`uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.

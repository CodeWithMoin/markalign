"""The load-bearing checks: the metric, the grounding contract, and the mock path.

    python3 test_alignment.py        # no pytest, no fixtures, exits non-zero on failure
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from backend.grading_engine import grade
from backend.schemas import Essay, GradingResult, Rubric, TeacherProfile, Trait
from eval.metrics import quadratic_weighted_kappa as qwk

# 1. perfect agreement
assert qwk([1, 2, 3, 4], [1, 2, 3, 4], 1, 4) == 1.0

# 2. hand-computable case, scores in 1..2 (n=2, so the only weight is w[0][1]=w[1][0]=1).
#    y_true=[1,1,2,2], y_pred=[1,2,1,2] -> O off-diagonal = 2 of 4 -> num = 2.
#    marginals: hist_t=[2,2], hist_p=[2,2]; E[0][1]=E[1][0]=2*2/4=1 -> den = 2.
#    kappa = 1 - 2/2 = 0.0 (chance-level agreement).
assert qwk([1, 1, 2, 2], [1, 2, 1, 2], 1, 2) == 0.0

# 3. far disagreement must cost more than near disagreement
y = [1, 2, 3, 4, 5, 6]
near = [v + 1 if v < 6 else v for v in y]
far = [min(6, v + 3) for v in y]
assert qwk(y, near, 1, 6) > qwk(y, far, 1, 6)

# 4. an ungrounded trait judgement is a validation failure, not prose we tolerate
try:
    GradingResult(essay_id="e", grader_id="g", holistic_score=3, confidence=0.5,
                  summary="s", trait_judgements=[{"trait": "ideas", "score": 2, "spans": []}])
    raise AssertionError("empty spans should not validate")
except ValidationError:
    pass

# 5. the ASAP converter's own selfcheck still passes
assert subprocess.run([sys.executable, str(ROOT / "data" / "convert_asap.py"), "--selfcheck"],
                      cwd=ROOT).returncode == 0

# 6. holistic_from_traits: the model's holistic is built the same way as the label
rubric = Rubric(set_id="t", prompt="p", holistic_min=0, holistic_max=12,
                holistic_from_traits=True,
                traits=[Trait(name=n, description=n, min_score=0, max_score=3)
                        for n in ("ideas", "organization", "style", "conventions")])
profile = TeacherProfile(grader_id="g", set_id="t", rewards=[], penalizes=[],
                         severity="balanced", trait_emphasis={}, notes="", n_examples=0)
essay = Essay(essay_id="e1", set_id="t", text="A patient day. " * 60)
res = grade(essay, rubric, profile, mock=True)
assert res.holistic_score == sum(j.score for j in res.trait_judgements)
assert all(0 <= j.score <= 3 for j in res.trait_judgements)

print("all checks passed")

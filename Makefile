# Alignment — one-word commands.
# Real targets read provider config (url/key/model) from .env automatically.
.PHONY: install test demo demo-real eval eval-real calibrate clean

PORT ?= 8000
SET7 := --data data/asap_set7_essays.json --rubric data/asap_set7_rubric.json
SAMPLE := --data data/asap_set7_sample.json --rubric data/asap_set7_rubric.json

install:          ## install python deps
	pip install -r requirements.txt

test:             ## run the assert-based checks (no key, no network)
	python3 test_alignment.py

demo:             ## run the web app in MOCK mode — free, no key → http://localhost:$(PORT)
	EDEXIA_MOCK=1 uvicorn backend.app:app --port $(PORT) --reload

demo-real:        ## run the web app with REAL grading (uses .env: gpt-5.6-luna) → http://localhost:$(PORT)
	EDEXIA_MOCK=0 uvicorn backend.app:app --port $(PORT) --reload

eval:             ## prove the eval pipeline on the committed real sample — MOCK, free
	python3 -m eval.run_eval --mock $(SAMPLE) --calib 8

eval-real:        ## real eval on ASAP set 7, 120 held-out essays, checklist grading (uses .env; costs API)
	python3 -m eval.run_eval $(SET7) --calib 25 --max-test 120 --checklist

eval-direct:      ## same eval with direct 0-N rating instead of the yes/no checklist
	python3 -m eval.run_eval $(SET7) --calib 25 --max-test 120

calibrate:        ## fit + apply scale calibration to the latest eval_report.json (uses .env; costs API)
	python3 -m eval.calibrate $(SET7) --report eval_report.json --calib 25

clean:            ## remove generated eval artifacts (committed snapshots in results/ are kept)
	rm -f eval_report.json calibration_result.json

# Short names for common commands. Adapted from ig/main (Ishank), Makefile.
# The baseline and classify targets spend HF credits: run them only when the team asked for that run.
PY := .venv/bin/python
RUN ?= runs/dry

.PHONY: setup test dry-run baseline classify

setup:
	./scripts/setup_tau2.sh

test:
	$(PY) -m pytest -q

dry-run:
	$(PY) scripts/baseline.py --dry-run --num-tasks 2 --output-dir $(RUN)

baseline:
	$(PY) scripts/baseline.py --num-tasks $(or $(N),5)

classify:
	$(PY) scripts/classify.py $(RUN)

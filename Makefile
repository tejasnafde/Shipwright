# Short names for common commands. Adapted from ig/main (Ishank), Makefile.
# The baseline and classify targets spend HF credits: run them only when the team asked for that run.
PY := .venv/bin/python
RUN ?= runs/dry

.PHONY: setup test dry-run agent-r-dry baseline classify control-room

setup:
	./scripts/setup_tau2.sh

test:
	$(PY) -m pytest -q

dry-run:
	$(PY) scripts/baseline.py --dry-run --num-tasks 2 --output-dir $(RUN)

agent-r-dry:
	$(PY) scripts/agent_r_search.py --dry-run --user-model dry-user --task-ids 0 1 --simulations 8 --output-dir $(RUN)/agent-r-search
	$(PY) scripts/agent_r_revise.py $(RUN)/agent-r-search --dry-run --output-dir $(RUN)/agent-r-revise
	$(PY) scripts/build_sft.py --agent-r $(RUN)/agent-r-revise --estimate-tokens --output-dir $(RUN)/agent-r-sft

baseline:
	$(PY) scripts/baseline.py --num-tasks $(or $(N),5)

classify:
	$(PY) scripts/classify.py $(RUN)

control-room:
	$(PY) -m streamlit run src/shipwright/control_room/app.py

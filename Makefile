# Short names for common commands. `baseline` calls model APIs and spends credits; the others are free.
#
#   make demo     everything free in one go: install, tests, a whole round with fake models, the published numbers
#                 rechecked from the public run files, then the control room on round 16 (http://localhost:8501)
#   make check    the same without opening the control room
PY := .venv/bin/python
RUN ?= runs/dry
# uv installs itself into ~/.local/bin; put it on the PATH for every recipe below.
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: demo check uv setup test dry-run dry-round reproduce pull-round-16 control-room-deps baseline classify control-room

demo: check pull-round-16 control-room-deps
	@echo "Opening the control room on http://localhost:8501 (Ctrl-C to stop)."
	$(PY) -m streamlit run src/shipwright/control_room/app.py

check: setup test dry-round reproduce
	@echo "All free checks passed: tests, a dry-run round, and every headline number recomputed from the public runs."

uv:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv (https://docs.astral.sh/uv)"; curl -LsSf https://astral.sh/uv/install.sh | sh; }

setup: uv
	./scripts/setup_tau2.sh

test:
	$(PY) -m pytest -q

# A whole round (baseline, classify, teacher, data, train, gate) with fake models: free, offline, about a minute.
dry-round:
	.venv/bin/shipwright run --config configs/rounds/default.yaml --dry-run

# The write-up's headline numbers, recomputed from the public runs dataset (about 2 MB, no token, no GPU).
reproduce:
	$(PY) scripts/reproduce.py

# Round 16's gate results (the unattended promotion) from the public runs dataset into runs/gate-r16.
pull-round-16:
	$(PY) -c "from pathlib import Path; from shipwright.control_room import hub_runs; print(hub_runs.pull_gate_results(16, Path('runs')))"

# The control room's one extra dependency (streamlit), kept out of the harness install.
control-room-deps:
	uv pip install -q --python $(PY) -e ".[space]"

dry-run:
	$(PY) scripts/baseline.py --dry-run --num-tasks 2 --output-dir $(RUN)

baseline:
	$(PY) scripts/baseline.py --num-tasks $(or $(N),5)

classify:
	$(PY) scripts/classify.py $(RUN)

control-room: control-room-deps
	$(PY) -m streamlit run src/shipwright/control_room/app.py

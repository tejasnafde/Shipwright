# control-room, the read-only demo UI (infra/space/README.md). Deliberately its own, minimal image: no GPU, no
# agenttune/auditkit, no secrets - it only reads the run-dir artifacts the
# harness scripts already produced (baseline.py, classify.py, tool_fix.py,
# round_status.py, gate_eval.py) and renders them. Meant to run as a Hugging
# Face Space (Docker SDK), which expects the app on 7860.
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/shipwright/control_room/ src/shipwright/control_room/
COPY src/shipwright/__init__.py src/shipwright/__init__.py
RUN pip install --no-cache-dir ".[space]"
# The demo ships with the artifacts from a real or dry-run round baked in -
# control-room is read-only and has no way to produce its own, see
# infra/space/README.md for how this directory gets populated before build.
COPY runs/ runs/
# The tool-patch ledger, read by the Fix & Train tab (patches active in / proposed from a run).
COPY patches/ patches/

ENV PYTHONPATH=/app/src

EXPOSE 7860

CMD ["streamlit", "run", "src/shipwright/control_room/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]

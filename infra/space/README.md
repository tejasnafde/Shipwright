# Deploying control-room as a Hugging Face Space

## Why a Space, not a Job

Cloud Run Jobs and HF Jobs are for the compute-and-cost services (agent runs, training, gate
evals) - they run once per round and stop. control-room is the opposite: a small, always-available,
read-only UI for judges to open on Demo Day, which is exactly what a Space is for. No GPU, no
secrets, no AgentTune/AuditKIT - it only reads the run-dir artifacts the harness scripts already
produced (`docs/system-design.html`, "Demo: the loop, round by round").

## Populate the artifacts before building

`infra/space/Dockerfile` copies `runs/` into the image (control-room cannot produce its own
artifacts - it is read-only by design). Before building or pushing, make sure `runs/` holds the
run dir(s) you want the demo to show - a real round's output (`scripts/baseline.py`,
`scripts/classify.py`, `scripts/round_status.py`, `scripts/gate_eval.py`, in that order, per
`docs/pipeline/baseline.md`/`docs/pipeline/gate.md`), or a `--dry-run`/`--no-llm`/`--local-dry-run` pass of the same
scripts for a quick offline check with no model calls.

## Build (context is the repo root, same as the main harness image)

```sh
docker build -f infra/space/Dockerfile -t shipwright-control-room:latest .
docker run -p 7860:7860 shipwright-control-room:latest   # open http://localhost:7860
```

## Create the Space (Docker SDK)

Hugging Face Spaces need a `README.md` with a YAML metadata header at the Space repo's root. This
project's own root `README.md` is the GitHub one; the Space gets its own copy with this header
prepended when pushed - do not overwrite the main repo README with it.

```yaml
---
title: Shipwright Control Room
emoji: 🛟
sdk: docker
app_port: 7860
pinned: false
---
```

```sh
# first time: create the Space
hf repo create <org>/shipwright-control-room --type space --space_sdk docker

# push: the Space repo's root needs Dockerfile, the metadata README above, and
# everything infra/space/Dockerfile's COPY lines reference (src/shipwright/control_room/,
# src/shipwright/__init__.py, runs/) - a small sync script or a dedicated git remote
# for the Space repo is the usual pattern, not committing Space-specific files into
# this repo's own root.
```

The exact push mechanics (a separate git remote pointed at the Space repo vs. the `huggingface_hub`
Python API's `upload_folder`) are a one-time setup step once someone is ready to actually publish
the demo - this file documents the shape, not a tested end-to-end push, since that needs a real
`HF_TOKEN` this environment does not have.

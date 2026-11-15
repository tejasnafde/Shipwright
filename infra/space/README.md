# Control room image

The control room (`src/shipwright/control_room/`) is a read-only Streamlit replay of a round: run, detect, classify,
route, fix, train, gate, outcome, task by task. It never calls a model, a trainer or the gate; it only reads run
directories. This image packages it with a fixed set of runs, for hosting as a Docker Hugging Face Space. It contains
no GPU code, no AgentTune or AuditKIT, and no secrets.

## Build and run

`infra/space/Dockerfile` copies `runs/` into the image, so pull the rounds to show first
(`scripts/pull_runs.py`, and `hub_runs.pull_gate_results` for gate results). The build context is the repository root.

```sh
docker build -f infra/space/Dockerfile -t shipwright-control-room .
docker run -p 7860:7860 shipwright-control-room      # http://localhost:7860
```

Without Docker: `streamlit run src/shipwright/control_room/app.py`.

## Publish as a Space

A Docker Space needs the Dockerfile, the paths its `COPY` lines reference, and a `README.md` with this header at the
Space repository's root (not in this repository's root):

```yaml
---
title: Shipwright Control Room
sdk: docker
app_port: 7860
---
```

Upload those files with `huggingface_hub.upload_folder` to a Space created with `hf repo create <name> --type space
--space_sdk docker`.

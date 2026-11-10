# Hugging Face setup for the Lexsi hackathon

Give this whole file to your agent, same as `gcp-setup.md`.

## Accounts and tokens

```sh
pip install -U huggingface_hub
huggingface-cli login   # paste a token with write access, from huggingface.co/settings/tokens
```

Store the token the same way as any other secret (see `gcp-setup.md` section 5, GCP Secret Manager) —
never in code, never in a notebook cell, never committed.

## Accept the Tiny Aya gate before anything else

Tiny Aya is a gated model. Visit `huggingface.co/CohereLabs/tiny-aya-global` (and any other Tiny Aya
variant you use) while logged in and accept the terms. A job that downloads it before the gate is
accepted fails with a 403, not a clear "please accept the license" message — if you hit a mysterious
403 pulling the model, this is almost always why.

## HF Jobs

HF Jobs run training and gate evals; output lands on the Hub, billed from HF credits.

- **A job stops after 30 minutes by default.** Always pass a timeout explicitly:
  `hf jobs run --timeout 2h ...` on the CLI, or `timeout=` in the Python client. This has already cost
  time once (see `AGENTS.md` known traps, 2026-11-08) — do not skip it.
- A job uses the credits of the namespace it runs in (your personal namespace vs. a team namespace,
  if one exists) — check which one before kicking off anything long.
- Prices as of the hackathon: T4 $0.40/h, L4 $0.80/h, A10G small $1.00/h. Budget a round's train + gate
  eval cost before starting it, the same way GCP Spot cost is budgeted in `gcp-setup.md`.

No custom Docker image: `scripts/train_job.py --launch` and `scripts/gate_job.py --launch` submit
through `huggingface_hub.run_uv_job`, which runs the target script (`scripts/train_lora.py`,
`scripts/gate_eval.py --in-job`) as a uv script - its own inline `# /// script` dependency block
(torch, agenttune, etc.) is installed at job start, not baked into an image. Always a plan-only
dry run first (the default, no `--launch`), which prints the flavor price x timeout before
anything is billed:

```sh
.venv/bin/python scripts/train_job.py --flavor l4x1 --timeout 2h ...       # plan only
.venv/bin/python scripts/train_job.py --flavor l4x1 --timeout 2h --launch ...
```

Live agent runs that collect traces (not training/gate evals) go through Cloud Run Jobs instead,
built with Cloud Build from `infra/docker/Dockerfile` - see `docs/cloud-runs.md`, not this file.

## Public artifacts: two ungated collections

Everything public at the end goes into exactly two HF collections, both ungated: one for models
(adapters, one tag per round), one for data (traces, curated datasets, benchmarks). Create them once:

```sh
# via the Hub UI: huggingface.co/collections/new
# or programmatically:
python -c "
from huggingface_hub import create_collection
create_collection(title='Shipwright models', namespace='<your-hf-username-or-org>')
create_collection(title='Shipwright data', namespace='<your-hf-username-or-org>')
"
```

`AuditKIT`'s `RunResult.push_to_hub(repo_id, private=False)` writes directly to the Hub with a
provenance/dataset-card file — add the resulting repo to the right collection by hand (or via
`add_collection_item`) after each push, it does not join a collection automatically. `sft.jsonl`
(`scripts/build_sft.py`) and trace run dirs push the same way, by hand, for now.

## Dataset downloads AuditKIT needs

`ak.load_bfcl(...)` pulls `gorilla-llm/Berkeley-Function-Calling-Leaderboard` from the Hub at eval time
— no separate setup beyond being logged in, but it is a real download each time unless cached locally
(`HF_HOME` / `datasets` cache), worth knowing before running the gate repeatedly during debugging.

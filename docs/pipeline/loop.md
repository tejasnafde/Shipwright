# The self-healing loop, unattended (`scripts/run_loop.py`)

One command takes the last gate refusal and keeps going, round after round, until a candidate is promoted, the money
runs out, or no remedy is left. Nobody has to watch it: every stop condition is in code, every step is resumable.

```sh
.venv/bin/python scripts/doctor.py                      # free: tokens, configs, data, every command the loop builds
.venv/bin/python scripts/doctor.py --cloud              # also a vLLM start with the adapter on an HF GPU (about $0.1)
.venv/bin/python scripts/run_loop.py --round 6 --from-round 5 --budget-usd 3.8 --max-rounds 2 \
    --failures-run r4-cand-train --screen-bfcl --gate-flavor a10g-large              # plan only: nothing is run
.venv/bin/python scripts/run_loop.py ...same... --launch                              # runs, spends
```

## What you need

An HF token with Jobs and Inference Providers access (`HF_TOKEN` in `.env`), the team's gcloud config (`lexsi`) for Vertex
(the teacher) and GCS, and the repo pushed (the cloud image and the gate bundle are built from HEAD). `doctor.py` checks
all of it and says which is missing. `run_loop.py --launch` runs the free checks first and starts nothing if one fails.

## What it does in a round

It reads the last gate decision (`gate_decision.json` on the Hub), chooses a remedy for what failed
(`src/shipwright/heal/recipe.py`, a ladder written down with its evidence), runs only the steps that remedy needs, gates
the result and either promotes it (`scripts/promote.py`, with rollback) or starts the next round from the new refusal.

| What failed | Remedy | Steps it runs | About |
|---|---|---|---|
| BFCL fell (the adapter answers in words where a function should be called) | 100 general function-calling examples per category in the data | data, train, BFCL screen, gate | $2.3 |
| looping, over-acting, context overflow, pass^k | the tool guards on (`src/shipwright/llm/guards.py`): a repeated lookup, an exchange to another product type and a second change to one order are rejected, and answered in words instead | gate only (the same adapter) | $0.8 |
| still failing with the guards on | the teacher's corrections of the candidate's wrong turns (`scripts/build_corrections.py`) | corrections, data, train, BFCL screen, gate | $2.7 |
| nothing left to try | the loop stops and says so | | |

`--screen-bfcl` scores BFCL alone on the candidate first (about 8 minutes) and stops if the gate's BFCL check could not
pass, so the 30 minute gate is only paid for when that check can.

## What keeps it reliable

- **Preflight** (`scripts/doctor.py`): every integration that broke a real run is a check.
- **Resumable**: each step is written to `runs/round-N.json` as it finishes; the same command resumes after a stop.
- **GPU queue**: a job that waits more than 12 minutes for a GPU is cancelled (nothing billed) and tried on the next type.
- **Budget**: a step starts only if its estimate fits in what is left of `--budget-usd`; each HF job has a time limit.
  A job is priced by its running seconds, not the time it waited.
- **Report**: `runs/loop-report-round-N.json` has every round's recipe, steps, decision, spend, the git sha, library
  versions, the frozen configs' hashes and the seed.
- **The gate is not touched.** `configs/gate.yaml` and `configs/splits.yaml` are frozen; the loop only reads them.

## Reading the result

`streamlit run src/shipwright/control_room/app.py`, pick the round: the first tab says in words what got better and worse.

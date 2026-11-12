# One round, end to end (`scripts/run_round.py`)

One command per stage, in order. Each stage has a cost estimate and a hard cap, and writes its result
to `runs/round-N.json` (run dir, exit code, estimate, actual cost, `cost_basis`: metered or estimated). Read the manifest, not the console.

| Stage | Where | What | Est. | Cap |
|---|---|---|---|---|
| `baseline` | Cloud Run job | Tiny Aya, no adapter, no tool patches, 74 train tasks, 1 trial | $1.50 | $2.00 |
| `classify` | Cloud Run job | label and route every failure | $0.05 | $0.30 |
| `teacher` | Cloud Run job | teacher replays the tool-fix and weight-fix failures, 3 trials; passes kept | $1.60 | $2.00 |
| `data` | laptop | `build_sft.py` from the teacher passes, pushed to the HF dataset repo | 0 | 0 |
| `train` | HF Job, L40S, 1h | LoRA with AgentTune (`--launch` to submit); an L4 was too tight for 185 examples | $1.00 | $1.80 |
| `ceiling` | Cloud Run job | side stage, any time: the teacher on the 20 held-out tasks, 3 trials (evaluation only) | $0.50 | $1.00 |
| `gate` | HF Job, A10G large, 90 min | base vs the adapter, 20 held-out tasks, k=3, HF router user simulator (`--launch` to submit) | $2.30 | $3.00 |

```sh
ARGS="--round 2 --user-model vertex/openai/gpt-oss-120b-maas --teacher-model vertex/deepseek-ai/deepseek-v4.1-flash-maas"
.venv/bin/python scripts/run_round.py plan $ARGS          # free: the table above, with what is done
.venv/bin/python scripts/run_round.py baseline $ARGS      # then classify, teacher, data, train, gate
```

Round 3 and later: a new prompt means a fresh `baseline`, but the teacher's passing traces from earlier rounds
stay valid training data (the SFT builder re-renders them with the current student prompt). Pool them, then
harvest more: a task the teacher has solved once passes again about 48% of the time (34% over all tasks), so
replaying solved tasks is the cheapest way to more passing traces.

```sh
POOL="r1-teacher-t1 r1-teacher-all-t2 r1-teacher-ds-a r1-teacher-ds-b r1-teacher-ds-c r2-teacher bake-dsv41flash"
.venv/bin/python scripts/run_round.py teacher $ARGS --only-solved-from $POOL --teacher-trials 5 --min-teacher-rate 0.5   # harvest the gap tasks first
.venv/bin/python scripts/run_round.py data $ARGS --extra-teacher-runs $POOL   # this round's passes + the pool
```

Other teacher modes: `--skip-solved-from $POOL` plays only tasks with no pass yet, `--skip-unsolved-from $POOL`
drops tasks it failed 3 decided trials in a row. The harvest is the main one for more data.

`--only-model` (set by the driver to the round's teacher) drops pooled runs written by another teacher, because
we have not checked the license of the others. On the existing runs the DeepSeek pool is 140 SFT examples from 47
tasks, against 21 from 13 in round 2 alone.

`--min-teacher-rate 0.5` keeps the gap tasks (the teacher mostly passes, the student fails: 22 of them on the
round 2 data) and plays the best first, so the cost cap cuts the weakest. Drop it for the wider harvest.

The `ceiling` stage runs the teacher on the held-out tasks once. Its pass rate is the ceiling the student is read
against (student pass^1 as a share of the teacher's). It is evaluation only: `build_sft.py` refuses a test-split run.

## After the gate: promote or roll back

```sh
.venv/bin/python scripts/promote.py promote runs/<gate dir>/gate_decision.json   # only a "promote" decision with an adapter
.venv/bin/python scripts/promote.py rollback                                       # one step back, rows are never deleted
.venv/bin/python scripts/promote.py show
```

`promotions/ledger.jsonl` is the record; `configs/systems/current.yaml` is written from it and becomes the `--current`
of the next round's gate, so every round has to beat the adapter that is deployed. The gate also prints pass^1 per
task slice (`0-1 writes`, `2+ writes`, from the task's expected actions, `heal/slices.py`) next to the decision:
context for where the weights helped, never part of the decision.

Not built yet: a baseline of the deployed adapter on the train tasks. The HF router cannot serve an adapter, so it
needs a GPU job (vLLM, like the gate). Until then round N+1 takes its failures from the base model's baseline.

## Checks the driver makes for you

- Before a cloud stage: the tree is clean, HEAD is pushed, and the job runs the image of HEAD. Right after a
  push it waits (`--wait-for-image-minutes`, default 20) for the Cloud Build to tag the image and repoint the job.
- Before a stage: the previous stage finished with exit code 0, had at most 25% infra errors, and (for `train`)
  the SFT data has at least 40 examples before `train`. `--force` skips these two health checks.
- `runs/_meta/` holds the logs of the driver, `train_job.py`, `gate_job.py` and `pull_runs.py`: they launch
  and move runs and have no run output of their own, so they stay out of `runs/`.

## How the cost stays under the estimate

- API stages pass their cap to the script as `--max-cost-usd`. The run meters its own tokens and, past
  the cap, every further model call raises and the remaining tasks end as infra errors
  (`cost_cap_hit: true` in `summary.json`). Calls in flight finish, so the final cost can pass the cap by cents.
- GPU stages are bounded by `price x timeout`, which `train_job.py` and `gate_job.py` print before `--launch`.
- The manifest splits `spent_usd` into `spent_metered_usd` (the stage measured its own cost) and `spent_estimated_usd`
  (GPU jobs and unmetered stages count their estimate until you read the bill).
- A stage does not start if its cap plus the spend so far passes `--round-budget-usd` (default $8; the whole team has $10, so check what others are spending first).
- Tiny Aya's price on the HF router is an assumption (`price_assumed` in `summary.json`). After `baseline`,
  read the real spend on the HF billing page and correct `BUDGETS` in the script if it differs.

## Rules that keep rounds comparable

- The base system has no tool patches. The weight candidate (`configs/systems/r2-weight.yaml`) has none
  either, so a gain is the weights. The tool fix is a separate candidate (`r1-tool.yaml`).
- Same prompt, same user model, same teacher and the frozen splits for the whole round.
- Cloud stages run on the owner's job (`--owner`, default `ishank`) and upload to
  `gs://lexsi-hackathon-runs/<owner>/`.

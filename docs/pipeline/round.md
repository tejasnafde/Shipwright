# One round, end to end (`scripts/run_round.py`)

One command per stage, in order. Each stage has a cost estimate and a hard cap, and writes its result
to `runs/round-N.json` (run dir, exit code, estimate, actual cost). Read the manifest, not the console.

| Stage | Where | What | Est. | Cap |
|---|---|---|---|---|
| `baseline` | Cloud Run job | Tiny Aya, no adapter, no tool patches, 74 train tasks, 1 trial | $1.50 | $2.00 |
| `classify` | Cloud Run job | label and route every failure | $0.05 | $0.30 |
| `teacher` | Cloud Run job | teacher replays the weight-fix failures, 3 trials; passes kept | $1.60 | $2.00 |
| `data` | laptop | `build_sft.py` from the teacher passes, pushed to the HF dataset repo | 0 | 0 |
| `train` | HF Job, L4, 1h | LoRA with AgentTune (`--launch` to submit) | $0.40 | $0.80 |
| `gate` | HF Job, A10G large, 1h | base vs the adapter, 20 held-out tasks, k=3 (`--launch` to submit) | $1.80 | $2.10 |

```sh
ARGS="--round 2 --user-model vertex/openai/gpt-oss-120b-maas --teacher-model vertex/deepseek-ai/deepseek-v4.1-flash-maas"
.venv/bin/python scripts/run_round.py plan $ARGS          # free: the table above, with what is done
.venv/bin/python scripts/run_round.py baseline $ARGS      # then classify, teacher, data, train, gate
```

## How the cost stays under the estimate

- API stages pass their cap to the script as `--max-cost-usd`. The run meters its own tokens and, past
  the cap, every further model call raises and the remaining tasks end as infra errors
  (`cost_cap_hit: true` in `summary.json`). Calls in flight finish, so the final cost can pass the cap by cents.
- GPU stages are bounded by `price x timeout`, which `train_job.py` and `gate_job.py` print before `--launch`.
- A stage does not start if its cap plus the spend so far passes `--round-budget-usd` (default $8; the whole team has $10, so check what others are spending first).
- Tiny Aya's price on the HF router is an assumption (`price_assumed` in `summary.json`). After `baseline`,
  read the real spend on the HF billing page and correct `BUDGETS` in the script if it differs.

## Rules that keep rounds comparable

- The base system has no tool patches. The weight candidate (`configs/systems/r2-weight.yaml`) has none
  either, so a gain is the weights. The tool fix is a separate candidate (`r1-tool.yaml`).
- Same prompt, same user model, same teacher and the frozen splits for the whole round.
- The cloud job must run the image built from HEAD: the driver checks this and refuses a dirty tree.
- Cloud stages run on the owner's job (`--owner`, default `ishank`) and upload to
  `gs://lexsi-hackathon-runs/<owner>/`.

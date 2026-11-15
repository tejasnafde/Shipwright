# Hugging Face

Shipwright uses Hugging Face for the student model (Inference Providers), the GPUs (HF Jobs) and every published
artifact (Hub repos).

## Token and model access

- One token with write access to the repos below and the Jobs permissions (read and write): a token that can list jobs
  but lacks `job.write` fails at launch with `403 missing permissions`. `shipwright doctor` checks this.
- The code reads it from `$SWITCHBOARD_SECRETS_DIR/HF_TOKEN`, then `$HF_TOKEN`; in Cloud Run it is the
  `SHIPWRIGHT_HF_TOKEN` secret. It is never logged (the logger redacts `hf_` values).
- Tiny Aya is gated: accept its terms on `huggingface.co/CohereLabs/tiny-aya-global` with the token's account. A
  download before that fails with a bare 403.

## Inference Providers

The student (`openai/CohereLabs/tiny-aya-global:cohere`) and the user simulator
(`openai/openai/gpt-oss-120b:deepinfra`) are called through the OpenAI-compatible router
`https://router.huggingface.co/v1`, billed to the token's account.

## HF Jobs

GPU work runs as HF Jobs launched with `huggingface_hub.run_uv_job` (`src/shipwright/loop/jobs.py`): the target script
is a uv script whose inline dependency block (torch, vLLM, AgentTune, AuditKIT) is installed at job start, so there is
no custom image.

| Job | Script | Flavor | Timeout |
|---|---|---|---|
| Train | `scripts/train_lora.py` | `l40sx1` ($1.80/h) | 45 minutes |
| Gate | `scripts/gate_eval.py --in-job` | `a10g-large` ($1.50/h) | 60 minutes |
| Candidate on train tasks | `scripts/gate_eval.py --train-eval` | `a10g-large` | 40 minutes |
| BFCL screen | `scripts/gate_eval.py --bfcl-screen` | `a10g-large` | 30 minutes |
| vLLM smoke test | `scripts/vllm_smoke.py` | `a10g-large` | 25 minutes |

- **Always pass a timeout**: a job stops after 30 minutes by default and pushes nothing. Size a timeout to the work,
  never to a budget; the loop's caps bound the spend.
- A job is billed for its running seconds, not its time in the queue. The loop cancels a job queued more than 12
  minutes and tries the next flavor in its config.
- The repository is private, so a launch uploads a code bundle of the committed tree to the runs dataset; commit
  before launching. Job arguments never end in `.py`, `.sh`, `.yaml`, `.yml` or `.toml`: `run_uv_job` treats those as
  local files to upload.
- Prices per hour are in `src/shipwright/cloud.py` (`GPU_PRICES`).

## Hub repositories

| Repo | Type | Content |
|---|---|---|
| `mitushigoyal5/shipwright-lora` | model | one LoRA adapter per round, on branch `round-N`; `round-16` is deployed |
| `mitushigoyal5/shipwright-sft` | dataset | the training data of each round, on branch `round-N`, with `sft_stats.json` |
| `mitushigoyal5/shipwright-runs` | dataset | every gate's runs and decision, code bundles, `metrics.jsonl`, the promotion ledger |

All three are public and grouped into two ungated collections (models, data), linked from `README.md`.
BFCL v3 is read from `gorilla-llm/Berkeley-Function-Calling-Leaderboard` at a pinned revision (`configs/gate.yaml`).

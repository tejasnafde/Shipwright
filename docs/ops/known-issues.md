# Known issues and how the service handles them

Each entry is a failure mode that broke a real run, what causes it, and where the fix lives. Most are also checked by
`shipwright doctor` before any paid run.

## The student model (Tiny Aya Global)

| Symptom | Cause | Handling |
|---|---|---|
| The router returns no `tool_calls` | Tiny Aya has no native tool calling; its template and the HF router drop `tools` | Prompted JSON tool calls (`llm/prompted_tools.py`); training data uses the same format |
| Requests rejected over 8,192 tokens | the context is 8,192 tokens and the provider counts template tokens we do not | `llm/context.py`: 512-token reply reserve, a 250-token tokenizer margin, old results and replies shortened before turns are dropped |
| The agent asks an identified customer to authenticate again | the turn with the `find_user_id` result was dropped from the window | `context.still_true_note` keeps the user and order ids in a trimmed request (system switch `identity_note`) |
| The agent uses ids such as `#W0000000` | it copies example values from tool descriptions and prompt examples | examples that look real are replaced with hints (`prompted_tools.example_hint`); prompt examples use `<...>` placeholders. Never put a real-looking id in a prompt |
| `get_product_details(<item id>)` fails repeatedly | item ids of a product already fetched are sent to the product lookup | `guards.repaired_call` reroutes a call whose id the conversation only knows as another kind, if the agent was shown that tool |
| Long lookup chains fill the window | the student keeps looking up where the base gives up | the `lookup_streak` guard (no 11th lookup without a reply), compact observations (`OBSERVATION_VERSION` 3), the `condensed_policy` switch |

Runs made before a prompt or observation format change are not comparable with runs after it; `run_config.json` records
`tool_list_version`, `observation_version` and every serving switch.

## tau2-bench

| Symptom | Cause | Handling |
|---|---|---|
| A run crashes on one bad tool call | tau2 runs `json.loads` on every argument string | `llm/patches.py` wraps malformed arguments so the environment returns a tool error |
| Calls to models that do not exist | tau2 calls its own default models (NL judge and others) | `scripts/baseline.py` reroutes them to the user model and counts them |
| The policy alone is not the agent prompt | tau2 appends a language clause at run time | each run records the real prompt in `prompts.json`; training data is rendered from it |
| Results land in tau2's own data dir | `save_to` is relative | `baseline.py` passes an absolute path |
| SFT rows fail "Conversation roles must alternate" | tau2 opens with a fixed agent greeting, so rows read system, assistant, user | `prompted_tools.without_opening_greeting` drops it for the student prompt and for training rows |

## The user simulator (gpt-oss-120b)

| Symptom | Cause | Handling |
|---|---|---|
| The conversation ends right after "yes, go ahead" | the simulator writes past its turn (`<|end|><|start|>assistant...`) and plays both sides | `patches.cut_leaked_turn` cuts at the first marker; an empty remainder is asked again |
| The customer hangs up mid-task | a bare `###STOP###` whose reasoning shows no intent to end | `patches.is_unexplained_stop` asks again |
| The customer speaks as the agent | the simulator answers in the agent's voice | `patches.speaks_as_agent` (always on) and `is_agent_voiced_reply` (switch `agent_voice_check`, set on both systems of a gate) ask again |
| Empty replies crash tau2 | "UserMessage must have either content or tool_calls" | asked again up to twice |

The gate uses the HF router (DeepInfra) gpt-oss for the simulator: it had no turn leaks where Vertex did, and has no
Vertex rate limit or token expiry.

## Model APIs

| Symptom | Cause | Handling |
|---|---|---|
| 429s from Vertex gpt-oss at 4 to 6 concurrent runs | rate limits | `vertex.call_with_refresh`: waits with jitter (429), new token (401), retries a 400 for user and judge calls, two timeout retries |
| Router 429, 5xx or timeouts end a run | no retry on the router path | `patches.call_with_transient_retries` waits 2, 5, 10, 20 s |
| Calls hang for minutes | no client timeout | 60 s (gpt-oss) or 180 s timeouts on Vertex calls |
| The classifier silently falls back to rules | AgentTune's `classify_batch` sends every failure at once and hits 429s | `agenttune_bridge.make_classifier` allows 4 at a time; check `per_source` in `failure_summary.json` |
| A Vertex call from an HF Job fails after about an hour | an HF Job cannot run gcloud; the token minted at launch expires | the gate job only serves and runs conversations; classification runs in Cloud Run |
| DeepSeek V4.1 Flash call fails in `us-central1` | the model is served only on the `global` endpoint | `vertex.py` uses `global` |
| A 30-minute teacher job dies on one 429 | two jobs calling the same model at once | a failed call skips its turn and answers are cached (`asked.jsonl`); do not run two DeepSeek jobs at once |

## HF Jobs, vLLM and training

| Symptom | Cause | Handling |
|---|---|---|
| A job stops after 30 minutes with nothing pushed | the HF Jobs default timeout | every launch passes a timeout; a gate needs 45 to 50 minutes of wall time, and a round config refuses a gate timeout under 60 minutes |
| `403 missing permissions: job.write` | a fine-grained token without the Jobs permission | `shipwright doctor` checks the token can launch, not only list |
| vLLM dies at start: `Could not find nvcc` | vLLM 0.32's FlashInfer sampler JIT-compiles a kernel | `gate_eval.py` starts vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`; after any serving change run `scripts/vllm_smoke.py` (about $0.10) |
| A gate that never started looks like a refusal | 0 valid runs, every check "missing" | read `valid_runs` before reading a refusal as a verdict |
| `FileNotFoundError` at launch | `run_uv_job` treats every argument ending in `.py`, `.sh`, `.yaml`, `.yml`, `.toml` as a local file | the gate bundle carries configs as `.json`; a test and the doctor check it |
| Training loss starts near 12 | a missing causal shift trains the model to copy the current token | the job compares its first-step loss with the model's own `labels=` loss and refuses to train if they differ |
| `warmup_ratio` rejected | removed in transformers 5 | `warmup_steps` as a fraction; the training script pins transformers, trl and peft |
| Out of memory on a 24 GB GPU | the 262,144-token vocabulary makes the logits tensor about 4 GB per 8k row | a windowed loss over the trained tokens (`logits_to_keep`), and an L40S for training |

A check that passes on random weights proves nothing about the real model: before a paid launch, write down what a
correct result looks like on the real weights (for training: a first-step loss of a few nats, equal to the model's own,
falling) and judge the run against that.

## AgentTune and AuditKIT

| Symptom | Cause | Handling |
|---|---|---|
| `DeploymentGate` approves with no evidence | it approves by default below `min_test_samples` (default 1) | our check requires 54 paired runs |
| `FailureDetector` misses runs | it resumes from `<audit>.offset` | the bridge deletes the offset file |
| `import agenttune.agentic` fails without torch | the package imports torch | modules needed off-GPU are loaded by file path |
| `pip install agenttune` fails on macOS | torch and bitsandbytes force a source build | installed with `--no-deps` (`scripts/setup_tau2.sh`) |
| The raw model text is missing from audit records | `to_audit_records()` keeps only tool name, arguments and status | the raw output of every turn is logged separately in `traces.jsonl` |
| pass^k scored on fewer than k trials | AuditKIT `reliability()` lowers k to the decided trials | `eval_metrics.pass_metrics` scores pass^k only on tasks with all k trials decided |
| Diagnostics only for the first trial | multi-trial rows keep the first trial's diagnostics | a second one-case-per-run pass |

## Google Cloud

| Symptom | Cause | Handling |
|---|---|---|
| `PERMISSION_DENIED` or `NOT_FOUND` on a resource that exists | the wrong gcloud account config | every gcloud call names `--configuration` and `--project` (`cloud.gcloud_command`); the `lexsi` config is used when present |
| `--args` rejects a value | gcloud rejects a repeated value in `--args` | write flags as `--flag=value` (`--fetch=owner/dir`) |
| A launch runs the previous image | Cloud Build tags the image before it repoints the job | `shipwright run --launch` waits for the job to point at the image of HEAD |
| Spend runs on after a job ends | billing data lags hours | each stage's worst case is checked against its cap before it starts; GPU jobs are priced by running seconds |

## Development

| Symptom | Cause | Handling |
|---|---|---|
| The doctor's dry-run check says "rejected" | CI's dry loop and the doctor's dry run share `runs/dry-*-r10`, and a dry run resumes | move `runs/dry-*` aside before trusting that check locally |
| zsh: `script.py $ARGS` fails with a usage error | zsh does not split variables into words | write the flags out, or use a round config |
| A gate bundle misses a change | the bundle is built from the committed tree | commit before a launch; `shipwright run --launch` requires a clean, pushed HEAD |

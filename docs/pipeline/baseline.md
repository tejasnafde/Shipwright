# Baseline: Tiny Aya Global on tau2-bench retail

Runs Tiny Aya Global as the tool-calling agent in tau2-bench, with no local GPU.
Both the agent and the user simulator go through Hugging Face Inference Providers
(OpenAI-compatible router `https://router.huggingface.co/v1`).

- Agent: `CohereLabs/tiny-aya-global:cohere` (litellm id `openai/CohereLabs/tiny-aya-global:cohere`)
- User simulator: `openai/gpt-oss-120b:cerebras` (litellm id `openai/openai/gpt-oss-120b:cerebras`)
- tau2-bench: pinned commit in `scripts/setup_tau2.sh`, cloned to `third_party/` (git-ignored)

The user simulator (and the teacher, the judge and the patch proposer) can also run on
Vertex AI: see "Vertex AI models" below. Team choice (2026-11-09): user simulator and judge
`vertex/openai/gpt-oss-120b-maas`.

## Setup (once)

```sh
./scripts/setup_tau2.sh
```

## Token

The scripts read the HF token at run time from `$SWITCHBOARD_SECRETS_DIR/HF_TOKEN`,
or from `$HF_TOKEN`. The token goes to litellm through process env vars only. It is never written to disk.

## Run

```sh
# 1. Offline check of the whole pipeline (fake LLM, no network, no token)
.venv/bin/python scripts/baseline.py --dry-run --num-tasks 2 --output-dir runs/dry

# 2. Does Tiny Aya return native tool calls through the router?
.venv/bin/python scripts/check_tool_calling.py

# 3. Cost check on 5 tasks, then the 20-task baseline
.venv/bin/python scripts/baseline.py --num-tasks 5
.venv/bin/python scripts/baseline.py --num-tasks 20
```

Other flags: `--task-split train|test|base` (default `train`, so held-out tasks stay unused; a `test` run
must pass `--task-ids` and may not include hidden ids unless `--open-hidden`, see `configs/splits.yaml`),
`--task-ids 0 1 2`, `--trials`, `--agent-model`, `--user-model` (e.g. `vertex/openai/gpt-oss-120b-maas`), `--max-concurrency` (default 4),
`--output-dir` (default `runs/baseline-<timestamp>/`), `--tool-patch-ids` (with `--tool-patches`:
exactly these patches, whatever their ledger status), `--agent-api-base` (agent on an
OpenAI-compatible endpoint such as vLLM; the user simulator stays on the router). Both are used
by the gate (`docs/pipeline/gate.md`).

## Output (in the run dir, git-ignored)

- `tau2/results.json`: tau2's own results file (`tau2 view` can read it).
- `traces.jsonl`: one line per task run. Task id, reward, pass, termination reason, the full message
  list, and per agent turn: raw model output text, reasoning (if any), raw tool calls as the model
  wrote them, parsed tool calls, tool results, and error flags.
- `summary.json`: pass rate, tool calls, malformed tool calls, unknown tool names, tool errors,
  tool calls written as text, average turns, token usage, a cost estimate.
- `run_config.json`: the arguments of the run.
- `prompts.json`: the agent's un-converted system prompt and tool schemas, by `prompt_sha`
  (prompted models only). Training data is rendered from it, see `docs/pipeline/training.md`.
- `otel_spans.jsonl`: the same runs as OpenTelemetry spans (optional viewer, see `docs/ops/observability.md`).
- `log.jsonl`, `errors.json`, `repro.json`, `live_spans.jsonl`: logs, failure report, rerun record
  and live spans; `summary.json` gets `status` (ok / partial / failed). See `docs/ops/debugging.md`.

## Prompted tool calls (Tiny Aya)

Tiny Aya has no native tool calling. Its chat template knows only system, user and
assistant turns, and the HF router (cohere provider) drops the `tools` parameter
(checked 2026-11-08: HTTP 200, no `tool_calls`, 74 prompt tokens).
`src/shipwright/llm/prompted_tools.py` handles this at the litellm layer, only for models
with `tiny-aya` in the name. tau2 stays unchanged.

- Request: `tools` and `tool_choice` are removed. The tool schemas, a short instruction
  and one example go into the system message. The example uses `<...>` placeholders, never a real-looking id: until 2026-11-11 it used `#W1234567`, and the student copied it into 28 of 74 baseline traces (`invented_id`, 34 of 71 failures).
- Reply format the model must use:

  ````
  ```json
  {"name": "get_order_details", "arguments": {"order_id": "<that order id>"}}
  ```
  ````

  Any other reply is plain text to the user.
- History: assistant tool calls become that JSON block; tool results become user turns
  `Tool result for <name>: <content>`.
- Response: the first JSON object (fenced or not, trailing text allowed, trivial single
  quotes allowed) becomes a native `tool_calls` entry. If the reply looks like a tool
  call but does not parse, it goes to the user as text and the trace marks
  `tool_call_parse: parse_failed`.
- The raw reply text is kept per turn (`raw_output`). `summary.json` counts
  `tool_call_parse_failures`.
- Training data for the student model must use this same format.

## Context budget (Tiny Aya)

Tiny Aya has an 8192-token context (prompt plus reply), and the cohere provider
rejects longer requests. `src/shipwright/llm/context.py` keeps each prompted request inside it:

- Tools are listed one per line, for example
  `get_order_details(order_id: str e.g. '#W0000000') - Get the status and details of an order.`
  (`?` marks an optional argument). About 2.9k chars instead of 12k of JSON schemas.
- Tool results in history are minified JSON.
- `max_tokens` is 700. The prompt budget is 8192 - 700 - a margin (100 with the real
  tokenizer, 400 with the 3.5 chars per token estimate). The baseline downloads
  `tokenizer.json` from the Tiny Aya repo with the HF token and falls back to the estimate.
- Over budget: the oldest middle turns are dropped (the system message, the first user
  message and the latest turns stay), with an `[earlier turns omitted]` note. If the latest
  message alone is too big, it is cut with a `[truncated N chars]` marker.
- Per turn the trace has `prompt_tokens_est`, `context_dropped_messages`,
  `context_truncated_chars` and `context_trimmed`. `summary.json` has `context_trimmed_turns`.

## tau2 default models

Some tau2 tasks call a built-in judge (`DEFAULT_LLM_NL_ASSERTIONS`, an OpenAI model).
`scripts/baseline.py` sets it to the user model, and reroutes any other model that is not
the agent or the user model to the user model (`rerouted_models` in `summary.json` counts them).
A ```json fence around the judge reply is removed, because tau2 parses it with `json.loads`.

## Infra errors

`summary.json` has `pass_rate` (all runs), `pass_rate_excluding_infra_errors`,
`num_infra_errors` and `infra_error_messages` (first 150 chars, counted).
Calls that do not go to Vertex retry 429s, timeouts and 5xx errors (`patches.call_with_transient_retries`);
Vertex calls have their own retries (`vertex.call_with_refresh`).

## Malformed tool calls

tau2 runs `json.loads` on every tool-call argument string, so one bad call would crash the
simulation. `src/shipwright/llm/patches.py` wraps the litellm call: invalid arguments become
`{"__malformed_arguments__": "<original text>"}`. The environment then returns a tool error
to the agent and the run goes on. The trace shows the original text.

## Vertex AI models

A model id `vertex/<publisher>/<model>` goes to the Vertex AI OpenAI-compatible endpoint
(`src/shipwright/llm/vertex.py`); every other id goes to the HF router as before.

| Role | Id | USD per 1M tokens (in / cached in / out) |
|---|---|---|
| teacher | `vertex/deepseek-ai/deepseek-v4.1-flash-maas` | 0.30 / 0.03 / 1.20 |
| user simulator, judge, proposer | `vertex/openai/gpt-oss-120b-maas` | 0.09 / 0.09 / 0.36 |

- Endpoint: `https://aiplatform.googleapis.com/v1/projects/$SHIPWRIGHT_GCP_PROJECT/locations/global/endpoints/openapi`
  (DeepSeek V4.1 Flash is only on `global`). Project default `lexsi-hackathon`.
- Inside tau2: `llm_patch` rewrites each `vertex/` call to litellm `openai/<publisher>/<model>`
  with that `api_base` and a fresh bearer token as `api_key`, per call. On a 401 it gets a new
  token and retries once. `classify.py` and `tool_fix.py` post to the same endpoint with httpx.
- Token: `gcloud --configuration=$SHIPWRIGHT_GCLOUD_CONFIG auth print-access-token` (default
  `personal`; Ishank and Mitushi set `SHIPWRIGHT_GCLOUD_CONFIG=lexsi`). It is cached and refreshed
  after 45 minutes (tokens last about 1 hour). Never Application Default Credentials. The binary is
  `$SHIPWRIGHT_GCLOUD`, else the Homebrew path, else `gcloud` from PATH. `$SHIPWRIGHT_VERTEX_TOKEN`
  is used as is, with no refresh (HF Jobs). The token is never logged or written to disk.
- A run whose agent and user models are both `vertex/` (or agent on `--agent-api-base`) needs no HF
  token; token counting then falls back to the estimate unless the HF token is set.
- gpt-oss-120b reasons first: a `max_tokens` below 1024 is raised to 1024.
- Check access and print the usage block (2 tiny calls, GCP billing):
  `.venv/bin/python scripts/check_vertex.py` (`--cache-probe` adds 2 calls of about 5k tokens).

## Prompt caching (teacher)

Vertex has no explicit context cache (`cachedContents`) for MaaS open models; that API is for
Gemini. DeepSeek V4.1 Flash has implicit prefix caching (Preview, 90% off cached tokens, from
4096 prompt tokens, hits not guaranteed); gpt-oss-120b is not listed for it (Google docs
"Use open models", 2026-11-09).

- Every teacher request starts with the same bytes: the system message (tau2 policy plus the
  compact tool list), then the conversation. No timestamps or run ids come before the conversation
  (`tests/llm/test_vertex.py` checks this). The fixed prefix is about 2.6k tokens, below the 4096
  minimum, so hits come from the earlier turns of the same run.
- Per call, cached tokens are read from the usage block (`prompt_tokens_details.cached_tokens`,
  or a Vertex field) and stored as `usage.cached_prompt_tokens` in the trace turn (only when the
  provider reports it) and as `gen_ai.usage.cache_read.input_tokens` in the span.
- On a call with no hit, Vertex returns `prompt_tokens_details: null` (checked 2026-11-09). The
  field on a hit is not verified yet: run `check_vertex.py --cache-probe` once.

## Cost

`summary.json`:

- `usage_by_role`: calls, prompt, completion and cached prompt tokens for `agent`, `user` and
  `judge` (tau2's NL judge and any rerouted default model), counted for every LLM call.
- `cost_estimate_usd`: USD per role (`teacher` instead of `agent` in teacher runs) and `total`,
  plus `by_role` with model, provider (`vertex`, `hf_router`, `local`, `dry_run`), tokens, cache
  hit rate and the prices used. Cached tokens are priced at the cached price.
  Prices: `src/shipwright/llm/costs.py`. Unknown models get an assumed price (`price_assumed: true`).
- `cached_prompt_tokens` and `cache_hit_rate` for agent and user, from the traces.

These are estimates. Check HF billing and GCP billing for the real number.

## Classify and route failures

```sh
.venv/bin/python scripts/classify.py runs/baseline-<timestamp> --no-llm   # AgentTune detector + rules, no network
.venv/bin/python scripts/classify.py runs/baseline-<timestamp>            # + AgentTune classifier (Vertex gpt-oss-120b)
.venv/bin/python scripts/round_status.py runs/baseline-<timestamp> --sft runs/sft-<ts>/sft.jsonl
```

- DETECT: a run is a failure when reward < 1 (outcome) OR AgentTune's `FailureDetector` flags
  it (loop_collapse: the same call with the same arguments 3 times; tool_crash: a tool call
  that errored). Each row records `detected_by`. A passed run that only the detector flags
  gets route `no_fix` and `outcome_failed: false`; the gate's label rates skip it.
  Infra-error runs get the label `environment`, route `no_fix`.
- Rule labels (`src/shipwright/heal/classify.py`): `loop`, `unknown_tool`, `malformed_args`,
  `tool_error_ignored`, `context_overflow`, `invented_id`, `tool_example_copied`
  (an invented id that equals an example value from a tool description, such as `#W0000000`).
- CLASSIFY: AgentTune's `FailureClassifier` on every failed run. `--classifier-model`, default
  `vertex/openai/gpt-oss-120b-maas` (Vertex AI, `src/shipwright/llm/vertex.py`: api_base and a fresh
  gcloud token per call, never logged); any other litellm id also works. Its root cause maps to
  our labels: `wrong_tool` -> `wrong_tool`, `wrong_routing` -> `wrong_order`,
  `incomplete_reasoning` -> `skipped_policy_step`, `hallucinated_output` -> `invented_answer`,
  `loop_collapse` -> `loop` (`AGENTTUNE_ROOT_CAUSE_MAP`). All are fault layer `model`.
- Our judge (`--judge`) is optional, only for failed runs that no rule explains (`--judge-all`:
  every failed run). `--judge-model`, default `vertex/openai/gpt-oss-120b-maas` (a raw HF router
  id such as `openai/gpt-oss-120b:cerebras` also works). Strict JSON, validated, one retry.
  Labels: `wrong_tool`, `bad_arguments`, `invented_id`, `wrong_order`, `skipped_policy_step`,
  `loop`, `invented_answer`, `premature_transfer`, `tool_description_unclear`, `environment`.
- Fault layer, first match wins: a tool-layer rule (`tool_example_copied`, `context_overflow`),
  then our judge, then AgentTune's mapped root cause, then the rule layer.
- ROUTE: fault layer `tool` goes to `tool_fix`, `model` to `weight_fix`, `environment` to `no_fix`.
- Output in the run dir: `audit.jsonl` (AgentTune audit records), `failures.jsonl` (one row per
  failure, with `agenttune_root_cause`, `agenttune_failures`, `detected_by`, `source`) and
  `failure_summary.json`.
- RETRAIN TRIGGER: `scripts/round_status.py` writes `round_status.json`: `should_train` and
  AgentTune's reason (config `configs/heal.yaml`).

### AgentTune and AuditKIT classes each step calls

| Step | Class or function | Our addition |
|---|---|---|
| Log | `EventLog.to_audit_records()` schema, written by `agenttune_bridge.trace_audit_records` | raw model output per turn in `state_snapshot` |
| DETECT | `agenttune.decide.closed_loop.failure_detector.FailureDetector.scan_audit_log` | union with reward < 1; rule labels as evidence |
| CLASSIFY | `failure_classifier.FailureClassifier` (subclass: fuller prompt, rule hits as hints), run by `agentic.heal_loop.as_sync_classifier` | label map, tool-layer rules win, 3-way router |
| RETRAIN TRIGGER | `agentic.heal_loop.SelfHealLoop` with `retraining_trigger.RetrainingTrigger` (`TrainingBuffer`, `RewardDriftTracker`, `should_trigger()`, `health()`) | teacher SFT examples as corrections |
| GATE | `deployment_gate.DeploymentGate.evaluate_decision` | min_test_samples floor; AND with AuditKIT pass^1, pass^k, ToolCallF1 and BFCL checks (`docs/pipeline/gate.md`) |

Not used as is: `Project.heal()` needs an AgentTune `Project` that collected the runs itself
(ours come from tau2), so we call the same `FailureDetector` on our own audit records, as
`heal()` does. `SelfHealLoop` runs without a `trainer_factory`: training is a separate HF Job
(`docs/pipeline/training.md`). `TrainingExampleGenerator` makes corrections only for `wrong_tool` and
`loop_collapse` and needs a live replay; our corrections are teacher runs. The
`agenttune.agentic` package imports torch, so without torch `agenttune_bridge.load_heal_loop`
loads `heal_loop.py` (which needs only AgentTune's contracts) directly.

## Tool fixes (left path: patch tool descriptions, no GPU)

```sh
.venv/bin/python scripts/tool_fix.py runs/baseline-<timestamp>        # propose (rules + LLM)
.venv/bin/python scripts/tool_fix.py --list
.venv/bin/python scripts/tool_fix.py --apply-proposed                 # or --apply <id> ...
.venv/bin/python scripts/baseline.py --num-tasks 20 --tool-patches patches/ledger.jsonl
.venv/bin/python scripts/tool_fix.py --rollback <id>
```

- Input: `failures.jsonl` rows with route `tool_fix`, and the tau2 tool specs.
- Rules: for `tool_example_copied`, concrete example ids in descriptions become placeholders
  (`#W<7 digits>`, `<10 digits>`, `<first>_<last>_<digits>`, `gift_card_<digits>`).
- LLM (`--model`, default `vertex/openai/gpt-oss-120b-maas`): gets one tool spec and up to 3 failure excerpts, and returns
  `{"patches": [{tool, field, new_text, reason}]}`. Only `description` and
  `parameters.<name>.description` can change. A patch with a concrete id is rejected (one retry).
- Ledger: `patches/ledger.jsonl` (committed). One row per patch: id, round, tool, field, before,
  after, reason, source, source failures, sha256 of before and after, status
  (`proposed`, `applied`, `rolled_back`) and a status history.
- Apply: only `applied` rows take effect, in ledger order, in the Tiny Aya prompted-tools layer
  (tau2 is not edited). Each must match its `before` sha256, so a conflict fails at start.
  Fields changed by LLM patches are shown in full in the tool list. Active patch ids go to
  `run_config.json` and `summary.json`.

## Tool guards (the "add validation" half of the tool fix, no GPU)

```sh
.venv/bin/python scripts/baseline.py --num-tasks 20 --tool-patches patches/ledger.jsonl --tool-guards
.venv/bin/python scripts/check_guards.py runs/<teacher run> runs/<student run>   # offline, no credits
```

Before a Tiny Aya tool call reaches tau2, `src/shipwright/llm/guards.py` checks it against the
retail policy. In order:

| Rule | Rejects | Policy |
|---|---|---|
| `early_transfer` | a transfer before any authentication attempt | transfer "if and only if the request cannot be handled" |
| `not_authenticated` | any lookup or change before a user id was found (`find_user_id_*`) | "authenticate ... even when the user already provides the user id" |
| `repeated_failed_call` | the same call that already returned an error | |
| `made_up_id` | an `*_id` / `*_ids` value that no customer message or tool result contains (`#` ignored) | "should not make up any information" |
| `wrong_order_status` | a `*_pending_*` tool on a non-pending order, a `*_delivered_*` tool on a non-delivered one | |
| `no_confirmation` | a change unless the customer's latest message says yes to an agent question | "obtain explicit user confirmation (yes)" |

On a violation the agent gets its rejected reply and a user turn with the reason, and is asked
again (`MAX_GUARD_RETRIES` = 2); after that the call runs as written, so a wrong guard can cost
turns but never blocks a task. tau2 only sees the final call. Each agent turn in `traces.jsonl`
has `guard_rejections` (rule and rejected text) and `guard_allowed_after_retries`;
`summary.json` has `tool_guards`, `guard_rejections_by_rule` and
`guard_allowed_after_retries_by_rule`.

Checked offline on round 1 (2026-11-12, `scripts/check_guards.py`):

- Teacher (DeepSeek V4.1 Flash), 82 passing runs, 512 tool calls: 3 runs get a rejection, all
  task 22, a second address change without a new "yes" (the guard adds one confirmation turn;
  the result is the same). One task 19 rejection is an exchange that tau2 then refused anyway.
- Student (Tiny Aya, `r1-baseline-train`), 56 failed valid runs: 53 have at least one call a
  guard rejects. By rule: not_authenticated 100, early_transfer 36, repeated_failed_call 33,
  made_up_id 12, no_confirmation 2, wrong_order_status 1.

Not measured yet: how many of those runs then pass. That needs a live run (credits).

## Weight fix (teacher data and LoRA training)

See `docs/pipeline/training.md`.

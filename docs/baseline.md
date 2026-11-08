# Baseline: Tiny Aya Global on tau2-bench retail

Runs Tiny Aya Global as the tool-calling agent in tau2-bench, with no local GPU.
Both the agent and the user simulator go through Hugging Face Inference Providers
(OpenAI-compatible router `https://router.huggingface.co/v1`).

- Agent: `CohereLabs/tiny-aya-global:cohere` (litellm id `openai/CohereLabs/tiny-aya-global:cohere`)
- User simulator: `openai/gpt-oss-120b:cerebras` (litellm id `openai/openai/gpt-oss-120b:cerebras`)
- tau2-bench: pinned commit in `scripts/setup_tau2.sh`, cloned to `third_party/` (git-ignored)

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

Other flags: `--task-split train|test|base` (default `train`, so held-out tasks stay unused),
`--task-ids 0 1 2`, `--trials`, `--agent-model`, `--user-model`, `--max-concurrency` (default 4),
`--output-dir` (default `runs/baseline-<timestamp>/`).

## Output (in the run dir, git-ignored)

- `tau2/results.json`: tau2's own results file (`tau2 view` can read it).
- `traces.jsonl`: one line per task run. Task id, reward, pass, termination reason, the full message
  list, and per agent turn: raw model output text, reasoning (if any), raw tool calls as the model
  wrote them, parsed tool calls, tool results, and error flags.
- `summary.json`: pass rate, tool calls, malformed tool calls, unknown tool names, tool errors,
  tool calls written as text, average turns, token usage, a cost estimate.
- `run_config.json`: the arguments of the run.
- `prompts.json`: the agent's un-converted system prompt and tool schemas, by `prompt_sha`
  (prompted models only). Training data is rendered from it, see `docs/training.md`.

## Prompted tool calls (Tiny Aya)

Tiny Aya has no native tool calling. Its chat template knows only system, user and
assistant turns, and the HF router (cohere provider) drops the `tools` parameter
(checked 2026-11-08: HTTP 200, no `tool_calls`, 74 prompt tokens).
`src/shipwright/prompted_tools.py` handles this at the litellm layer, only for models
with `tiny-aya` in the name. tau2 stays unchanged.

- Request: `tools` and `tool_choice` are removed. The tool schemas, a short instruction
  and one example go into the system message.
- Reply format the model must use:

  ````
  ```json
  {"name": "get_order_details", "arguments": {"order_id": "#W1234567"}}
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
rejects longer requests. `src/shipwright/context.py` keeps each prompted request inside it:

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

## Malformed tool calls

tau2 runs `json.loads` on every tool-call argument string, so one bad call would crash the
simulation. `src/shipwright/llm_patch.py` wraps the litellm call: invalid arguments become
`{"__malformed_arguments__": "<original text>"}`. The environment then returns a tool error
to the agent and the run goes on. The trace shows the original text.

## Cost

The cost in `summary.json` is an estimate from token counts and assumed prices
(`PRICES` in `scripts/baseline.py`). Check HF billing for the real number.

## Classify and route failures

```sh
.venv/bin/python scripts/classify.py runs/baseline-<timestamp> --no-llm   # rules only, no network
.venv/bin/python scripts/classify.py runs/baseline-<timestamp>            # rules, then the judge
```

- DETECT: a run fails when reward < 1. Infra-error runs get the label `environment`, route `no_fix`.
- Rule labels (`src/shipwright/classify.py`): `loop`, `unknown_tool`, `malformed_args`,
  `tool_error_ignored`, `context_overflow`, `invented_id`, `tool_example_copied`
  (an invented id that equals an example value from a tool description, such as `#W0000000`).
- Judge: `openai/gpt-oss-120b:cerebras` via the HF router, only for failed runs that no rule
  explains (`--judge-all` judges every failed run). Strict JSON, validated, one retry.
  Labels: `wrong_tool`, `bad_arguments`, `invented_id`, `wrong_order`, `skipped_policy_step`,
  `loop`, `invented_answer`, `premature_transfer`, `tool_description_unclear`, `environment`.
- ROUTE: fault layer `tool` goes to `tool_fix`, `model` to `weight_fix`, `environment` to `no_fix`.
- Output in the run dir: `failures.jsonl` (one row per failed run) and `failure_summary.json`.

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
- LLM (`openai/gpt-oss-120b:cerebras`): gets one tool spec and up to 3 failure excerpts, and returns
  `{"patches": [{tool, field, new_text, reason}]}`. Only `description` and
  `parameters.<name>.description` can change. A patch with a concrete id is rejected (one retry).
- Ledger: `patches/ledger.jsonl` (committed). One row per patch: id, round, tool, field, before,
  after, reason, source, source failures, sha256 of before and after, status
  (`proposed`, `applied`, `rolled_back`) and a status history.
- Apply: only `applied` rows take effect, in ledger order, in the Tiny Aya prompted-tools layer
  (tau2 is not edited). Each must match its `before` sha256, so a conflict fails at start.
  Fields changed by LLM patches are shown in full in the tool list. Active patch ids go to
  `run_config.json` and `summary.json`.

## Weight fix (teacher data and LoRA training)

See `docs/training.md`.

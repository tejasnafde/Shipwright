# The agent harness

`scripts/baseline.py` runs an agent on tau2-bench retail and records everything the loop needs. Every stage that plays
conversations calls it: the student baseline, the candidate on the train tasks, the teacher (`teacher_runs.py`), the
ceiling and the gate.

```sh
.venv/bin/python scripts/baseline.py --dry-run --num-tasks 2 --output-dir runs/dry      # offline, fake models
.venv/bin/python scripts/baseline.py --num-tasks 5 --user-model openai/openai/gpt-oss-120b:deepinfra   # live
```

Live runs execute in Cloud Run (`docs/setup/cloud-runs.md`); the loop builds every command itself.

## Models and providers

| Role | Model | Provider |
|---|---|---|
| Agent (student) | Tiny Aya Global, `openai/CohereLabs/tiny-aya-global:cohere` | HF router; vLLM with the adapter in a gate (`--agent-api-base`) |
| User simulator and tau2 judge | gpt-oss-120b, `openai/openai/gpt-oss-120b:deepinfra` | HF router |
| Teacher | DeepSeek V4.1 Flash, `vertex/deepseek-ai/deepseek-v4.1-flash-maas` | Vertex AI |
| Classifier, judge, tool-fix proposer | gpt-oss-120b, `vertex/openai/gpt-oss-120b-maas` | Vertex AI |

An id `vertex/<publisher>/<model>` goes to Vertex AI's OpenAI-compatible endpoint (`src/shipwright/llm/vertex.py`);
every other id goes to the HF router unchanged. tau2's own default models (its NL-assertion judge and others) are
rerouted to the user model and counted in `summary.json` (`rerouted_models`).

## Prompted tool calls

Tiny Aya has no native tool calling: its chat template knows system, user and assistant turns only, and the router
drops `tools`. `src/shipwright/llm/prompted_tools.py` translates at the litellm layer for any model named `tiny-aya`;
tau2 is not modified.

- **Request:** `tools` is removed. A compact tool list (one line per tool, about 2.9k characters instead of 12k of
  JSON schema), a five-step "how to handle a request" checklist and one example with `<...>` placeholders go into
  the system message. Example values in tool descriptions that look real are replaced with a hint, because the model
  copies them as ids.
- **Reply protocol:** a tool call is one fenced JSON block; anything else is text to the customer.

  ````
  ```json
  {"name": "get_order_details", "arguments": {"order_id": "<that order id>"}}
  ```
  ````

- **History:** assistant tool calls become that block; tool results become user turns
  `Tool result for <name>: <compact result>`. Order and user records arrive without empty fields
  (`OBSERVATION_VERSION` 3).
- **Response:** the first JSON object becomes a native `tool_calls` entry. A reply that looks like a call but does
  not parse goes to the customer as text and is marked `tool_call_parse: parse_failed`.

Training data is rendered with the same functions, so the student trains on exactly the requests it is served.

## Context budget

Tiny Aya's window is 8,192 tokens. `src/shipwright/llm/context.py` keeps every request inside it: a 512-token reply
reserve, a 250-token margin with the real tokenizer (400 with the character estimate), then, when over budget, old
tool results and old replies are shortened, and only then the oldest middle turns are dropped with an
`[earlier turns omitted]` note. With `identity_note` the note also restates the authenticated user and their order ids.
Each trace turn records the estimate and what was trimmed.

## Serving switches

Per-system switches (system config keys; `baseline.py` flags), all recorded in `run_config.json` and kept by a
promotion:

| Key | Flag | What it does |
|---|---|---|
| `tool_guards` | `--tool-guards` | policy checks before a tool call runs (`docs/pipeline/tool_fix.md`) |
| `state_reminders` | `--state-reminders` | a short status note at the end of each agent request (`llm/reminders.py`) |
| `condensed_policy` | `--condensed-policy` | the retail policy in fewer words (2,714 to 1,815 fixed-prompt tokens) |
| `identity_note` | `--identity-note` | a trimmed request keeps the user id and order ids |
| `agent_voice_check` | `--agent-voice-check` | re-ask the simulated customer when it speaks as the agent (set on both systems) |
| `tool_patches` | `--tool-patches`, `--tool-patch-ids` | applied tool-description patches from the ledger |

## Reliability

- A malformed tool-call argument becomes a tool error for the agent instead of a crash (`llm/patches.py`).
- Router calls retry 429, 5xx and timeouts (2, 5, 10, 20 s); Vertex calls refresh tokens on 401, wait on 429 and
  retry timeouts (`vertex.call_with_refresh`).
- The user simulator's reply is asked again when it is empty, leaks past its turn, stops without reason or speaks as
  the agent (`docs/ops/known-issues.md`).
- One task's exception is an infra error in the results, not a dead run. A rerun into the same `--output-dir`
  keeps decided runs and redoes infra errors (tau2 `auto_resume`).
- `--max-cost-usd` stops the run when its metered spend reaches the cap.

## Splits

`--task-split train` is the default. A `test` run must name `--task-ids`; hidden ids are refused unless `--open-hidden`
(`docs/evaluation.md`).

## Output

| File | Content |
|---|---|
| `traces.jsonl` | one row per conversation: task, trial, reward, termination, every message, and per agent turn the raw model output, parsed calls, tool results, guard rejections, reminders and context trimming |
| `summary.json` | pass rate, tool calls and errors, parse failures, guard counts, token usage per role, cost estimate per role and provider, status |
| `run_config.json` | the arguments, active patches, switches, tool list and observation versions |
| `prompts.json` | the agent's unconverted system prompt and tool schemas, by hash |
| `tau2/results.json` | tau2's own results |
| `otel_spans.jsonl`, `live_spans.jsonl` | OpenTelemetry spans (`docs/ops/observability.md`) |
| `log.jsonl`, `errors.json`, `repro.json` | logs, failure report, rerun record (`docs/ops/debugging.md`) |

Costs are estimates from metered tokens and the prices in `src/shipwright/llm/costs.py`; cached prompt tokens are
priced at the cached rate.

# Observability: OpenTelemetry spans

Optional. `traces.jsonl` is the source of truth. The spans are a second view of the same
data in the OTel format, so standard trace viewers can show a run.

Two span files:
- `live_spans.jsonl`: spans created while the run runs (run > task > agent turn > LLM call or
  tool call, and every other external call), with the same trace id as the log lines. On Cloud
  Run they also go to Cloud Trace. See `docs/ops/debugging.md` (logs, failure report, why.py).
- `otel_spans.jsonl`: spans rebuilt from traces.jsonl after the run, with the full message text
  (this page).

## What a run writes

`scripts/baseline.py` (and every script that calls it: teacher runs, gate evals) writes
`otel_spans.jsonl` into the run dir. It is always on: the spans are built from the trace
records after tau2 finishes, so nothing runs inside the agent loop. An export error only
prints a warning. The run and `traces.jsonl` are not affected.

One OTel trace per task run (`src/shipwright/observability/otel_export.py`):

| Span | Attributes (main ones) |
|---|---|
| `invoke_agent <domain>` | `shipwright.task_id`, `.reward`, `.passed`, `.termination_reason`, `.reward_info`, `gen_ai.request.model` |
| `user_message` | `shipwright.content` |
| `agent_turn` | `shipwright.message_index`, `.errors` |
| `chat <model>` (child of the turn) | `shipwright.raw_output`, `.parsed_tool_calls`, `.tool_call_parse`, `.parse_failed`, `.prompt_tokens_est`, `.context_trimmed`, `.context_dropped_messages`, `gen_ai.usage.*`, `gen_ai.output.messages` |
| `execute_tool <name>` (child of the turn) | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `shipwright.result_size`, `.tool_error` |

Span times come from the tau2 message timestamps. `gen_ai.*` keys follow the OTel GenAI
semantic conventions. `openinference.span.kind`, `input.value` and `output.value` are set
too, so Phoenix shows the span kinds and the text.

## View a run in Phoenix (local)

```sh
docker run --rm -p 6006:6006 arizephoenix/phoenix:latest
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006
.venv/bin/python scripts/baseline.py --dry-run --num-tasks 2 --output-dir runs/dry
```

Open http://localhost:6006. With `OTEL_EXPORTER_OTLP_ENDPOINT` set, the spans also go to
that endpoint over OTLP/HTTP. Without it, only the file is written. Other backends
(Langfuse, Jaeger) work the same way. Set their auth with `OTEL_EXPORTER_OTLP_HEADERS`.

To send an older run, re-export its traces:

```sh
.venv/bin/python -c "import json,sys; sys.path.insert(0,'src'); from shipwright.observability.otel_export import export_run; \
d='runs/<dir>'; export_run([json.loads(l) for l in open(d+'/traces.jsonl')], d+'/replay')"
```

## Standard OTel traces as input

`scripts/otel_to_traces.py spans.jsonl -o traces.jsonl` converts span files back to trace
records (`src/shipwright/observability/otel_import.py`). Spans from our harness convert without loss
(CI checks this on the dry run). Spans from another GenAI tracer convert on a best-effort
basis: `chat` spans become agent turns and `execute_tool` spans become tool calls.

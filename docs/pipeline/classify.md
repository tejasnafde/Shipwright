# Detect, classify, route

`scripts/classify.py` turns a run directory into labelled failures and decides, per failure, which remedy can fix it.
The loop's `classify` stage and the gate both call it, so both systems in a gate are labelled the same way.

```sh
.venv/bin/python scripts/classify.py runs/<run> --no-llm   # AgentTune detector + rules, offline
.venv/bin/python scripts/classify.py runs/<run>            # + AgentTune FailureClassifier (Vertex gpt-oss-120b)
```

## Detect

A run is a failure when its tau2 reward is below 1, or when AgentTune's `FailureDetector` flags it
(`loop_collapse`: the same call with the same arguments three times; `tool_crash`: a tool call that errored). The
traces are converted to AgentTune audit records first (`agenttune_bridge.trace_audit_records`, with the raw model output
of every turn added, which AgentTune's own records drop). A passed run that only the detector flags is recorded with
route `no_fix`. Infra-error runs get the label `environment` and route `no_fix`.

## Classify

Three sources, all recorded per failure (`source`, `detected_by`):

1. **Rules** (`src/shipwright/heal/classify.py`, deterministic): `loop`, `unknown_tool`, `malformed_args`,
   `tool_error_ignored`, `context_overflow`, `invented_id`, `tool_example_copied` (an invented id equal to an example
   value from a tool description the run was shown), `premature_transfer` (a transfer before any attempt to
   authenticate).
2. **AgentTune `FailureClassifier`** on every failed run, through a subclass with a fuller prompt and the rule hits as
   hints, at most 4 requests at a time. Its root causes map into our labels: `wrong_tool` to `wrong_tool`,
   `wrong_routing` to `wrong_order`, `incomplete_reasoning` to `skipped_policy_step`, `hallucinated_output` to
   `invented_answer`, `loop_collapse` to `loop`.
3. **An optional judge** (`--judge`) for failures no rule explains, with strict JSON output and one retry. Its labels
   include `bad_arguments`, `tool_description_unclear` and `environment`.

## Route

Each failure gets a fault layer, first match wins: a tool-layer rule (`tool_example_copied`, `context_overflow`), then
the judge, then AgentTune's mapped root cause, then the remaining rules.

| Fault layer | Route | Remedy |
|---|---|---|
| tool | `tool_fix` | tool-description patches and tool guards (`docs/pipeline/tool_fix.md`) |
| model | `weight_fix` | teacher data and LoRA training (`docs/pipeline/training.md`) |
| environment | `no_fix` | none: infra errors and simulator faults are not the agent's |

## Output

| File | Content |
|---|---|
| `failures.jsonl` | one row per failure: labels, fault layer, route, source, AgentTune root cause and failures, the first wrong turn |
| `failure_summary.json` | counts per label, route and source; check `per_source` covers the failures before trusting it |
| `audit.jsonl` | the AgentTune audit records the detector read |

## Retraining trigger

Whether a round trains at all is AgentTune's `RetrainingTrigger` (`SelfHealLoop`), configured in `configs/heal.yaml`:
it fires on the buffer of weight-fix failures that have a teacher correction, and blocks when too many have none. The
loop's train stage calls it, and `scripts/round_status.py runs/<run> --sft <sft.jsonl>` runs it by hand
(`round_status.json`: `should_train` and AgentTune's reason).

# Something broke: find where and why, then rerun it

Every script run writes a failure report, a repro record and structured logs. Three steps:

```sh
.venv/bin/python scripts/why.py <run>                 # 1. where and why
#                                                     # 2. open the logs and trace links it prints
.venv/bin/python scripts/rerun.py <run> --task <id>   # 3. the exact command to run it again
```

`<run>` is any of:

| Form | Example |
|---|---|
| local run dir | `runs/baseline-20261109-120000` |
| GCS run dir (Cloud Run Jobs) | `gs://lexsi-hackathon-runs/main/baseline-20261109-120000` |
| Cloud Run execution name (the run id) | `shipwright-main-x7k2p` |
| job index | `gs://lexsi-hackathon-runs/main/jobs/shipwright-main-x7k2p.json` |
| local run id | `local-20261109-120000-ab12` (looked up in `runs/*/repro.json`) |

GCS reads use the named gcloud config (`docs/setup/gcp-setup.md`) and only small files (`traces.jsonl` only with
`--task`).

## 1. why.py

Prints the run status, the first error (where: step, component, task, turn; what: the exception;
the model's last input and raw output before it; the last stack lines), the failed tasks, the
other errors, a timeline (start, end, errors), the links, and the next commands.

`--task <id>` shows one task turn by turn: every message, the agent's raw output, tool results
(errors marked), the logged error with its stack, and links filtered to that task.
`--all` lists every failed task, not only the first 10.

## 2. The links

- **Logs**: Logs Explorer, filtered by `jsonPayload.run_id="<run id>"` (and `jsonPayload.task_id`
  for a task). On Cloud Run every stdout line is a JSON log entry. From a shell:
  `gcloud --configuration=lexsi --project=lexsi-hackathon logging read 'jsonPayload.run_id="<id>"' --freshness=1d`.
- **Trace**: Cloud Trace, one trace per run: `job <script>` > `<script>` > `task <id>` >
  `agent_turn` > `chat <model>`, plus `execute_tool <name>`, `gcs_upload`, `token_fetch`, ...
  Each log line has `logging.googleapis.com/trace` and `spanId`, so Logs Explorer shows the
  span of a line, and Cloud Trace shows the logs of a span.

Local runs have the same links, but nothing is sent to GCP from a laptop: use `log.jsonl` and
`live_spans.jsonl` in the run dir instead.

## 3. rerun.py

Reads `repro.json` and prints the command. Nothing runs without `--execute`.

```sh
.venv/bin/python scripts/rerun.py <run> --task 3                       # local, offline (fake LLM)
.venv/bin/python scripts/rerun.py <run> --where cloud --offline        # dry run on your Cloud Run Job
.venv/bin/python scripts/rerun.py <run> --where cloud                  # the same run again (spends credits)
```

- `local-dry-run` (default) adds `--dry-run` (baseline, teacher runs), `--no-llm` (classify,
  tool_fix) or `--local-dry-run` (gate_eval), and drops `--launch` and `--push`. Live runs never
  run on a laptop.
- `cloud` runs `gcloud run jobs execute shipwright-<owner>` with the same flags. The job runs the
  latest image of its branch; the note shows the image and git sha of the original run.
- A command that calls model APIs needs `--execute --allow-spend`, so a paid rerun is always deliberate.
- `--output-dir` becomes `runs/rerun-<run id>-<stamp>`; `--fail-at` is kept (it reproduces the failure).

## What every run writes

In each run dir (and uploaded to GCS by `job_entry.py`):

| File | What |
|---|---|
| `log.jsonl` | every log line of the run, JSON (the same lines as stdout) |
| `errors.json` | status, counts, first error, every error (stack, model, last input and output), every failed task (first error, turn, last model input, raw output), trace id, links |
| `repro.json` | run id, script, argv, owner, git sha (and dirty flag), image tag and digest, tau2 commit, package versions, models, seed, task ids, tool patch ids |
| `summary.json` | adds `status`: `ok`, `partial` or `failed`, with `status_counts` and `status_by_step` |
| `live_spans.jsonl` | the spans created while the run ran |
| `otel_spans.jsonl` | spans rebuilt from traces.jsonl after the run, with full text (docs/ops/observability.md) |

Status: `failed` = the script crashed, exited non-zero, or every task run hit an infra error
(the script then exits 1). `partial` = it finished, but some task runs hit infra errors or errors
were logged. `ok` = no errors. A task the agent failed (reward < 1) is an outcome, not an error.
A later script on the same dir (classify after baseline) adds its own step to `status_by_step`;
`status` is the worst.

On Cloud Run, `gs://lexsi-hackathon-runs/<owner>/jobs/<run id>.json` lists the run dirs, exit
code, log and links of one execution. If the script dies without a report (killed, out of
memory), `job_entry.py` writes `errors.json` with the tail of the console log.

## Logs: the fields

One JSON object per line on stdout (human-readable on a TTY; `SHIPWRIGHT_LOG_FORMAT=json|human`):
`severity`, `message`, `time`, `logger`, `run_id`, `task_id`, `trial`, `turn`, `step`
(run, detect, classify, route, tool_fix, teacher, sft, train, gate), `component`
(`llm.agent`, `llm.user`, `llm.judge`, `tool`, `gcs_upload`, `token_fetch`, `hf_hub_upload`, ...),
`model`, `git_sha`, `image_tag`, `owner`, `trace_id`, `span_id`, and for an error `error_class`,
`error`, `exc_info` (full stack).

Every external call logs one line: target, `latency_ms`, `status`, `tokens_in`, `tokens_out`,
`tokens_cached`, `retries`, and on a failure the error and stack. That covers the agent, user
simulator, judge and teacher (the litellm wrapper in `shipwright.llm.patches`), the AgentTune
classifier, our judge, BFCL calls, Vertex token fetches, GCS uploads, HF Hub uploads and HF Job
launches. Tool calls (tau2 `Environment.get_response`) log the tool, latency and error flag.

Secrets are redacted from every field: `hf_...`, `ya29....`, `Bearer ...`, `sk-...`
(`shipwright.observability.log.redact`, tested in `tests/observability/test_observability.py`).

## In code

```python
from shipwright.observability import log
logger = log.get_logger(__name__)
logger.info("tool patches active", patches=active)    # keywords become JSON fields
with log.bind(task_id=tid):                            # context for every line in the block
    ...
log.attach_run_dir(output_directory)                        # log.jsonl, errors.json, repro.json go here
log.note_repro(seed=run_options.seed)                       # extra facts for repro.json

if __name__ == "__main__":
    sys.exit(log.run_script(main, step="run"))
```

`shipwright.observability.instrument.external(kind, target)` times and logs any other external call.

## Test the failure path

Every script run through `log.run_script` takes a hidden `--fail-at <point>`:
`agent_call` (the agent's second model call in each task raises), `tool_call`, `summary`
(baseline, after tau2 and before the summary), or a step name (raises when the script starts).

```sh
.venv/bin/python scripts/baseline.py --dry-run --num-tasks 2 --output-dir runs/fail --fail-at agent_call
.venv/bin/python scripts/why.py runs/fail
```

## Cloud Trace

On Cloud Run, spans also go to Cloud Trace (v2 REST API, metadata-server token of
`shipwright-runner@`, which has `roles/cloudtrace.agent`). An export error only warns once; the
file export goes on. `SHIPWRIGHT_CLOUD_TRACE=0` turns it off, `=1` forces it on. Read a trace from a shell:

```sh
curl -s -H "Authorization: Bearer $(gcloud --configuration=lexsi auth print-access-token)" \
  https://cloudtrace.googleapis.com/v1/projects/lexsi-hackathon/traces/<trace id> | head -c 2000
```

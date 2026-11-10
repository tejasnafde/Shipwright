"""Per-run failure report (``errors.json``), run status, and console links (docs/ops/debugging.md).

Pure functions on plain dicts, so the report can be built and tested offline.

errors.json:
  run_id, run_dir, status, counts, trace_id, links {logs, trace}
  first_error     the first logged error, else the first failed task
  errors[]        every ERROR log record of the run: time, step, component, task, trial, turn,
                  message, error class, error, stack, model, last model input and raw output,
                  trace and span id, links
  failed_tasks[]  every task run that did not pass: reward, termination, the first error and
                  its turn, the last model input (trimmed) and the raw output, links
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

CONSOLE = "https://console.cloud.google.com"
DEFAULT_PROJECT = "lexsi-hackathon"
TRIM = 1500
INFRA_ERROR = "infrastructure_error"
STATUS_ORDER = {"ok": 0, "partial": 1, "failed": 2}


def trim(value: Any, limit: int = TRIM) -> Any:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f" ... [{len(text) - limit} more chars]"


def logs_url(run_id: str, project: str = DEFAULT_PROJECT, task_id: str | None = None) -> str:
    """Logs Explorer, filtered to one run (and one task), last 30 days."""
    q = f'jsonPayload.run_id="{run_id}"'
    if task_id is not None:
        q += f'\njsonPayload.task_id="{task_id}"'
    return f"{CONSOLE}/logs/query;query={quote(q, safe='')};duration=P30D?project={project}"


def trace_url(trace_id: str, project: str = DEFAULT_PROJECT) -> str:
    return f"{CONSOLE}/traces/list?project={project}&tid={trace_id}"


def links(run_id: str | None, trace_id: str | None, project: str = DEFAULT_PROJECT,
          task_id: str | None = None) -> dict[str, str]:
    out = {}
    if run_id:
        out["logs"] = logs_url(run_id, project, task_id)
    if trace_id:
        out["trace"] = trace_url(trace_id, project)
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _turn_number(trace: dict, message_index: int | None) -> int | None:
    for n, t in enumerate(trace.get("agent_turns") or []):
        if t.get("message_index") == message_index:
            return n
    return None


def _message_text(msg: dict | None) -> str | None:
    if not msg:
        return None
    text = msg.get("content")
    if msg.get("tool_calls"):
        text = (text or "") + " " + json.dumps(msg["tool_calls"], default=str)
    return f"[{msg.get('role')}] {text}" if text is not None else f"[{msg.get('role')}]"


def failed_task(trace: dict, run_id: str | None = None, trace_id: str | None = None,
                span_id: str | None = None, project: str = DEFAULT_PROJECT) -> dict:
    """The failure entry for one trace record that did not pass."""
    messages = trace.get("messages") or []
    turns = trace.get("agent_turns") or []
    bad = next((t for t in turns if t.get("errors")), None)
    pick = bad or (turns[-1] if turns else None)
    idx = pick.get("message_index") if pick else None
    prev = messages[idx - 1] if isinstance(idx, int) and idx > 0 and idx - 1 < len(messages) else None
    if trace.get("error"):
        first = f"{trace.get('error_type') or 'error'}: {trace.get('error')}"
    elif bad:
        first = ", ".join(bad["errors"])
    else:
        first = f"reward {trace.get('reward')} ({trace.get('termination_reason')})"
    task_id = None if trace.get("task_id") is None else str(trace.get("task_id"))
    return {
        "task_id": task_id,
        "trial": trace.get("trial"),
        "reward": trace.get("reward"),
        "termination_reason": trace.get("termination_reason"),
        "infra_error": trace.get("termination_reason") == INFRA_ERROR,
        "first_error": trim(first, 500),
        "turn": _turn_number(trace, idx) if pick else None,
        "message_index": idx,
        "turn_errors": (bad or {}).get("errors") or [],
        "last_model_input": trim(_message_text(prev)),
        "last_raw_output": trim(pick.get("raw_output") if pick else None),
        "reward_info": trim(trace.get("reward_info"), 800),
        "trace_id": trace_id,
        "span_id": span_id,
        "links": links(run_id, trace_id, project, task_id),
    }


def status_of(exit_code: int | None, traces: list[dict], errors: list[dict], crashed: bool) -> tuple[str, dict]:
    """ok / partial / failed, and the counts behind it.

    failed: the script crashed or exited non-zero, or every task run is an infra error.
    partial: it finished, but some task runs hit infra errors or errors were logged.
    ok: finished with no errors. Tasks the agent failed (reward < 1) are an outcome, not an
    error: they are counted, but they do not change the status.
    """
    n = len(traces)
    infra = sum(1 for t in traces if t.get("termination_reason") == INFRA_ERROR)
    passed = sum(1 for t in traces if t.get("passed"))
    counts = {"tasks": n, "passed": passed, "failed": n - passed, "infra_errors": infra,
              "errors_logged": len(errors), "exit_code": exit_code}
    if crashed or (exit_code not in (0, None)) or (n and infra == n):
        return "failed", counts
    if infra or errors:
        return "partial", counts
    return "ok", counts


def worst(*statuses: str | None) -> str:
    known = [s for s in statuses if s in STATUS_ORDER]
    return max(known, key=STATUS_ORDER.__getitem__) if known else "ok"


def build(run_dir: Path, *, run_id: str | None, step: str | None, exit_code: int | None, crashed: bool,
          errors: Iterable[dict], trace_id: str | None, task_spans: dict | None = None,
          project: str = DEFAULT_PROJECT) -> dict:
    """The errors.json body for one run dir (reads traces.jsonl if it is there)."""
    traces = read_jsonl(Path(run_dir) / "traces.jsonl")
    errors = [dict(e) for e in errors]
    task_spans = task_spans or {}
    for e in errors:
        e.setdefault("links", links(run_id, e.get("trace_id") or trace_id, project, e.get("task_id")))
    by_task: dict[str, dict] = {}
    for e in errors:
        if e.get("task_id") is not None:
            by_task.setdefault(str(e["task_id"]), e)
    failed = []
    for t in traces:
        if t.get("passed"):
            continue
        key = (None if t.get("task_id") is None else str(t.get("task_id")), t.get("trial"))
        entry = failed_task(t, run_id, trace_id, task_spans.get(key), project)
        logged = by_task.get(entry["task_id"] or "")
        if logged:
            entry["logged_error"] = {k: logged.get(k) for k in
                                     ("message", "error_class", "error", "step", "component", "turn", "model")}
            if logged.get("turn") is not None:
                entry["turn"] = logged["turn"]
            entry["last_model_input"] = entry["last_model_input"] or logged.get("last_input")
            entry["last_raw_output"] = entry["last_raw_output"] or logged.get("last_output")
        failed.append(entry)
    status, counts = status_of(exit_code, traces, errors, crashed)
    first = errors[0] if errors else (failed[0] if failed else None)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "step": step,
        "status": status,
        "counts": counts,
        "trace_id": trace_id,
        "links": links(run_id, trace_id, project),
        "first_error": first,
        "errors": errors,
        "failed_tasks": failed,
    }


def write(run_dir: Path, report: dict) -> None:
    """Write errors.json and the status into summary.json.

    A second script on the same dir (classify after baseline) keeps the earlier errors:
    errors from other run ids stay, and the status is the worst of all steps.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "errors.json"
    old: dict = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except ValueError:
            old = {}
    steps = dict(old.get("status_by_step") or {})
    if old.get("run_id") and old.get("run_id") != report.get("run_id"):
        kept = [e for e in old.get("errors") or [] if e.get("run_id") not in (None, report.get("run_id"))]
        kept += [dict(e, run_id=e.get("run_id") or old.get("run_id")) for e in old.get("errors") or []
                 if e.get("run_id") is None]
        report = {**report, "errors": kept + report["errors"]}
        if report.get("first_error") is None and report["errors"]:
            report["first_error"] = report["errors"][0]
    steps[report.get("step") or "run"] = report["status"]
    report = {**report, "status_by_step": steps, "status": worst(*steps.values())}
    path.write_text(json.dumps(report, indent=2, default=str))

    summary_path = run_dir / "summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except ValueError:
            summary = {}
    summary["status"] = report["status"]
    summary["status_by_step"] = steps
    summary["status_counts"] = report["counts"]
    summary["run_id"] = summary.get("run_id") or report.get("run_id")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

"""Why did a run fail? Prints a short report: status, first error, failing tasks, links.

Usage:
  .venv/bin/python scripts/why.py runs/<dir>
  .venv/bin/python scripts/why.py gs://lexsi-hackathon-runs/<owner>/<dir>
  .venv/bin/python scripts/why.py <run id>              # e.g. shipwright-tejas-abc12 (Cloud Run execution)
  .venv/bin/python scripts/why.py <ref> --task 3       # one task, turn by turn

Reads errors.json, summary.json, repro.json and log.jsonl (and traces.jsonl for --task), only small
files. GCS reads use gcloud with $SHIPWRIGHT_GCLOUD_CONFIG (default personal). docs/ops/debugging.md.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.observability import report  # noqa: E402
from shipwright.observability.runinfo import Source, describe, resolve  # noqa: E402

KEY_EVENTS = ("script start", "run start", "run done", "script end", "job start", "job end")


def _short_stack(stack: str | None, line_count: int = 4) -> list[str]:
    """The last few non-empty lines of a stack trace."""
    if not stack:
        return []
    stack_lines = [line for line in stack.strip().splitlines() if line.strip()]
    return stack_lines[-line_count:]


def _where(record: dict) -> str:
    """Where a log record or failed task happened: "step x > component > task 3 trial 0 > turn 2"."""
    location_parts = []
    if record.get("step"):
        location_parts.append(f"step {record['step']}")
    if record.get("component"):
        location_parts.append(str(record["component"]))
    if record.get("task_id") is not None:
        task_text = f"task {record['task_id']}"
        if record.get("trial") is not None:
            task_text += f" trial {record['trial']}"
        location_parts.append(task_text)
    if record.get("turn") is not None:
        location_parts.append(f"turn {record['turn']}")
    return " > ".join(location_parts) or "?"


def _time(timestamp_text: str | None) -> datetime | None:
    try:
        if timestamp_text:
            return datetime.fromisoformat(timestamp_text)
        return None
    except ValueError:
        return None


def timeline(logs: list[dict], limit: int = 12) -> list[str]:
    """Key events and the first errors, with seconds since the first line."""
    if not logs:
        return []

    first_time = _time(logs[0].get("time"))
    timeline_rows = []
    errors_shown = 0
    task_runs_ended = 0
    for log_record in logs:
        message = log_record.get("message") or ""
        severity = log_record.get("severity")
        is_error = severity in ("ERROR", "CRITICAL")
        is_shown = message in KEY_EVENTS or (is_error and errors_shown < 3)
        if message.startswith("task ") and message.endswith(" end"):
            task_runs_ended += 1
        if not is_shown:
            continue
        if is_error:
            errors_shown += 1

        record_time = _time(log_record.get("time"))
        if record_time and first_time:
            seconds_text = f"+{(record_time - first_time).total_seconds():7.1f}s"
        else:
            seconds_text = "        "

        extra_text = ""
        if message == "script end":
            extra_text = f" exit {log_record.get('exit_code')}, status {log_record.get('status')}"
        elif is_error:
            extra_text = f" [{_where(log_record)}] {log_record.get('error_class') or ''}"
        timeline_rows.append(f"  {seconds_text}  {severity or '':7} {message}{extra_text}")

    timeline_rows.append(f"  ({len(logs)} log lines, {task_runs_ended} task runs ended)")
    return timeline_rows[-limit - 1:]


def render_header(source: Source, errors: dict, summary: dict, repro_record: dict, index: dict | None) -> list[str]:
    """RUN line, directory, code version and counts."""
    run_id = errors.get("run_id") or repro_record.get("run_id") or summary.get("run_id") or (index or {}).get("run_id")
    status = (errors.get("status") or summary.get("status") or "unknown").upper()
    script_command = f"{repro_record.get('script', '?')} {' '.join(repro_record.get('argv') or [])}"
    lines = [
        f"RUN {run_id}  status {status}  ({script_command})",
        f"  dir:    {source.location}",
    ]

    git_text = f"git {(repro_record.get('git_sha') or '?')[:12]}"
    if repro_record.get("git_dirty"):
        git_text += " (dirty)"
    code_parts = [git_text]
    if repro_record.get("image"):
        code_parts.append(f"image {repro_record['image'].rsplit('/', 1)[-1]}")
    if repro_record.get("tau2_commit"):
        code_parts.append(f"tau2 {repro_record['tau2_commit'][:8]}")
    lines.append(f"  code:   {', '.join(code_parts)}")

    counts = errors.get("counts") or summary.get("status_counts") or {}
    if counts:
        lines.append(
            f"  counts: {counts.get('tasks', 0)} task runs, {counts.get('passed', 0)} passed,"
            f" {counts.get('failed', 0)} failed, {counts.get('infra_errors', 0)} infra errors,"
            f" {counts.get('errors_logged', 0)} errors logged, exit {counts.get('exit_code')}"
        )
    if not errors:
        missing_note = "  no errors.json: the run did not finish, or it predates the failure report."
        if index and index.get("log"):
            missing_note += f" Job log: {index['log']}"
        lines.append(missing_note)
    return lines


def render_first_error(first_error: dict) -> list[str]:
    lines = ["", "FIRST ERROR"]
    if "message" in first_error:  # a logged error
        where_line = f"  where:  {_where(first_error)}"
        if first_error.get("model"):
            where_line += f"   (model {first_error['model']})"
        lines.append(where_line)
        what = f"{first_error.get('error_class') or ''}: {first_error.get('error') or ''}".strip(": ")
        lines.append(f"  what:   {describe(what or first_error.get('message'), 300)}")
        lines.append(f"  log:    {describe(first_error.get('message'), 200)}")
        if first_error.get("last_input"):
            lines.append(f"  input:  {describe(first_error['last_input'], 240)}")
        if first_error.get("last_output"):
            lines.append(f"  output: {describe(first_error['last_output'], 240)}   (model's raw output before it)")
        for stack_line in _short_stack(first_error.get("stack")):
            lines.append(f"  stack:  {stack_line}")
    else:  # a failed task, no logged error
        lines.append(
            f"  where:  task {first_error.get('task_id')} trial {first_error.get('trial')}"
            f" > turn {first_error.get('turn')}"
        )
        lines.append(f"  what:   {describe(first_error.get('first_error'), 300)}")
        if first_error.get("last_model_input"):
            lines.append(f"  input:  {describe(first_error['last_model_input'], 240)}")
        if first_error.get("last_raw_output"):
            lines.append(f"  output: {describe(first_error['last_raw_output'], 240)}")
    return lines


def render_failed_tasks(failed_tasks: list[dict], show_all: bool) -> list[str]:
    lines = ["", f"FAILED TASKS ({len(failed_tasks)})"]
    shown_tasks = failed_tasks if show_all else failed_tasks[:10]
    for failed_task in shown_tasks:
        lines.append(
            f"  task {failed_task.get('task_id')} trial {failed_task.get('trial')}"
            f"  reward {failed_task.get('reward')}  {failed_task.get('termination_reason')}"
            f"  turn {failed_task.get('turn')}: {describe(failed_task.get('first_error'), 110)}"
        )
    if len(failed_tasks) > 10 and not show_all:
        lines.append(f"  ... {len(failed_tasks) - 10} more (--all)")
    return lines


def render_other_errors(all_errors: list[dict], first_error: dict | None) -> list[str]:
    first_is_listed = bool(all_errors) and all_errors[0] == first_error
    if first_is_listed:
        other_errors = all_errors[1:6]
    else:
        other_errors = all_errors[:5]
    if not other_errors:
        return []

    other_count = len(all_errors) - (1 if first_is_listed else 0)
    lines = ["", f"OTHER ERRORS ({other_count})"]
    for error in other_errors:
        error_class = describe(error.get("error_class") or "", 40)
        error_text = describe(error.get("error") or error.get("message"), 120)
        lines.append(f"  [{_where(error)}] {error_class} {error_text}")
    return lines


def render_links(links: dict, index: dict | None) -> list[str]:
    lines = ["", "LINKS"]
    for link_name, link_url in links.items():
        lines.append(f"  {link_name + ':':7} {link_url}")
    if index and index.get("log"):
        lines.append(f"  joblog: {index['log']}")
    return lines


def render(source: Source, index: dict | None = None, show_all: bool = False, ref: str | None = None) -> str:
    """The failure report of one run: header, first error, failed tasks, timeline, links, next steps."""
    errors = source.json("errors.json") or {}
    summary = source.json("summary.json") or {}
    repro_record = source.json("repro.json") or {}
    logs = source.jsonl("log.jsonl", max_bytes=20_000_000) or []
    ref = ref or source.location

    lines = render_header(source, errors, summary, repro_record, index)

    first_error = errors.get("first_error")
    if first_error:
        lines += render_first_error(first_error)

    failed_tasks = sorted(
        errors.get("failed_tasks") or [],
        key=lambda failed_task: (not failed_task.get("infra_error"), str(failed_task.get("task_id"))),
    )
    if failed_tasks:
        lines += render_failed_tasks(failed_tasks, show_all)

    lines += render_other_errors(errors.get("errors") or [], first_error)

    timeline_rows = timeline(logs)
    if timeline_rows:
        lines += ["", "TIMELINE", *timeline_rows]

    run_id = errors.get("run_id") or repro_record.get("run_id") or summary.get("run_id") or (index or {}).get("run_id")
    links = errors.get("links") or report.links(run_id, repro_record.get("trace_id"))
    if links:
        lines += render_links(links, index)

    lines += ["", "NEXT"]
    first_failed_task_id = failed_tasks[0].get("task_id") if failed_tasks else None
    if first_failed_task_id is not None:
        lines.append(f"  zoom:   .venv/bin/python scripts/why.py {ref} --task {first_failed_task_id}")
        lines.append(
            f"  rerun:  .venv/bin/python scripts/rerun.py {ref} --task {first_failed_task_id} --where local-dry-run"
        )
    else:
        lines.append(f"  rerun:  .venv/bin/python scripts/rerun.py {ref} --where local-dry-run")
    return "\n".join(lines)


def render_trial(trace: dict, max_chars: int) -> list[str]:
    """One trial of a task: header line, then every message."""
    header = (
        f"trial {trace.get('trial')}: reward {trace.get('reward')}, {trace.get('termination_reason')},"
        f" {trace.get('num_messages')} messages"
    )
    if trace.get("error"):
        header += f", error: {describe(trace.get('error'), 200)}"
    lines = ["", header]

    agent_turn_by_message_index = {
        turn.get("message_index"): (turn_number, turn)
        for turn_number, turn in enumerate(trace.get("agent_turns") or [])
    }
    for message_index, message in enumerate(trace.get("messages") or []):
        role = message.get("role")
        if role == "assistant" and message_index in agent_turn_by_message_index:
            turn_number, turn = agent_turn_by_message_index[message_index]
            if turn.get("raw_output") is not None:
                text = turn.get("raw_output")
            else:
                text = message.get("content")
            flag = ""
            if turn.get("errors"):
                flag = f"  <-- {', '.join(turn['errors'])}"
            lines.append(f"  [{message_index:3}] agent turn {turn_number}: {describe(text, max_chars)}{flag}")
        elif role == "tool":
            error_mark = " (ERROR)" if message.get("error") else ""
            lines.append(f"  [{message_index:3}] tool{error_mark}: {describe(message.get('content'), max_chars)}")
        else:
            lines.append(f"  [{message_index:3}] {role}: {describe(message.get('content'), max_chars)}")
    return lines


def render_task(source: Source, task_id: str, max_chars: int = 300) -> str:
    """Every trial of one task, message by message, with the logged error and links."""
    errors = source.json("errors.json") or {}
    traces = source.jsonl("traces.jsonl", max_bytes=50_000_000)
    lines = [f"TASK {task_id}  in {source.location}"]
    if traces is None:
        lines.append("  traces.jsonl is missing or over 50 MB.")
        traces = []

    task_traces = [trace for trace in traces if str(trace.get("task_id")) == str(task_id)]
    if not task_traces and traces:
        known_task_ids = ", ".join(sorted({str(trace.get("task_id")) for trace in traces}))
        lines.append(f"  no task {task_id} here (tasks: {known_task_ids})")
    for trace in task_traces:
        lines += render_trial(trace, max_chars)

    for logged_error in errors.get("errors") or []:
        if str(logged_error.get("task_id")) == str(task_id):
            lines.append("")
            lines.append(
                f"LOGGED ERROR [{_where(logged_error)}] {logged_error.get('error_class')}:"
                f" {describe(logged_error.get('error'), 300)}"
            )
            for stack_line in _short_stack(logged_error.get("stack"), 8):
                lines.append(f"  {stack_line}")
            break

    failed_task_entry = next(
        (failed for failed in errors.get("failed_tasks") or [] if str(failed.get("task_id")) == str(task_id)),
        None,
    )
    links = (failed_task_entry or {}).get("links") or report.links(
        errors.get("run_id"), errors.get("trace_id"), task_id=str(task_id)
    )
    if links:
        lines += ["", "LINKS"]
        for link_name, link_url in links.items():
            lines.append(f"  {link_name + ':':7} {link_url}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ref", help="run dir, gs:// path, job index (gs://.../jobs/<id>.json) or run id")
    parser.add_argument("--task", default=None, help="zoom into one task id")
    parser.add_argument("--all", action="store_true", help="list every failed task")
    options = parser.parse_args(argv)

    try:
        sources, index = resolve(options.ref)
    except LookupError as lookup_error:
        print(str(lookup_error), file=sys.stderr)
        return 2

    if not sources:
        no_run_dir_message = f"{options.ref}: the job wrote no run dir."
        if index:
            no_run_dir_message += f" Job log: {index.get('log')}"
        print(no_run_dir_message)
        if index:
            print(f"exit {index.get('exit_code')}; links: {index.get('links')}")
        return 1

    for source_number, source in enumerate(sources):
        if source_number:
            print("\n" + "-" * 72 + "\n")
        if options.task:
            print(render_task(source, options.task))
        else:
            if len(sources) == 1:
                ref = options.ref
            else:
                ref = source.location
            print(render(source, index, options.all, ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())

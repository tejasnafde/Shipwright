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


def _short_stack(stack: str | None, lines: int = 4) -> list[str]:
    if not stack:
        return []
    rows = [r for r in stack.strip().splitlines() if r.strip()]
    return rows[-lines:]


def _where(e: dict) -> str:
    parts = []
    if e.get("step"):
        parts.append(f"step {e['step']}")
    if e.get("component"):
        parts.append(str(e["component"]))
    if e.get("task_id") is not None:
        parts.append(f"task {e['task_id']}" + (f" trial {e['trial']}" if e.get("trial") is not None else ""))
    if e.get("turn") is not None:
        parts.append(f"turn {e['turn']}")
    return " > ".join(parts) or "?"


def _time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def timeline(logs: list[dict], limit: int = 12) -> list[str]:
    """Key events and the first errors, with seconds since the first line."""
    if not logs:
        return []
    t0 = _time(logs[0].get("time"))
    rows, errors, task_end = [], 0, 0
    for d in logs:
        msg = d.get("message") or ""
        sev = d.get("severity")
        keep = msg in KEY_EVENTS or (sev in ("ERROR", "CRITICAL") and errors < 3)
        if msg.startswith("task ") and msg.endswith(" end"):
            task_end += 1
        if not keep:
            continue
        if sev in ("ERROR", "CRITICAL"):
            errors += 1
        t = _time(d.get("time"))
        dt = f"+{(t - t0).total_seconds():7.1f}s" if t and t0 else "        "
        extra = ""
        if msg == "script end":
            extra = f" exit {d.get('exit_code')}, status {d.get('status')}"
        elif sev in ("ERROR", "CRITICAL"):
            extra = f" [{_where(d)}] {d.get('error_class') or ''}"
        rows.append(f"  {dt}  {sev or '':7} {msg}{extra}")
    rows.append(f"  ({len(logs)} log lines, {task_end} task runs ended)")
    return rows[-limit - 1:]


def render(src: Source, index: dict | None = None, show_all: bool = False, ref: str | None = None) -> str:
    errors = src.json("errors.json") or {}
    summary = src.json("summary.json") or {}
    rep = src.json("repro.json") or {}
    logs = src.jsonl("log.jsonl", max_bytes=20_000_000) or []
    ref = ref or src.location
    out: list[str] = []
    run_id = errors.get("run_id") or rep.get("run_id") or summary.get("run_id") or (index or {}).get("run_id")
    status = (errors.get("status") or summary.get("status") or "unknown").upper()
    out.append(f"RUN {run_id}  status {status}  ({rep.get('script', '?')} {' '.join(rep.get('argv') or [])})")
    out.append(f"  dir:    {src.location}")
    code = [f"git {(rep.get('git_sha') or '?')[:12]}" + (" (dirty)" if rep.get("git_dirty") else "")]
    if rep.get("image"):
        code.append(f"image {rep['image'].rsplit('/', 1)[-1]}")
    if rep.get("tau2_commit"):
        code.append(f"tau2 {rep['tau2_commit'][:8]}")
    out.append(f"  code:   {', '.join(code)}")
    c = errors.get("counts") or summary.get("status_counts") or {}
    if c:
        out.append(f"  counts: {c.get('tasks', 0)} task runs, {c.get('passed', 0)} passed, {c.get('failed', 0)} failed,"
                   f" {c.get('infra_errors', 0)} infra errors, {c.get('errors_logged', 0)} errors logged,"
                   f" exit {c.get('exit_code')}")
    if not errors:
        out.append("  no errors.json: the run did not finish, or it predates the failure report."
                   + (f" Job log: {index['log']}" if index and index.get("log") else ""))

    first = errors.get("first_error")
    if first:
        out.append("")
        out.append("FIRST ERROR")
        if "message" in first:  # a logged error
            out.append(f"  where:  {_where(first)}" + (f"   (model {first['model']})" if first.get("model") else ""))
            what = f"{first.get('error_class') or ''}: {first.get('error') or ''}".strip(": ")
            out.append(f"  what:   {describe(what or first.get('message'), 300)}")
            out.append(f"  log:    {describe(first.get('message'), 200)}")
            if first.get("last_input"):
                out.append(f"  input:  {describe(first['last_input'], 240)}")
            if first.get("last_output"):
                out.append(f"  output: {describe(first['last_output'], 240)}   (model's raw output before it)")
            for row in _short_stack(first.get("stack")):
                out.append(f"  stack:  {row}")
        else:  # a failed task, no logged error
            out.append(f"  where:  task {first.get('task_id')} trial {first.get('trial')} > turn {first.get('turn')}")
            out.append(f"  what:   {describe(first.get('first_error'), 300)}")
            if first.get("last_model_input"):
                out.append(f"  input:  {describe(first['last_model_input'], 240)}")
            if first.get("last_raw_output"):
                out.append(f"  output: {describe(first['last_raw_output'], 240)}")

    failed = sorted(errors.get("failed_tasks") or [], key=lambda t: (not t.get("infra_error"), str(t.get("task_id"))))
    if failed:
        out.append("")
        out.append(f"FAILED TASKS ({len(failed)})")
        for t in failed if show_all else failed[:10]:
            out.append(f"  task {t.get('task_id')} trial {t.get('trial')}  reward {t.get('reward')}"
                       f"  {t.get('termination_reason')}  turn {t.get('turn')}: {describe(t.get('first_error'), 110)}")
        if len(failed) > 10 and not show_all:
            out.append(f"  ... {len(failed) - 10} more (--all)")

    all_errors = errors.get("errors") or []
    other = all_errors[1:6] if all_errors and all_errors[0] == first else all_errors[:5]
    if other:
        out.append("")
        out.append(f"OTHER ERRORS ({len(all_errors) - (1 if all_errors and all_errors[0] == first else 0)})")
        for e in other:
            out.append(f"  [{_where(e)}] {describe(e.get('error_class') or '', 40)} {describe(e.get('error') or e.get('message'), 120)}")

    rows = timeline(logs)
    if rows:
        out.append("")
        out.append("TIMELINE")
        out.extend(rows)

    links = errors.get("links") or report.links(run_id, rep.get("trace_id"))
    if links:
        out.append("")
        out.append("LINKS")
        for k, v in links.items():
            out.append(f"  {k + ':':7} {v}")
        if index and index.get("log"):
            out.append(f"  joblog: {index['log']}")

    out.append("")
    out.append("NEXT")
    task = (failed[0].get("task_id") if failed else None)
    if task is not None:
        out.append(f"  zoom:   .venv/bin/python scripts/why.py {ref} --task {task}")
        out.append(f"  rerun:  .venv/bin/python scripts/rerun.py {ref} --task {task} --where local-dry-run")
    else:
        out.append(f"  rerun:  .venv/bin/python scripts/rerun.py {ref} --where local-dry-run")
    return "\n".join(out)


def render_task(src: Source, task_id: str, max_chars: int = 300) -> str:
    errors = src.json("errors.json") or {}
    traces = src.jsonl("traces.jsonl", max_bytes=50_000_000)
    out = [f"TASK {task_id}  in {src.location}"]
    if traces is None:
        out.append("  traces.jsonl is missing or over 50 MB.")
        traces = []
    runs = [t for t in traces if str(t.get("task_id")) == str(task_id)]
    if not runs and traces:
        out.append(f"  no task {task_id} here (tasks: {', '.join(sorted({str(t.get('task_id')) for t in traces}))})")
    for t in runs:
        out.append("")
        out.append(f"trial {t.get('trial')}: reward {t.get('reward')}, {t.get('termination_reason')},"
                   f" {t.get('num_messages')} messages" + (f", error: {describe(t.get('error'), 200)}" if t.get("error") else ""))
        turns = {a.get("message_index"): (n, a) for n, a in enumerate(t.get("agent_turns") or [])}
        for i, m in enumerate(t.get("messages") or []):
            role = m.get("role")
            if role == "assistant" and i in turns:
                n, a = turns[i]
                text = a.get("raw_output") if a.get("raw_output") is not None else m.get("content")
                flag = f"  <-- {', '.join(a['errors'])}" if a.get("errors") else ""
                out.append(f"  [{i:3}] agent turn {n}: {describe(text, max_chars)}{flag}")
            elif role == "tool":
                mark = " (ERROR)" if m.get("error") else ""
                out.append(f"  [{i:3}] tool{mark}: {describe(m.get('content'), max_chars)}")
            else:
                out.append(f"  [{i:3}] {role}: {describe(m.get('content'), max_chars)}")
    for e in errors.get("errors") or []:
        if str(e.get("task_id")) == str(task_id):
            out.append("")
            out.append(f"LOGGED ERROR [{_where(e)}] {e.get('error_class')}: {describe(e.get('error'), 300)}")
            for row in _short_stack(e.get("stack"), 8):
                out.append(f"  {row}")
            break
    entry = next((f for f in errors.get("failed_tasks") or [] if str(f.get("task_id")) == str(task_id)), None)
    links = (entry or {}).get("links") or report.links(errors.get("run_id"), errors.get("trace_id"), task_id=str(task_id))
    if links:
        out.append("")
        out.append("LINKS")
        for k, v in links.items():
            out.append(f"  {k + ':':7} {v}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref", help="run dir, gs:// path, job index (gs://.../jobs/<id>.json) or run id")
    p.add_argument("--task", default=None, help="zoom into one task id")
    p.add_argument("--all", action="store_true", help="list every failed task")
    args = p.parse_args(argv)
    try:
        sources, index = resolve(args.ref)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    if not sources:
        print(f"{args.ref}: the job wrote no run dir." + (f" Job log: {index.get('log')}" if index else ""))
        if index:
            print(f"exit {index.get('exit_code')}; links: {index.get('links')}")
        return 1
    for n, src in enumerate(sources):
        if n:
            print("\n" + "-" * 72 + "\n")
        print(render_task(src, args.task) if args.task else render(src, index, args.all, args.ref if len(sources) == 1 else src.location))
    return 0


if __name__ == "__main__":
    sys.exit(main())

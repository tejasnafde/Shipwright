"""Offline check of the tool-call guards (src/shipwright/llm/guards.py) on recorded runs.

For every agent tool call in each run dir's traces.jsonl, the guards are run on the
conversation as it was just before that call. Reports, per run dir:

  passed runs   calls the guards would have rejected (false alarms: these runs were right)
  failed runs   calls the guards would have rejected (what they would catch)

Use it on a teacher run (no false alarm should block a passing teacher run) and on a
student baseline. No network, no credits.

  .venv/bin/python scripts/check_guards.py runs/r1-teacher-t1 runs/r1-baseline-train
  .venv/bin/python scripts/check_guards.py runs/r1-baseline-train --tasks-passed-by runs/r1-teacher-t1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal.sft_data import trace_to_openai  # noqa: E402
from shipwright.llm import guards  # noqa: E402


@dataclass
class GuardReport:
    runs: int = 0
    runs_with_rejection: int = 0
    calls: int = 0
    rejections: Counter = field(default_factory=Counter)
    examples: list = field(default_factory=list)


def parse_options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directories", type=Path, nargs="+")
    parser.add_argument("--tasks-passed-by", type=Path, action="append", default=[],
                        help="only count tasks that this run dir passed at least once (repeatable)")
    parser.add_argument("--examples", type=int, default=3, help="example rejections to print per group")
    return parser.parse_args(argv)


def read_traces(run_directory: Path) -> list[dict]:
    traces = []
    with open(run_directory / "traces.jsonl") as trace_file:
        for line in trace_file:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def tasks_passed(run_directories: list[Path]) -> set[str]:
    passed = set()
    for run_directory in run_directories:
        for trace in read_traces(run_directory):
            if trace.get("passed"):
                passed.add(str(trace["task_id"]))
    return passed


def check_trace(trace: dict, report: GuardReport, max_examples: int) -> None:
    messages = trace_to_openai(trace["messages"])
    report.runs += 1
    rejected_in_this_run = False
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            report.calls += 1
            call = guards.tool_call_from_message(tool_call)
            violation = guards.first_violation(messages[:index], call)
            if violation is None:
                continue
            report.rejections[violation.rule] += 1
            rejected_in_this_run = True
            if len(report.examples) < max_examples:
                arguments = json.dumps(call.arguments)[:100]
                report.examples.append(f"task {trace['task_id']}: {violation.rule}: {call.name}({arguments})")
    if rejected_in_this_run:
        report.runs_with_rejection += 1


def print_report(title: str, report: GuardReport) -> None:
    print(f"  {title}: {report.runs} runs, {report.calls} tool calls; "
          f"{report.runs_with_rejection} runs with a rejection; by rule {dict(report.rejections)}")
    for example in report.examples:
        print(f"      {example}")


def main(argv: list[str] | None = None) -> int:
    options = parse_options(argv)
    only_tasks = tasks_passed(options.tasks_passed_by) if options.tasks_passed_by else None
    for run_directory in options.run_directories:
        passed_report, failed_report = GuardReport(), GuardReport()
        for trace in read_traces(run_directory):
            if trace.get("termination_reason") == "infrastructure_error":
                continue
            if only_tasks is not None and str(trace["task_id"]) not in only_tasks:
                continue
            if trace.get("passed"):
                check_trace(trace, passed_report, options.examples)
            else:
                check_trace(trace, failed_report, options.examples)
        print(run_directory.name)
        print_report("passed runs (rejections here are false alarms)", passed_report)
        print_report("failed runs (rejections here are catches)", failed_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

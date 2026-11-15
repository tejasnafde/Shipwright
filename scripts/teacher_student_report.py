"""Teacher vs student on the same tasks: how much of the teacher's lead the trained student closes.

Reads three run dirs that ran the SAME tasks with the same harness (prompted tools, 8192 window,
same user simulator): the teacher (the ceiling stage), the base student and the trained student
(the two sides of a gate run). No network, no credits.

  .venv/bin/python scripts/teacher_student_report.py \\
      --teacher runs/r9-ceiling --base runs/gate-r8/base --student runs/gate-r8/r8-weight

Reports, per system: pass^1 (AuditKIT, the gate's own scoring) with a 95% interval, pass^k,
tasks solved at least once, and runs cut short by a leaking user simulator. Then:

  gap closed = (student pass^1 - base pass^1) / (teacher pass^1 - base pass^1)

with a 95% bootstrap interval over tasks, an exact McNemar test of base vs student on the same
(task, trial) pairs, and a per-task table. ``--drop-simulator-leaks`` scores without the runs in
which the user simulator wrote past its turn (see patches.cut_leaked_turn): use it for runs made
before that fix, and say so when you quote the number.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal.eval_metrics import pass_metrics, reliability_rows  # noqa: E402
from shipwright.heal.traces import has_simulator_leak, read_traces  # noqa: E402

INFRA_ERROR = "infrastructure_error"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 300


@dataclass
class SystemRuns:
    label: str
    run_directory: Path
    traces: list[dict]
    leaked_runs: int = 0
    infra_errors: int = 0
    metrics: dict = field(default_factory=dict)


def parse_options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", type=Path, required=True, help="teacher run dir (shipwright stage ceiling)")
    parser.add_argument("--base", type=Path, required=True, help="base student run dir (gate's current side)")
    parser.add_argument("--student", type=Path, required=True, help="trained student run dir (gate's candidate side)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--drop-simulator-leaks", action="store_true",
                        help="score without runs where the user simulator wrote past its turn")
    parser.add_argument("--output", type=Path, help="also write the report as JSON here")
    return parser.parse_args(argv)






def is_passed(trace: dict) -> bool:
    return trace.get("passed") in (True, "True", "true", 1)


def load_system(label: str, run_directory: Path, task_ids: set[str] | None, drop_leaks: bool, k: int) -> SystemRuns:
    system = SystemRuns(label=label, run_directory=run_directory, traces=[])
    for trace in read_traces(run_directory):
        if task_ids is not None and str(trace.get("task_id")) not in task_ids:
            continue
        if trace.get("termination_reason") == INFRA_ERROR:
            system.infra_errors += 1
            continue
        if has_simulator_leak(trace):
            system.leaked_runs += 1
            if drop_leaks:
                continue
        system.traces.append(trace)
    system.metrics = pass_metrics(reliability_rows(system.traces, k), k)
    return system


def task_pass_rates(system: SystemRuns) -> dict[str, float]:
    rates = {}
    for task_id, task in system.metrics["per_task"].items():
        if task["n_decided"]:
            rates[str(task_id)] = task["n_success"] / task["n_decided"]
    return rates


def gap_closed(teacher: float, base: float, student: float) -> float | None:
    if teacher - base <= 0:
        return None
    return (student - base) / (teacher - base)


def bootstrap_gap_interval(teacher: SystemRuns, base: SystemRuns, student: SystemRuns) -> tuple[float, float] | None:
    """95% interval of gap closed, resampling tasks (each system's pass^1 is a mean over tasks)."""
    teacher_rates, base_rates, student_rates = task_pass_rates(teacher), task_pass_rates(base), task_pass_rates(student)
    shared_tasks = sorted(set(teacher_rates) & set(base_rates) & set(student_rates))
    if len(shared_tasks) < 2:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        drawn = [generator.choice(shared_tasks) for _ in shared_tasks]
        value = gap_closed(
            sum(teacher_rates[task] for task in drawn) / len(drawn),
            sum(base_rates[task] for task in drawn) / len(drawn),
            sum(student_rates[task] for task in drawn) / len(drawn),
        )
        if value is not None:
            samples.append(value)
    if not samples:
        return None
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]


def mcnemar_exact(base: SystemRuns, student: SystemRuns) -> dict:
    """Base vs student on the same (task, trial): only the pairs where exactly one passed count."""
    base_by_pair = {(str(trace["task_id"]), str(trace.get("trial"))): is_passed(trace) for trace in base.traces}
    student_by_pair = {(str(trace["task_id"]), str(trace.get("trial"))): is_passed(trace) for trace in student.traces}
    shared_pairs = set(base_by_pair) & set(student_by_pair)
    only_student = sum(1 for pair in shared_pairs if student_by_pair[pair] and not base_by_pair[pair])
    only_base = sum(1 for pair in shared_pairs if base_by_pair[pair] and not student_by_pair[pair])
    discordant = only_student + only_base
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(only_student, only_base)
        p_value = min(1.0, 2 * sum(comb(discordant, index) for index in range(smaller + 1)) / 2 ** discordant)
    return {"pairs": len(shared_pairs), "only_student_passed": only_student, "only_base_passed": only_base,
            "p_value": p_value}


def percent(value: float | None) -> str:
    return "  n/a" if value is None else f"{100 * value:5.1f}%"


def solved_at_least_once(system: SystemRuns) -> int:
    return sum(1 for task in system.metrics["per_task"].values() if task["n_success"] > 0)


def print_report(systems: list[SystemRuns], report: dict) -> None:
    print(f"{'system':10s} {'runs':>5s} {'pass^1':>7s} {'95% interval':>16s} {'pass^k':>7s} "
          f"{'solved>=1':>9s} {'leaks':>6s} {'infra':>6s}  run dir")
    for system in systems:
        metrics = system.metrics
        interval = metrics.get("pass_1_ci")
        interval_text = f"{percent(interval[0])}-{percent(interval[1]).strip()}" if interval else "n/a"
        print(f"{system.label:10s} {len(system.traces):5d} {percent(metrics['pass_1']):>7s} {interval_text:>16s} "
              f"{percent(metrics['pass_k']):>7s} {solved_at_least_once(system):9d} {system.leaked_runs:6d} "
              f"{system.infra_errors:6d}  {system.run_directory}")
    gap = report["gap_closed"]
    interval = report["gap_closed_interval"]
    print()
    print(f"gap closed (pass^1): {percent(gap)}"
          + (f"  (95% bootstrap over tasks: {percent(interval[0]).strip()} to {percent(interval[1]).strip()})" if interval else ""))
    test = report["mcnemar_base_vs_student"]
    print(f"McNemar, base vs student on {test['pairs']} shared (task, trial) pairs: student only "
          f"{test['only_student_passed']}, base only {test['only_base_passed']}, exact p = {test['p_value']:.3f}")
    print()
    print("per task (passes / decided trials):")
    print(f"  {'task':>5s}  {'teacher':>8s} {'base':>6s} {'student':>8s}")
    for task_id, row in report["per_task"].items():
        print(f"  {task_id:>5s}  {row['teacher']:>8s} {row['base']:>6s} {row['student']:>8s}")


def per_task_table(teacher: SystemRuns, base: SystemRuns, student: SystemRuns) -> dict:
    all_tasks = set(teacher.metrics["per_task"]) | set(base.metrics["per_task"]) | set(student.metrics["per_task"])
    table = {}
    for task_id in sorted(all_tasks, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        row = {}
        for label, system in (("teacher", teacher), ("base", base), ("student", student)):
            task = system.metrics["per_task"].get(task_id)
            row[label] = f"{task['n_success']}/{task['n_decided']}" if task else "-"
        table[str(task_id)] = row
    return table


def main(argv: list[str] | None = None) -> int:
    options = parse_options(argv)
    base = load_system("base", options.base, None, options.drop_simulator_leaks, options.k)
    task_ids = {str(trace["task_id"]) for trace in read_traces(options.base)}
    teacher = load_system("teacher", options.teacher, task_ids, options.drop_simulator_leaks, options.k)
    student = load_system("student", options.student, task_ids, options.drop_simulator_leaks, options.k)
    systems = [teacher, base, student]
    report = {
        "drop_simulator_leaks": options.drop_simulator_leaks,
        "systems": {system.label: {"run_dir": str(system.run_directory), "runs": len(system.traces),
                                   "leaked_runs": system.leaked_runs, "infra_errors": system.infra_errors,
                                   "pass_1": system.metrics["pass_1"], "pass_1_ci": system.metrics["pass_1_ci"],
                                   "pass_k": system.metrics["pass_k"],
                                   "solved_at_least_once": solved_at_least_once(system)}
                    for system in systems},
        "gap_closed": gap_closed(teacher.metrics["pass_1"] or 0, base.metrics["pass_1"] or 0,
                                 student.metrics["pass_1"] or 0),
        "gap_closed_interval": bootstrap_gap_interval(teacher, base, student),
        "mcnemar_base_vs_student": mcnemar_exact(base, student),
        "per_task": per_task_table(teacher, base, student),
    }
    print_report(systems, report)
    if options.output:
        options.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

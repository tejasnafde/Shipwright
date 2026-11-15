"""Which tasks the teacher replays: the failed tasks of the fixable routes, minus the ones it never solves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import teacher_runs  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_failed_task_ids_take_both_fixable_routes_and_skip_no_fix(tmp_path):
    write_jsonl(tmp_path / "failures.jsonl", [
        {"task_id": 1, "route": "tool_fix"},
        {"task_id": 2, "route": "weight_fix"},
        {"task_id": 3, "route": "no_fix"},
        {"task_id": 1, "route": "tool_fix"},
    ])
    assert teacher_runs.failed_task_ids(tmp_path) == ["1", "2"]
    assert teacher_runs.failed_task_ids(tmp_path, ("weight_fix",)) == ["2"]


def test_unsolved_tasks_need_three_decided_failures_and_no_pass(tmp_path):
    trace = lambda task_id, passed, reason="user_stop": {  # noqa: E731
        "task_id": task_id, "passed": passed, "termination_reason": reason}
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    write_jsonl(teacher_dir / "traces.jsonl", [
        *[trace(1, False)] * 3,                                         # never solved: skipped
        *[trace(2, False)] * 2, trace(2, False, "infrastructure_error"),  # only 2 decided trials: kept
        *[trace(3, False)] * 2, trace(3, True),                         # solved once: kept
    ])
    assert teacher_runs.unsolved_task_ids([teacher_dir]) == {"1"}


def test_resolve_task_ids_skips_unsolved(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    write_jsonl(baseline_dir / "failures.jsonl", [
        {"task_id": 1, "route": "weight_fix"}, {"task_id": 2, "route": "tool_fix"}])
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    write_jsonl(teacher_dir / "traces.jsonl", [
        {"task_id": 1, "passed": False, "termination_reason": "user_stop"}] * 3)
    options = teacher_runs.parse_args([
        "--teacher-model", "t", "--user-model", "u", "--from-run", str(baseline_dir),
        "--skip-unsolved-from", str(teacher_dir)])
    assert teacher_runs.resolve_task_ids(options) == ["2"]


def test_solved_tasks_are_skipped(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    write_jsonl(baseline_dir / "failures.jsonl", [
        {"task_id": 1, "route": "weight_fix"}, {"task_id": 2, "route": "tool_fix"}])
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    write_jsonl(teacher_dir / "traces.jsonl", [
        {"task_id": 1, "passed": True, "termination_reason": "user_stop"},
        {"task_id": 2, "passed": False, "termination_reason": "user_stop"}])
    options = teacher_runs.parse_args([
        "--teacher-model", "t", "--user-model", "u", "--from-run", str(baseline_dir),
        "--skip-solved-from", str(teacher_dir)])
    assert teacher_runs.resolve_task_ids(options) == ["2"]


def test_harvest_keeps_only_solved_tasks(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    write_jsonl(baseline_dir / "failures.jsonl", [
        {"task_id": 1, "route": "weight_fix"}, {"task_id": 2, "route": "tool_fix"}])
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    write_jsonl(teacher_dir / "traces.jsonl", [
        {"task_id": 1, "passed": True, "termination_reason": "user_stop"},
        {"task_id": 2, "passed": False, "termination_reason": "user_stop"}])
    options = teacher_runs.parse_args([
        "--teacher-model", "t", "--user-model", "u", "--from-run", str(baseline_dir),
        "--only-solved-from", str(teacher_dir)])
    assert teacher_runs.resolve_task_ids(options) == ["1"]


def test_min_teacher_rate_keeps_the_gap_tasks_best_first(tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    write_jsonl(baseline_dir / "failures.jsonl", [
        {"task_id": task_id, "route": "weight_fix"} for task_id in (1, 2, 3, 4)])
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    decided = lambda task_id, passed: {"task_id": task_id, "passed": passed, "termination_reason": "user_stop"}  # noqa: E731
    write_jsonl(teacher_dir / "traces.jsonl", [
        decided(1, True), decided(1, False), decided(1, False),  # 1/3: below the bar
        decided(2, True), decided(2, True),                      # 2/2
        decided(3, True), decided(3, False),                     # 1/2
        decided(4, False),                                       # never passed
    ])
    options = teacher_runs.parse_args([
        "--teacher-model", "t", "--user-model", "u", "--from-run", str(baseline_dir),
        "--only-solved-from", str(teacher_dir), "--min-teacher-rate", "0.5"])
    assert teacher_runs.resolve_task_ids(options) == ["2", "3"]


def test_ceiling_runs_the_heldout_tasks_on_the_test_split():
    options = teacher_runs.parse_args(["--teacher-model", "t", "--user-model", "u", "--ceiling"])
    task_ids = teacher_runs.resolve_task_ids(options)
    assert len(task_ids) == 20
    argv = teacher_runs.build_baseline_argv(options, task_ids)
    assert argv[argv.index("--task-split") + 1] == "test"


def test_hidden_ceiling_opens_exactly_the_hidden_tasks_and_keeps_nothing_for_training():
    from shipwright.heal import splits

    options = teacher_runs.parse_args(["--teacher-model", "t", "--user-model", "u", "--hidden-ceiling"])
    task_ids = teacher_runs.resolve_task_ids(options)
    assert task_ids == splits.hidden_ids() and not set(task_ids) & set(splits.heldout_ids())
    argv = teacher_runs.build_baseline_argv(options, task_ids)
    assert argv[argv.index("--task-split") + 1] == "test" and "--open-hidden" in argv
    assert teacher_runs.is_ceiling_run(options)

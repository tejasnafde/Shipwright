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

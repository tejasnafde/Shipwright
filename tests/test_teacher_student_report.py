"""scripts/teacher_student_report.py: gap closed, McNemar and the simulator-leak filter. Offline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("teacher_student_report", ROOT / "scripts" / "teacher_student_report.py")
report = importlib.util.module_from_spec(spec)
sys.modules["teacher_student_report"] = report
spec.loader.exec_module(report)


def run(task_id, trial, passed, user_text="Yes, go ahead."):
    return {"task_id": task_id, "trial": trial, "passed": passed,
            "messages": [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": user_text}]}


def test_gap_closed():
    assert abs(report.gap_closed(teacher=0.6, base=0.1, student=0.35) - 0.5) < 1e-9
    assert report.gap_closed(teacher=0.1, base=0.1, student=0.2) is None  # no gap to close


def test_simulator_leak_is_found():
    assert report.has_simulator_leak(run("1", "0", False, "Yes.<|end|><|start|>assistant ###STOP###"))
    assert not report.has_simulator_leak(run("1", "0", False, "Yes. ###STOP###"))


def test_mcnemar_counts_only_discordant_pairs():
    base = report.SystemRuns("base", Path("."), [run("1", "0", False), run("1", "1", True), run("2", "0", False)])
    student = report.SystemRuns("student", Path("."), [run("1", "0", True), run("1", "1", True), run("2", "0", True)])
    test = report.mcnemar_exact(base, student)
    assert (test["pairs"], test["only_student_passed"], test["only_base_passed"]) == (3, 2, 0)
    assert abs(test["p_value"] - 0.5) < 1e-9  # 2 discordant pairs, both one way: 2 * (1/4)

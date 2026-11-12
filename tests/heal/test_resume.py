"""A crashing task becomes an infra error (the run goes on), and a rerun into the same dir
resumes: decided runs are kept, infra errors are run again (tau2 auto_resume, scripts/baseline.py).
From Tejas's branch (the Agent-R half of the test was dropped with Agent-R)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tau2")
ROOT = Path(__file__).resolve().parents[2]


def run(output_directory: Path, *extra_arguments: str, check: bool = True) -> dict:
    subprocess.run([sys.executable, "scripts/baseline.py", "--dry-run", "--num-tasks", "3",
                    "--output-dir", str(output_directory), *extra_arguments],
                   cwd=ROOT, check=check, capture_output=True, timeout=300)
    return json.loads((output_directory / "summary.json").read_text())


def simulation_ids(output_directory: Path) -> list[str]:
    lines = (output_directory / "traces.jsonl").read_text().splitlines()
    return sorted(json.loads(line)["simulation_id"] for line in lines)


def test_crash_is_isolated_and_rerun_resumes(tmp_path):
    output_directory = tmp_path / "run"
    first = run(output_directory, "--fail-at", "agent_call", check=False)  # every task's agent crashes (exit 1)
    assert first["num_runs"] == 3 and first["num_infra_errors"] == 3

    second = run(output_directory)  # the infra errors are run again
    assert second["num_runs"] == 3 and second["num_infra_errors"] == 0
    decided = simulation_ids(output_directory)

    third = run(output_directory)  # nothing left to do: the decided runs are kept as they are
    assert third["num_runs"] == 3 and simulation_ids(output_directory) == decided

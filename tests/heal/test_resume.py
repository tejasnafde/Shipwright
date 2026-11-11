"""A crashing task becomes an infra error (the run goes on), and a rerun into the same dir
resumes: decided runs are kept, infra errors are run again (tau2 auto_resume, scripts/baseline.py)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tau2")
ROOT = Path(__file__).resolve().parents[2]


def run(out: Path, *extra: str, check: bool = True) -> dict:
    subprocess.run([sys.executable, "scripts/baseline.py", "--dry-run", "--num-tasks", "3", "--output-dir", str(out),
                    *extra], cwd=ROOT, check=check, capture_output=True, timeout=300)
    return json.loads((out / "summary.json").read_text())


def sim_ids(out: Path) -> list[str]:
    return sorted(json.loads(line)["simulation_id"] for line in (out / "traces.jsonl").read_text().splitlines())


def test_crash_is_isolated_and_rerun_resumes(tmp_path):
    out = tmp_path / "run"
    first = run(out, "--fail-at", "agent_call", check=False)  # every task's agent crashes (exit 1)
    assert first["num_runs"] == 3 and first["num_infra_errors"] == 3

    second = run(out)  # the infra errors are run again
    assert second["num_runs"] == 3 and second["num_infra_errors"] == 0
    decided = sim_ids(out)

    third = run(out)  # nothing left to do: the decided runs are kept as they are
    assert third["num_runs"] == 3 and sim_ids(out) == decided


def search(out: Path, *extra: str, check: bool = True) -> dict:
    subprocess.run([sys.executable, "scripts/agent_r_search.py", "--dry-run", "--user-model", "dry-user",
                    "--task-ids", "0", "1", "--simulations", "4", "--output-dir", str(out), *extra],
                   cwd=ROOT, check=check, capture_output=True, timeout=300)
    return json.loads((out / "search_summary.json").read_text())


def test_agent_r_search_isolates_a_task_crash_and_resumes(tmp_path):
    out = tmp_path / "search"
    first = search(out, "--fail-at", "search_task", check=False)
    assert all("error" in row for row in first["per_task"].values())  # both crashed, the run finished

    second = search(out)
    assert second["rollouts"] > 0 and not any(r.get("resumed") for r in second["per_task"].values())

    third = search(out)  # both tasks are done: loaded, not searched again
    assert all(r.get("resumed") for r in third["per_task"].values())
    assert third["rollouts"] == second["rollouts"]

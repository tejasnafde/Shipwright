"""The round driver's pure parts: which arguments each cloud stage passes, and the cost bookkeeping."""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
specification = importlib.util.spec_from_file_location("run_round", ROOT / "scripts" / "run_round.py")
run_round = importlib.util.module_from_spec(specification)
sys.modules["run_round"] = run_round
specification.loader.exec_module(run_round)


def make_options(**overrides) -> argparse.Namespace:
    values = dict(round=2, owner="ishank", user_model="u", teacher_model="t", max_concurrency=4, dry_run=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_baseline_runs_the_pure_base_on_the_train_split_with_a_cost_cap():
    arguments, fetched, output_dir = run_round.cloud_stage_arguments("baseline", make_options())
    assert "--tool-patches" not in arguments  # the base system has no patches
    assert arguments[arguments.index("--task-split") + 1] == "train"
    assert arguments[arguments.index("--max-cost-usd") + 1] == str(run_round.BUDGETS["baseline"].cap_usd)
    assert fetched == [] and output_dir == "r2-baseline"


def test_teacher_reads_the_baseline_failures_and_is_capped():
    arguments, fetched, output_dir = run_round.cloud_stage_arguments("teacher", make_options())
    assert arguments[arguments.index("--from-run") + 1] == "runs/r2-baseline"
    assert "--trials=3" in arguments
    assert "--max-cost-usd" in arguments
    assert fetched == ["r2-baseline"] and output_dir == "r2-teacher"


def test_every_paid_stage_has_a_cap_above_its_estimate():
    for stage in ("baseline", "classify", "teacher", "train", "gate"):
        budget = run_round.BUDGETS[stage]
        assert budget.cap_usd > budget.estimate_usd


def test_unmetered_stages_count_their_estimate_as_spent(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    manifest = {"stages": {
        "baseline": {"actual_usd": 1.2, "estimate_usd": 1.5},
        "classify": {"actual_usd": None, "estimate_usd": 0.05},
    }}
    run_round.save_manifest(manifest, 2)
    assert manifest["spent_usd"] == 1.25

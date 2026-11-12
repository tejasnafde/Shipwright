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
    values = dict(round=2, owner="ishank", user_model="u", teacher_model="t", max_concurrency=4, dry_run=False,
                  skip_unsolved_from=[], skip_solved_from=[], only_solved_from=[], min_teacher_rate=0.0, teacher_trials=3, extra_teacher_runs=[], force=False)
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
    assert fetched == ["ishank/r2-baseline"] and output_dir == "r2-teacher"


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


def test_teacher_can_skip_the_tasks_earlier_teacher_runs_never_solved():
    options = make_options(skip_unsolved_from=["r2-teacher"])
    arguments, fetched, _ = run_round.cloud_stage_arguments("teacher", options)
    assert arguments[arguments.index("--skip-unsolved-from") + 1] == "runs/r2-teacher"
    assert fetched == ["ishank/r2-baseline", "ishank/r2-teacher"]


def test_teacher_skips_solved_tasks_and_fetches_their_run_dirs():
    options = make_options(skip_unsolved_from=["r2-teacher"], skip_solved_from=["r1-teacher-t1", "r2-teacher"])
    arguments, fetched, _ = run_round.cloud_stage_arguments("teacher", options)
    assert arguments[arguments.index("--skip-solved-from") + 1:][:2] == ["runs/r1-teacher-t1", "runs/r2-teacher"]
    assert fetched == ["ishank/r2-baseline", "ishank/r2-teacher", "ishank/r1-teacher-t1"]


def test_manifest_splits_metered_and_estimated_spend(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    manifest = {"stages": {
        "baseline": {"actual_usd": 1.25, "estimate_usd": 1.5},
        "train": {"actual_usd": None, "estimate_usd": 0.4},
    }}
    run_round.save_manifest(manifest, 9)
    assert manifest["spent_metered_usd"] == 1.25 and manifest["spent_estimated_usd"] == 0.4
    assert manifest["spent_usd"] == 1.65
    assert manifest["stages"]["baseline"]["cost_basis"] == "metered"
    assert manifest["stages"]["train"]["cost_basis"] == "estimated"


def test_health_check_stops_a_round_after_too_many_infra_errors_or_too_few_kept_traces():
    import pytest

    flaky = {"stages": {"baseline": {"num_runs": 74, "num_infra_errors": 30}}}
    with pytest.raises(SystemExit, match="infra errors"):
        run_round.check_previous_stage_is_healthy(flaky, "classify", "baseline")
    healthy = {"stages": {"baseline": {"num_runs": 74, "num_infra_errors": 9}}}
    run_round.check_previous_stage_is_healthy(healthy, "classify", "baseline")
    small = {"stages": {"data": {"examples": 21}}}
    with pytest.raises(SystemExit, match="21 examples"):
        run_round.check_previous_stage_is_healthy(small, "train", "data")
    run_round.check_previous_stage_is_healthy({"stages": {"data": {"examples": 140}}}, "train", "data")


def test_image_check_waits_for_the_job_to_be_repointed(monkeypatch):
    import subprocess

    def fake_run_command(command):
        output = {"rev-parse": "abc1234\n", "status": "", "rev-list": "0\n"}[command[3]]
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    job_digests = iter(["sha256:old", "sha256:old", "sha256:new"])
    sleeps = []
    monkeypatch.setattr(run_round, "run_command", fake_run_command)
    monkeypatch.setattr(run_round, "commit_image_digest", lambda commit: "sha256:new")
    monkeypatch.setattr(run_round, "job_image_digest", lambda owner: next(job_digests))
    monkeypatch.setattr(run_round.time, "sleep", sleeps.append)
    run_round.check_cloud_job_runs_this_commit("ishank", wait_minutes=20)
    assert len(sleeps) == 2  # two polls saw the old image, the third saw the new one


def test_harvest_plays_only_solved_tasks_with_more_trials():
    options = make_options(only_solved_from=["r1-teacher-t1"], teacher_trials=5)
    arguments, fetched, _ = run_round.cloud_stage_arguments("teacher", options)
    assert arguments[arguments.index("--only-solved-from") + 1] == "runs/r1-teacher-t1"
    assert "--trials=5" in arguments and "ishank/r1-teacher-t1" in fetched


def test_ceiling_runs_the_teacher_on_the_heldout_tasks_with_a_cap():
    arguments, fetched, output_dir = run_round.cloud_stage_arguments("ceiling", make_options())
    assert "--ceiling" in arguments and "--trials=3" in arguments
    assert arguments[arguments.index("--max-cost-usd") + 1] == str(run_round.BUDGETS["ceiling"].cap_usd)
    assert fetched == [] and output_dir == "r2-ceiling"


def test_harvest_passes_the_gap_threshold():
    options = make_options(only_solved_from=["r1-teacher-t1"], min_teacher_rate=0.5)
    arguments, _, _ = run_round.cloud_stage_arguments("teacher", options)
    assert arguments[arguments.index("--min-teacher-rate") + 1] == "0.5"


def test_no_cloud_stage_repeats_a_value_in_its_job_arguments():
    options = make_options(user_model="vertex/openai/gpt-oss-120b-maas", teacher_model="vertex/deepseek-ai/deepseek-v4.1-flash-maas",
                           only_solved_from=["r1-teacher-t1", "r1-teacher-ds-a"], min_teacher_rate=0.5, teacher_trials=5)
    for stage in ("baseline", "classify", "teacher", "ceiling"):
        arguments, fetched, _ = run_round.cloud_stage_arguments(stage, options)
        job_arguments = [f"--fetch={name}" for name in fetched] + arguments
        assert len(job_arguments) == len(set(job_arguments)), stage


def test_pool_names_can_belong_to_another_owner():
    options = make_options(only_solved_from=["tejas/r1-teacher-t1", "r2-teacher"], min_teacher_rate=0.5)
    arguments, fetched, _ = run_round.cloud_stage_arguments("teacher", options)
    assert fetched == ["ishank/r2-baseline", "tejas/r1-teacher-t1", "ishank/r2-teacher"]
    assert arguments[arguments.index("--only-solved-from") + 1:][:2] == ["runs/r1-teacher-t1", "runs/r2-teacher"]

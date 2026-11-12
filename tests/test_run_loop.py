"""The loop's decisions: what a round costs, when it may start, what a rerun skips, and the recipe it keeps."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
specification = importlib.util.spec_from_file_location("run_loop", ROOT / "scripts" / "run_loop.py")
run_loop = importlib.util.module_from_spec(specification)
sys.modules["run_loop"] = run_loop
specification.loader.exec_module(run_loop)


def test_a_step_that_does_not_fit_in_the_budget_is_refused():
    manifest = {"stages": {"candidate_train": {"actual_usd": 0.6}}}
    run_loop.check_fits_in_budget(manifest, "classify", budget_usd=5.0)
    with pytest.raises(SystemExit, match="passes the budget"):
        run_loop.check_fits_in_budget(manifest, "gate", budget_usd=1.2)
    with pytest.raises(SystemExit, match="passes the budget"):  # earlier rounds' spend counts
        run_loop.check_fits_in_budget(manifest, "classify", budget_usd=2.0, spent_in_earlier_rounds=1.4)


def test_a_round_needs_its_estimate_or_a_little_more_than_the_last_one_cost():
    estimate = sum(run_loop.STEP_ESTIMATES.values())
    assert run_loop.cost_of_a_full_round(None) == pytest.approx(estimate)
    assert run_loop.cost_of_a_full_round(1.0) == pytest.approx(estimate)
    assert run_loop.cost_of_a_full_round(estimate * 2) == pytest.approx(estimate * 2.2)


def test_a_rerun_skips_finished_steps_unless_told_to_redo_them():
    manifest = {"stages": {"data": {"exit_code": 0}, "train": {"exit_code": 1}}}
    options = argparse.Namespace(redo=[])
    assert run_loop.step_is_done(manifest, "data", options)
    assert not run_loop.step_is_done(manifest, "train", options)  # failed: run again
    assert not run_loop.step_is_done(manifest, "classify", options)  # never run
    assert not run_loop.step_is_done(manifest, "data", argparse.Namespace(redo=["data"]))


def test_the_chosen_recipe_is_kept_in_the_manifest_so_a_rerun_uses_the_same_one():
    decision = {
        "checks": [{"name": "bfcl", "passed": False}],
        "metrics_current": {"label_rates": {}}, "metrics_candidate": {"label_rates": {}},
    }
    manifest: dict = {"stages": {}}
    options = argparse.Namespace(from_round=99)  # no manifest for round 99: the default recipe is the start
    first = run_loop.next_recipe_for(manifest, options, decision)
    assert first.general_replay == 100 and manifest["recipe"]["general_replay"] == 100
    again = run_loop.next_recipe_for(manifest, options, {"checks": []})  # even a different decision
    assert again.general_replay == 100 and again.reasons == first.reasons


def test_the_candidate_config_of_the_next_round_points_at_its_own_adapter(tmp_path, monkeypatch):
    configs = tmp_path / "configs" / "systems"
    configs.mkdir(parents=True)
    (configs / "r3-weight.yaml").write_text(
        "# Round 3 weight-fix candidate\nname: r3-weight\nadapter:\n  repo: x/y\n  revision: round-3\nroute: weight_fix\n")
    calls = []
    monkeypatch.setattr(run_loop, "ROOT", tmp_path)
    def fake_run(command, **keyword_arguments):
        calls.append(command)
        return argparse.Namespace(returncode=1)  # `git diff --cached --quiet` says: the file changed

    monkeypatch.setattr(run_loop.subprocess, "run", fake_run)
    run_loop.write_candidate_config(argparse.Namespace(round=4, from_round=3))
    written = (configs / "r4-weight.yaml").read_text()
    assert "name: r4-weight" in written and "revision: round-4" in written and "# Round 4" in written
    commit = next(command for command in calls if "commit" in command)
    assert commit[-2:] == ["--", "configs/systems/r4-weight.yaml"]  # only its own file is committed


def test_a_copied_run_dir_holds_real_files_not_links_into_a_cache(tmp_path):
    cache = tmp_path / "cache" / "blobs"
    cache.mkdir(parents=True)
    (cache / "abc").write_text('{"task_id": 1}\n')
    snapshot = tmp_path / "cache" / "snapshots" / "run"
    snapshot.mkdir(parents=True)
    (snapshot / "traces.jsonl").symlink_to(cache / "abc")
    destination = tmp_path / "runs" / "r4-cand-train"
    run_loop.copy_run_directory(snapshot, destination)
    assert not (destination / "traces.jsonl").is_symlink()
    assert (destination / "traces.jsonl").read_text() == '{"task_id": 1}\n'


def test_other_costs_count_as_spent():
    manifest = {"stages": {"gate": {"actual_usd": 0.5}}, "other_costs": [{"usd": 0.37}]}
    assert run_loop.spent_so_far(manifest) == pytest.approx(0.87)


def test_every_command_the_loop_builds_parses_in_the_script_it_calls():
    import train_job

    options = argparse.Namespace(round=5)
    chosen = run_loop.recipe_module.Recipe(epochs=2.0, learning_rate=1e-4, boost=3)
    launch_options, training_arguments = train_job.parse_args(run_loop.train_command(options, chosen)[2:])
    assert launch_options.dataset_revision == "round-5" and launch_options.flavor == run_loop.TRAIN_FLAVOR
    assert launch_options.launch is True
    assert training_arguments == ["--epochs", "2", "--lr", "0.0001"]  # handed to train_lora.py, not parsed here


def test_rerunning_the_gate_step_does_not_commit_an_unchanged_config(tmp_path, monkeypatch):
    configs = tmp_path / "configs" / "systems"
    configs.mkdir(parents=True)
    (configs / "r3-weight.yaml").write_text("name: r3-weight\nadapter:\n  revision: round-3\n")
    calls = []

    def fake_run(command, **keyword_arguments):
        calls.append(command)
        return argparse.Namespace(returncode=0)  # nothing staged: already committed

    monkeypatch.setattr(run_loop, "ROOT", tmp_path)
    monkeypatch.setattr(run_loop.subprocess, "run", fake_run)
    run_loop.write_candidate_config(argparse.Namespace(round=4, from_round=3))
    assert not any("commit" in command for command in calls) and not any("push" in command for command in calls)


def test_the_bfcl_screen_applies_the_gates_own_rule():
    assert run_loop.bfcl_screen_passes(41.0, 43.0, 2.0)  # exactly 2 points below is allowed
    assert not run_loop.bfcl_screen_passes(40.99, 43.0, 2.0)
    assert not run_loop.bfcl_screen_passes(33.67, 43.0, 2.0)  # round 4
    assert run_loop.bfcl_screen_passes(50.0, 43.0, 2.0)  # better than the live system


def test_a_round_with_the_failures_given_does_not_pay_for_the_steps_it_skips():
    options = argparse.Namespace(failures_run="r4-cand-train", screen_bfcl=True)
    assert run_loop.planned_steps(options) == ["data", "train", "bfcl_screen", "gate"]
    assert run_loop.cost_of_a_full_round(None, run_loop.planned_steps(options)) == pytest.approx(2.35)
    plain = argparse.Namespace(failures_run=None, screen_bfcl=False)
    assert run_loop.planned_steps(plain) == ["candidate_train", "classify", "data", "train", "gate"]

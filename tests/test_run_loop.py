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


SOURCE_CONFIG = """name: r3-weight
model: CohereLabs/tiny-aya-global
adapter:
  repo: x/y
  revision: round-3
route: weight_fix
targets: [invented_id, loop]
"""


def write_source_config(tmp_path):
    configs = tmp_path / "configs" / "systems"
    configs.mkdir(parents=True)
    (configs / "r3-weight.yaml").write_text(SOURCE_CONFIG)
    return configs


def fake_git(calls, staged_changes=True):
    def fake_run(command, **keyword_arguments):
        calls.append(command)
        return argparse.Namespace(returncode=1 if staged_changes else 0)  # `git diff --cached --quiet`: 1 = changed

    return fake_run


def test_a_trained_round_points_its_config_at_its_own_adapter(tmp_path, monkeypatch):
    configs = write_source_config(tmp_path)
    calls = []
    monkeypatch.setattr(run_loop, "ROOT", tmp_path)
    monkeypatch.setattr(run_loop.subprocess, "run", fake_git(calls))
    chosen = run_loop.recipe_module.Recipe(tool_guards=False)
    run_loop.write_candidate_config(argparse.Namespace(round=4, from_round=3), chosen, retrain=True)
    written = (configs / "r4-weight.yaml").read_text()
    assert "name: r4-weight" in written and "revision: round-4" in written and "Round 4 candidate" in written
    assert "tool_guards: false" in written
    commit = next(command for command in calls if "commit" in command)
    assert commit[-2:] == ["--", "configs/systems/r4-weight.yaml"]  # only its own file is committed


def test_a_guards_only_round_keeps_the_adapter_and_turns_the_guards_on(tmp_path, monkeypatch):
    configs = write_source_config(tmp_path)
    monkeypatch.setattr(run_loop, "ROOT", tmp_path)
    monkeypatch.setattr(run_loop.subprocess, "run", fake_git([]))
    chosen = run_loop.recipe_module.Recipe(tool_guards=True, reasons=["loops: guards on"])
    run_loop.write_candidate_config(argparse.Namespace(round=6, from_round=3), chosen, retrain=False)
    written = (configs / "r6-weight.yaml").read_text()
    assert "revision: round-3" in written and "tool_guards: true" in written and "the same as the round before" in written


def test_rerunning_the_gate_step_does_not_commit_an_unchanged_config(tmp_path, monkeypatch):
    write_source_config(tmp_path)
    calls = []
    monkeypatch.setattr(run_loop, "ROOT", tmp_path)
    monkeypatch.setattr(run_loop.subprocess, "run", fake_git(calls, staged_changes=False))
    run_loop.write_candidate_config(argparse.Namespace(round=4, from_round=3), run_loop.recipe_module.Recipe(), True)
    assert not any("commit" in command for command in calls) and not any("push" in command for command in calls)


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


class FakeJobs:
    """A fake Hub client: each job id walks through the stages given for it."""

    def __init__(self, stages_by_job):
        self.stages = {job: list(stages) for job, stages in stages_by_job.items()}
        self.cancelled = []

    def inspect_job(self, job_id):
        stage = self.stages[job_id][0] if len(self.stages[job_id]) == 1 else self.stages[job_id].pop(0)
        return argparse.Namespace(status=argparse.Namespace(stage=stage), flavor="l40sx1",
                                  durations=argparse.Namespace(running_secs=1800))

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)


def test_a_job_stuck_in_the_queue_is_cancelled_and_costs_nothing():
    jobs = FakeJobs({"j1": ["SCHEDULING"]})
    now = [0.0]
    stage, cost = run_loop.wait_for_job("j1", queue_limit_seconds=600, hf_api=jobs,
                                        sleep=lambda seconds: now.__setitem__(0, now[0] + 400), clock=lambda: now[0])
    assert (stage, cost) == ("QUEUE_TIMEOUT", 0.0) and jobs.cancelled == ["j1"]


def test_a_job_that_runs_is_priced_by_its_running_seconds():
    jobs = FakeJobs({"j2": ["SCHEDULING", "RUNNING", "COMPLETED"]})
    stage, cost = run_loop.wait_for_job("j2", queue_limit_seconds=600, hf_api=jobs, sleep=lambda seconds: None)
    assert stage == "COMPLETED" and cost == pytest.approx(0.5 * 1.80)  # 1800 s on an l40sx1


def test_the_second_gpu_type_is_tried_when_the_first_waits_too_long():
    launched = []

    def launch(command):
        launched.append(command)
        return f"job-{len(launched)}"

    def wait(job_id, queue_limit_seconds):
        return ("QUEUE_TIMEOUT", 0.0) if job_id == "job-1" else ("COMPLETED", 0.8)

    result = run_loop.run_hf_job(lambda flavor: ["launch", flavor], ["a10g-large", "l40sx1"], 12, launch=launch, wait=wait)
    assert launched == [["launch", "a10g-large"], ["launch", "l40sx1"]]
    assert result["flavor"] == "l40sx1" and result["stage"] == "COMPLETED" and result["cost"] == 0.8
    assert [attempt["stage"] for attempt in result["tried"]] == ["QUEUE_TIMEOUT", "COMPLETED"]


def test_the_last_gpu_type_waits_as_long_as_it_takes_and_a_failure_is_not_retried():
    limits = []

    def wait(job_id, queue_limit_seconds):
        limits.append(queue_limit_seconds)
        return "ERROR", 0.1

    result = run_loop.run_hf_job(lambda flavor: [flavor], ["l40sx1"], 12, launch=lambda command: "j", wait=wait)
    assert limits == [None] and result["stage"] == "ERROR" and len(result["tried"]) == 1


def test_a_harness_only_remedy_runs_only_the_gate():
    recipe = run_loop.recipe_module
    previous = recipe.Recipe(general_replay=100)
    guards_on = recipe.Recipe(general_replay=100, tool_guards=True)
    options = argparse.Namespace(failures_run=None, screen_bfcl=True)
    assert run_loop.planned_steps(options, guards_on, previous) == ["gate"]
    with_corrections = recipe.Recipe(general_replay=100, tool_guards=True, corrections=True)
    steps = run_loop.planned_steps(argparse.Namespace(failures_run="r4-cand-train", screen_bfcl=True), with_corrections, guards_on)
    assert steps == ["corrections", "data", "train", "bfcl_screen", "gate"]


def test_a_real_litellm_reply_can_be_turned_into_the_fallback_text():
    """patches.ask_again_until_policy_ok edits the reply in place: check it on litellm's own classes, not a stand-in."""
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message, ModelResponse

    from shipwright.llm import guards, patches

    message = Message(content=None, tool_calls=[ChatCompletionMessageToolCall(
        id="x", type="function", function=Function(name="get_order_details", arguments='{"order_id": "#W1"}'))])
    response = ModelResponse(choices=[{"index": 0, "finish_reason": "tool_calls", "message": message}], model="m")
    earlier = [
        {"role": "user", "content": "email a@b.c, order #W1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
            "name": "find_user_id_by_email", "arguments": '{"email": "a@b.c"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "u_1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {
            "name": "get_order_details", "arguments": '{"order_id": "#W1"}'}}]},
        {"role": "tool", "tool_call_id": "b", "content": '{"order_id": "#W1", "status": "pending", "items": []}'}]
    patched = patches.ask_again_until_policy_ok(earlier, {"messages": []}, response, lambda request: response)
    final_message = patched.choices[0].message
    assert final_message.tool_calls is None and "already looked that up" in final_message.content
    assert guards.GUARDS  # the rule list is the one the loop's doctor checks


def test_a_real_litellm_reply_with_an_item_id_is_rewritten_in_place():
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message, ModelResponse

    from shipwright.llm import patches

    message = Message(content=None, tool_calls=[ChatCompletionMessageToolCall(
        id="x", type="function", function=Function(name="get_product_details", arguments='{"product_id": "222"}'))])
    response = ModelResponse(choices=[{"index": 0, "finish_reason": "tool_calls", "message": message}], model="m")
    earlier = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {
            "name": "get_product_details", "arguments": '{"product_id": "P2"}'}}]},
        {"role": "tool", "tool_call_id": "b", "content": '{"name": "Shelf", "product_id": "P2", "variants": {"222": {"item_id": "222"}}}'}]
    patched = patches.ask_again_until_policy_ok(earlier, {"messages": []}, response, lambda request: response)
    function = patched.choices[0].message.tool_calls[0].function
    assert function.name == "get_item_details" and json.loads(function.arguments) == {"item_id": "222"}
    trace = patched.choices[0].message.provider_specific_fields[patches.TRACE_FIELD]
    assert trace["guard_repairs"] == [{"from": "get_product_details", "to": "get_item_details", "arguments": {"item_id": "222"}}]

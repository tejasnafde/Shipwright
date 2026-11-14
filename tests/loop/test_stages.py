"""The stages (shipwright.loop.stages) and the command (shipwright.loop.cli): what they launch and write. Fake HF jobs,
no network, no spend."""

import json
from pathlib import Path

import pytest
import yaml

from shipwright.loop import cli, controller, jobs, stages
from shipwright.loop.recipe import Recipe
from shipwright.loop.workspace import Workspace

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "rounds" / "default.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def make_stages(tmp_path, round_number=10, recipe=None, **state_fields) -> stages.Stages:
    state = controller.ChainState(recipe=recipe or Recipe(epochs=2, learning_rate=1e-4), **state_fields)
    workspace = Workspace(config(), runs_directory=tmp_path)
    return stages.Stages(config(), workspace, round_number, state, dry_run=False)


def test_a_real_run_refuses_to_start_on_a_laptop():
    with pytest.raises(SystemExit, match="in the cloud"):
        cli.check_where(here=True, dry_run=False, config=config())


def test_the_train_stage_launches_the_configured_job_with_the_recipe_and_seed(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(jobs, "launch_train", lambda job: launched.append(job) or f"job-{job.flavor}")
    monkeypatch.setattr(jobs, "wait_for_job", lambda job_id, limit=None: ("COMPLETED", 0.9))
    stage = make_stages(tmp_path, adapter_round=10)
    monkeypatch.setattr(stage, "retraining_trigger", lambda: {"should_train": True})
    entry = stage.train()
    assert entry["exit_code"] == 0 and entry["actual_usd"] == 0.9
    job = launched[0]
    assert job.flavor == "l40sx1" and job.timeout == "45m" and job.revision == "round-10"
    assert job.training_arguments[:6] == ["--epochs", "2", "--lr", "0.0001", "--seed", "300"]


def test_a_no_from_agenttunes_trigger_stops_training_without_a_job(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "launch_train", lambda job: pytest.fail("launched"))
    stage = make_stages(tmp_path)
    monkeypatch.setattr(stage, "retraining_trigger", lambda: {"should_train": False, "reason": "buffer"})
    entry = stage.train()
    assert entry["exit_code"] == 1 and entry["no_retry"] and "buffer" in entry["note"]


def test_the_retraining_trigger_reads_the_failures_and_the_sft_data(tmp_path):
    pytest.importorskip("agenttune")
    stage = make_stages(tmp_path, failures_run="r10-baseline")
    failures = tmp_path / "r10-baseline"
    failures.mkdir()
    rows = [{"task_id": str(i), "simulation_id": f"s{i}", "route": "weight_fix", "labels": ["invented_id"],
             "agenttune_root_cause": "hallucinated_output", "outcome_failed": True} for i in range(12)]
    (failures / "failures.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    sft = tmp_path / "r10-sft"
    sft.mkdir()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    (sft / "sft.jsonl").write_text("".join(json.dumps({"task_id": str(i), "messages": messages}) + "\n" for i in range(12)))
    trigger = stage.retraining_trigger()
    assert trigger["should_train"] and trigger["num_with_correction"] == 12
    assert (failures / "round_status.json").exists()


def test_a_trained_round_serves_its_own_adapter_and_the_recipes_guards(tmp_path):
    stage = make_stages(tmp_path, recipe=Recipe(tool_guards=True), adapter_round=10)
    system = yaml.safe_load(stage.write_candidate().read_text())
    assert system["name"] == "r10-weight" and system["adapter"]["revision"] == "round-10" and system["tool_guards"]
    assert stage.state.candidate_system.endswith("r10-round/r10-weight.yaml")


def test_a_guards_only_round_keeps_the_last_candidates_adapter(tmp_path):
    stage = make_stages(tmp_path, round_number=11, recipe=Recipe(tool_guards=True), adapter_round=7,
                        candidate_system="configs/systems/r8-weight.yaml")
    system = yaml.safe_load(stage.write_candidate().read_text())
    r8 = yaml.safe_load((ROOT / "configs" / "systems" / "r8-weight.yaml").read_text())
    assert system["name"] == "r11-weight" and system["adapter"] == r8["adapter"] and system["tool_guards"]


def test_each_round_of_corrections_trains_on_the_earlier_rounds_too(tmp_path):
    for round_number, rows in ((7, ['{"a": 1}', '{"b": 2}']), (8, ['{"b": 2}', '{"c": 3}'])):  # one row is in both
        directory = tmp_path / f"r{round_number}-corrections"
        directory.mkdir()
        (directory / "corrections.jsonl").write_text("\n".join(rows) + "\n")
    combined = make_stages(tmp_path, round_number=8).all_corrections()
    assert len(combined.read_text().splitlines()) == 3


def test_a_missing_pooled_teacher_run_stops_the_data_stage(tmp_path):
    entry = make_stages(tmp_path).data()
    assert entry["exit_code"] == 1 and entry["no_retry"] and "missing" in entry["note"]


def test_the_bfcl_screen_applies_the_gates_own_rule(tmp_path, monkeypatch):
    last = {"metrics_current": {"bfcl": {"accuracy_points": 43.0}}, "thresholds": {"bfcl_max_drop_points": 2.0}}
    stage = make_stages(tmp_path, adapter_round=10, decisions=[last])
    monkeypatch.setattr(stage, "gate_job", lambda *a, **k: jobs.JobResult("COMPLETED", 0.2, "j", "a10g-large"))
    score = tmp_path / "bfcl.json"
    for points, passes in ((40.5, False), (41.0, True)):
        score.write_text(json.dumps({"accuracy_points": points}))
        monkeypatch.setattr(stages, "hub_file", lambda prefix, name: score)
        entry = stage.bfcl_screen()
        assert (entry["exit_code"] == 0) is passes and entry["needed_points"] == 41.0


def test_the_whole_loop_runs_offline_and_walks_a_refusal_then_a_promotion(tmp_path):
    exit_code = cli.run_loop(config(), CONFIG_PATH, 2, dry_run=True, runs_directory=tmp_path)
    chain = json.loads((tmp_path / "dry-chain-r10" / "chain.json").read_text())
    assert exit_code == 0 and chain["status"] == "promoted"
    assert [entry["status"] for entry in chain["rounds"]] == ["rejected", "promoted"]
    ledger = (tmp_path / "dry-promotions" / "ledger.jsonl").read_text()
    assert '"round": 11' in ledger and '"tool_guards": true' in ledger


def test_a_gate_decision_on_the_hub_is_reused_only_for_the_same_candidate(tmp_path):
    candidate = tmp_path / "r10-weight.yaml"
    candidate.write_text(yaml.safe_dump({"name": "r10-weight", "adapter": {"repo": "x/y", "revision": "round-10"},
                                         "tool_guards": True}))
    same = {"systems": {"candidate": {"adapter": {"repo": "x/y", "revision": "round-10"}, "tool_guards": True,
                                      "state_reminders": False}}}
    other = {"systems": {"candidate": {"adapter": {"repo": "x/y", "revision": "round-9"}, "tool_guards": True}}}
    assert stages.same_candidate(same, candidate) and not stages.same_candidate(other, candidate)


def test_a_resumed_chain_learns_the_last_candidate_from_the_manifest():
    saved = {"stages": {"gate": {"ok": True, "candidate": "runs/r10-round/r10-weight.yaml",
                                 "decision": {"decision": "reject", "checks": [{"name": "pass_k", "passed": False}],
                                              "metrics_current": {"label_rates": {}},
                                              "metrics_candidate": {"label_rates": {}}}}}}
    state = controller.ChainState(recipe=Recipe())
    hooks = controller.Hooks(stage_functions=lambda n, s: {}, save_manifest=lambda n, m: None,
                             load_manifest=lambda n: saved, promote=lambda n, d: None,
                             write_next_round=lambda n, d, r: None, log=lambda m: None)
    controller.run_round(10, ["gate"], hooks, state, config(), 0.0, {})
    assert state.candidate_system == "runs/r10-round/r10-weight.yaml"

"""scripts/run_round_all.py: what the real stages launch, with fake HF jobs. No network, no spend."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
specification = importlib.util.spec_from_file_location("run_round_all", ROOT / "scripts" / "run_round_all.py")
run_round_all = importlib.util.module_from_spec(specification)
sys.modules["run_round_all"] = run_round_all
specification.loader.exec_module(run_round_all)

from shipwright.heal import recipe as recipe_module  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "rounds" / "default.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def state(tool_guards=False) -> dict:
    return {"recipe": recipe_module.Recipe(epochs=2, learning_rate=1e-4, tool_guards=tool_guards),
            "baseline_run": "r10-baseline", "teacher_run": "r10-teacher", "adapter_round": 10}


def test_a_real_chain_refuses_to_run_on_a_laptop():
    with pytest.raises(SystemExit, match="runs in the cloud"):
        run_round_all.run_here(config(), CONFIG_PATH, 1, dry_run=False)


def test_the_train_stage_launches_the_configured_job_with_the_recipe_and_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round_all, "RUNS", tmp_path)
    launched = {}

    def fake_run_hf_job(make_command, flavors):
        launched["commands"] = [make_command(flavor) for flavor in flavors]
        return {"stage": "COMPLETED", "cost": 0.9, "job": "j1", "flavor": flavors[0], "tried": []}

    run_loop = run_round_all.load_script_module("run_loop")
    monkeypatch.setattr(run_loop, "run_hf_job", fake_run_hf_job)
    stages = run_round_all.Stages(config(), 10, state(), dry_run=False)
    monkeypatch.setattr(stages, "retraining_trigger", lambda: {"should_train": True})
    entry = stages.train()
    assert entry["exit_code"] == 0 and entry["actual_usd"] == 0.9
    command = launched["commands"][0]
    assert command[command.index("--flavor") + 1] == "l40sx1" and "--launch" in command
    assert command[command.index("--timeout") + 1] == "45m"
    assert command[command.index("--revision") + 1] == "round-10"
    after = command[command.index("--") + 1:]
    assert after[:6] == ["--epochs", "2", "--lr", "0.0001", "--seed", "300"]
    assert launched["commands"][1][launched["commands"][1].index("--flavor") + 1] == "a100-large"


def test_a_no_from_the_trigger_launches_nothing(monkeypatch):
    stages = run_round_all.Stages(config(), 10, state(), dry_run=False)
    monkeypatch.setattr(stages, "retraining_trigger", lambda: {"should_train": False, "reason": "buffer small"})
    entry = stages.train()
    assert entry["exit_code"] == 1 and entry["no_retry"] and "buffer small" in entry["note"]


def test_a_dry_gate_writes_the_candidate_and_the_round_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round_all, "RUNS", tmp_path)
    dry = run_round_all.dry_run_config(config())
    entry = run_round_all.Stages(dry, 10, state(tool_guards=True), dry_run=True).gate()
    candidate = yaml.safe_load((tmp_path / "dry-r10-round" / "r10-weight.yaml").read_text())
    assert candidate["adapter"] == {"repo": dry["train"]["output_repo"], "revision": "round-10"}
    assert candidate["tool_guards"] is True
    assert entry["decision"]["decision"] == "promote"  # the dry gate approves once the guards are on
    assert json.loads((tmp_path / "dry-r10-round" / "gate_decision.json").read_text())["dry_run"]
    assert not (ROOT / "configs" / "systems" / "r10-weight.yaml").exists()  # a dry run leaves configs/ alone


def test_a_failed_gate_job_is_a_failed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(run_round_all, "RUNS", tmp_path)
    monkeypatch.setattr(run_round_all, "write_candidate_system", lambda *a: ROOT / "configs" / "systems" / "base.yaml")
    run_loop = run_round_all.load_script_module("run_loop")
    monkeypatch.setattr(run_loop, "run_hf_job", lambda make_command, flavors: {
        "stage": "ERROR", "cost": 0.4, "job": "j2", "flavor": flavors[0], "tried": []})
    entry = run_round_all.Stages(config(), 10, state(), dry_run=False).gate()
    assert entry["exit_code"] == 1 and "ERROR" in entry["note"] and entry["actual_usd"] == pytest.approx(1.0)


def test_round_10_config_starts_from_the_learned_recipe_and_turns_reminders_on(tmp_path, monkeypatch):
    from shipwright.heal import round_chain

    round_10 = round_chain.load_config(ROOT / "configs" / "rounds" / "r10-fixes.yaml")
    assert round_10["owner"] == "mitushi" and round_10["recipe"]["tool_guards"] is True
    monkeypatch.setattr(run_round_all, "RUNS", tmp_path)
    dry = run_round_all.dry_run_config(round_10)
    run_round_all.Stages(dry, 10, state(tool_guards=True), dry_run=True).gate()
    candidate = yaml.safe_load((tmp_path / "dry-r10-round" / "r10-weight.yaml").read_text())
    assert candidate["state_reminders"] is True and candidate["tool_guards"] is True
    bad = dict(dry, gate={**dry["gate"], "candidate_extra": {"adapter": None}})
    with pytest.raises(SystemExit):
        run_round_all.write_candidate_system(bad, 10, state(tool_guards=True))

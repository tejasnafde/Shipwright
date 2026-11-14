"""The loop's decisions (shipwright.loop.controller and config): stage order per remedy, stop rules, one retry, spend
caps, resume, later rounds up to and including correction rounds. Fakes only: no scripts, no network, no spend."""

from pathlib import Path

import pytest
import yaml

from shipwright.heal import promotion
from shipwright.loop import config as config_module
from shipwright.loop import controller

ROOT = Path(__file__).resolve().parents[2]


def config(**changes) -> dict:
    loaded = yaml.safe_load((ROOT / "configs" / "rounds" / "default.yaml").read_text())
    loaded.update(changes)
    return loaded


def roomy_config() -> dict:
    """The default config with a total cap that several rounds fit in."""
    loaded = config()
    loaded["spend"]["total"] = 50.0
    return loaded


def decision(approve: bool, failed=("pass_k",), pass_1: float = 0.1) -> dict:
    checks = [{"name": name, "passed": approve} for name in failed]
    return {"decision": "promote" if approve else "reject", "checks": checks,
            "metrics_current": {"label_rates": {}}, "metrics_candidate": {"label_rates": {}, "pass_1": pass_1}}


class Fakes:
    """Stage functions that return scripted entries, and records of everything the chain asked for."""

    def __init__(self, scripted: dict | None = None, gate_decisions=None):
        self.scripted = scripted or {}          # stage -> list of entries, one per call (the last one repeats)
        self.gate_decisions = list(gate_decisions or [decision(False)])
        self.calls: list[tuple[int, str]] = []
        self.manifests: dict[int, dict] = {}
        self.promoted: list[int] = []
        self.next_rounds: list[tuple[int, object]] = []
        self.states: list[dict] = []

    def entry(self, round_number: int, stage: str) -> dict:
        self.calls.append((round_number, stage))
        if stage in self.scripted:
            queue = self.scripted[stage]
            result = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(result, BaseException):
                raise result
            return dict(result)
        if stage == "gate":
            return {"exit_code": 0, "actual_usd": 1.0, "decision": self.gate_decisions.pop(0)}
        if stage in ("data", "bfcl_screen", "corrections"):
            return {"exit_code": 0, "actual_usd": 0.0, "examples": 100}
        return {"exit_code": 0, "actual_usd": 0.5, "num_runs": 10, "num_infra_errors": 0}

    def hooks(self, saved_manifests: dict | None = None) -> controller.Hooks:
        def stage_functions(round_number, state):
            self.states.append({"recipe": state.recipe, "adapter_round": state.adapter_round,
                                "failures_run": state.failures_run})
            return {stage: (lambda stage=stage: self.entry(round_number, stage)) for stage in config_module.STAGES}

        def save(round_number, manifest):
            self.manifests[round_number] = manifest

        return controller.Hooks(
            stage_functions=stage_functions, save_manifest=save,
            load_manifest=lambda round_number: (saved_manifests or {}).get(round_number),
            promote=lambda round_number, gate_decision: self.promoted.append(round_number),
            write_next_round=lambda n, d, r: self.next_rounds.append((n, r)), log=lambda message: None)


def run(fakes: Fakes, rounds: int = 1, chain_config: dict | None = None, **start) -> controller.ChainResult:
    return controller.run_chain(chain_config or config(), rounds, fakes.hooks(), {"resolved": True}, **start)


def stages_of(fakes: Fakes, round_number: int) -> list[str]:
    return [stage for number, stage in fakes.calls if number == round_number]


def test_the_shipped_round_configs_are_valid():
    for path in sorted((ROOT / "configs" / "rounds").glob("*.yaml")):
        assert config_module.validate(yaml.safe_load(path.read_text())) == [], path.name


def test_a_fresh_round_runs_every_stage_in_order_and_a_refusal_writes_the_next_round_input():
    fakes = Fakes()
    result = run(fakes)
    assert stages_of(fakes, 10) == controller.FRESH_STAGES  # default.yaml screens BFCL
    assert result.status == controller.REJECTED and "pass_k" in result.reason
    assert fakes.next_rounds and fakes.next_rounds[0][1].tool_guards  # the recipe the refusal leads to
    manifest = fakes.manifests[10]
    assert manifest["status"] == "rejected" and manifest["config"] == {"resolved": True}
    assert result.spent_usd == pytest.approx(0.5 * 4 + 1.0)


def test_without_the_screen_the_bfcl_stage_is_left_out():
    chain_config = config()
    chain_config["gate"]["screen_bfcl"] = False
    fakes = Fakes()
    run(fakes, chain_config=chain_config)
    assert "bfcl_screen" not in stages_of(fakes, 10)


def test_an_approval_is_promoted():
    fakes = Fakes(gate_decisions=[decision(True)])
    assert run(fakes).status == controller.PROMOTED
    assert fakes.promoted == [10]


def test_a_failed_promotion_stops_the_chain_and_is_recorded():
    fakes = Fakes(gate_decisions=[decision(True)])
    hooks = fakes.hooks()
    hooks.promote = lambda n, d: (_ for _ in ()).throw(ValueError("no adapter"))
    result = controller.run_chain(config(), 1, hooks, {})
    assert result.status == controller.STOPPED and "promotion failed" in result.reason


def test_a_failed_stage_is_tried_once_more_and_both_attempts_are_paid_for():
    fakes = Fakes({"baseline": [{"exit_code": 1, "actual_usd": 0.3, "note": "crash"},
                                {"exit_code": 0, "actual_usd": 0.4, "num_runs": 10, "num_infra_errors": 0}]})
    result = run(fakes)
    assert stages_of(fakes, 10).count("baseline") == 2
    entry = fakes.manifests[10]["stages"]["baseline"]
    assert entry["ok"] and entry["actual_usd"] == pytest.approx(0.7) and len(entry["attempts"]) == 2
    assert result.status == controller.REJECTED


def test_a_stage_that_fails_twice_stops_the_round():
    fakes = Fakes({"teacher": [{"exit_code": 2, "actual_usd": 0.1, "note": "Vertex 500"}]})
    result = run(fakes)
    assert result.status == controller.STOPPED and result.reason == "teacher: Vertex 500"
    assert stages_of(fakes, 10) == ["baseline", "classify", "teacher", "teacher"]


def test_a_crashing_stage_is_a_failed_attempt_not_a_dead_chain():
    fakes = Fakes({"gate": [RuntimeError("hub 503"), SystemExit("launch failed")]})
    result = run(fakes)
    assert result.status == controller.STOPPED and "SystemExit: launch failed" in result.reason


def test_too_many_infra_errors_are_retried_then_stop_the_round():
    fakes = Fakes({"baseline": [{"exit_code": 0, "actual_usd": 0.5, "num_runs": 74, "num_infra_errors": 20}]})
    result = run(fakes)
    assert result.status == controller.STOPPED and "20 of 74 runs were infra errors" in result.reason
    assert stages_of(fakes, 10) == ["baseline", "baseline"]


def test_too_few_sft_examples_stop_without_a_retry():
    fakes = Fakes({"data": [{"exit_code": 0, "actual_usd": 0.0, "examples": 12}]})
    result = run(fakes)
    assert result.status == controller.STOPPED and "12 examples" in result.reason
    assert stages_of(fakes, 10).count("data") == 1


def test_a_no_from_the_retraining_trigger_is_not_retried():
    fakes = Fakes({"train": [{"exit_code": 1, "actual_usd": 0.0, "no_retry": True, "note": "AgentTune says no"}]})
    result = run(fakes)
    assert result.status == controller.STOPPED and stages_of(fakes, 10).count("train") == 1


def test_a_stage_whose_cap_does_not_fit_is_not_started():
    chain_config = config()
    chain_config["spend"]["total"] = 4.5  # 2.0 spent before the gate: its cap 3.0 does not fit
    fakes = Fakes()
    result = run(fakes, chain_config=chain_config)
    assert result.status == controller.OVER_BUDGET and result.reason.startswith("gate:")
    assert "gate" not in stages_of(fakes, 10)


def test_a_retry_is_not_started_when_its_cap_no_longer_fits():
    chain_config = config()
    chain_config["spend"]["total"] = 2.5
    fakes = Fakes({"baseline": [{"exit_code": 1, "actual_usd": 0.6, "note": "crash"}]})
    result = run(fakes, chain_config=chain_config)
    assert result.status == controller.OVER_BUDGET
    assert fakes.manifests[10]["stages"]["baseline"]["actual_usd"] == pytest.approx(0.6)


def test_a_rerun_resumes_after_the_finished_stages():
    saved = {10: {"round": 10, "stages": {
        "baseline": {"ok": True, "actual_usd": 1.5}, "classify": {"ok": True, "actual_usd": 0.05},
        "teacher": {"ok": False, "actual_usd": 0.2}}}}
    fakes = Fakes()
    result = controller.run_chain(config(), 1, fakes.hooks(saved), {})
    assert stages_of(fakes, 10) == ["teacher", "data", "train", "bfcl_screen", "gate"]
    assert result.spent_usd == pytest.approx(1.5 + 0.05 + 0.5 + 0.5 + 1.0)  # the failed earlier teacher try is replaced


def test_a_refusal_turns_the_guards_on_and_gates_the_same_adapter():
    fakes = Fakes(gate_decisions=[decision(False), decision(True)])
    result = run(fakes, rounds=3)
    assert result.status == controller.PROMOTED and [entry.round_number for entry in result.rounds] == [10, 11]
    assert stages_of(fakes, 11) == ["gate"]
    assert fakes.states[1]["recipe"].tool_guards and fakes.states[1]["adapter_round"] == 10
    assert fakes.manifests[11]["failures_run"] == "r10-baseline"


def test_a_bfcl_refusal_retrains_from_the_first_rounds_data():
    fakes = Fakes(gate_decisions=[decision(False, failed=("bfcl",)), decision(True)])
    run(fakes, rounds=2)
    assert stages_of(fakes, 11) == controller.RETRAIN_STAGES
    assert fakes.states[1]["adapter_round"] == 11 and fakes.states[1]["recipe"].general_replay == 100


def test_guards_then_corrections_run_in_the_same_chain_on_the_candidates_own_failures():
    # Before 2026-11-14 the cloud chain stopped here ("handoff") and a laptop script did the corrections.
    fakes = Fakes(gate_decisions=[decision(False), decision(False), decision(True)])
    result = run(fakes, rounds=3, chain_config=roomy_config())
    assert result.status == controller.PROMOTED
    assert stages_of(fakes, 12) == controller.CORRECTION_STAGES
    round_12 = fakes.states[2]
    assert round_12["recipe"].corrections_rounds == 1 and round_12["failures_run"] == "r12-cand-train"
    assert round_12["adapter_round"] == 12


def test_corrections_stop_when_a_round_did_not_improve_the_gate():
    gates = [decision(False), decision(False, pass_1=0.2), decision(False, pass_1=0.2), decision(True)]
    fakes = Fakes(gate_decisions=gates)
    result = run(fakes, rounds=4, chain_config=roomy_config())
    assert result.status == controller.EXHAUSTED and len(result.rounds) == 3
    assert "did not improve" in result.reason


def test_a_chain_can_continue_from_a_refused_gate():
    chain_config = config(start={"from_round": 8, "candidate": "configs/systems/r8-weight.yaml"},
                          recipe={"tool_guards": True, "general_replay": 100})
    fakes = Fakes(gate_decisions=[decision(True)])
    result = run(fakes, chain_config=chain_config, start_decision=decision(False), start_adapter_round=7)
    assert result.status == controller.PROMOTED
    assert stages_of(fakes, 10) == controller.CORRECTION_STAGES  # guards were already on: corrections next


def test_the_total_cap_holds_across_rounds():
    chain_config = config()
    chain_config["spend"]["total"] = 6.0
    chain_config["spend"]["gate"] = 1.5
    fakes = Fakes(gate_decisions=[decision(False), decision(True)])
    result = run(fakes, rounds=2, chain_config=chain_config)
    assert result.status == controller.PROMOTED and result.spent_usd <= 6.0
    chain_config["spend"]["total"] = 4.0
    fakes = Fakes(gate_decisions=[decision(False), decision(True)])
    result = run(fakes, rounds=2, chain_config=chain_config)
    assert result.status == controller.OVER_BUDGET and result.rounds[-1].round_number == 11


@pytest.mark.parametrize("change, problem", [
    (lambda c: c["train"].update(flavors=["a100-large"], timeout="1h"), "train: worst case $2.50"),
    (lambda c: c["gate"].update(timeout="40m"), "gate.timeout: under 60m"),
    (lambda c: c["train"].update(flavors=["h100"]), "no price for ['h100']"),
    (lambda c: c["spend"].pop("teacher"), "spend.teacher: missing"),
    (lambda c: c["spend"].pop("corrections"), "spend.corrections: missing"),
    (lambda c: c.pop("seed"), "seed: missing"),
    (lambda c: c["gate"].update(candidate_extra={"temperature": 1}), "candidate_extra: temperature"),
    (lambda c: c.update(start={"from_round": 8}), "start: needs"),
])
def test_a_config_that_cannot_work_is_refused(change, problem):
    chain_config = config()
    change(chain_config)
    assert any(problem in found for found in config_module.validate(chain_config))


def test_a_promotion_keeps_the_serving_switches_of_the_gated_candidate(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    candidate = {"name": "r11-weight", "adapter": {"repo": "x/y", "revision": "round-10"}, "tool_guards": True,
                 "state_reminders": True}
    promotion.promote(ledger, {"decision": "promote", "round": 11, "systems": {"candidate": candidate}}, "d.json")
    deployed = promotion.deployed_system(promotion.read_ledger(ledger))
    assert deployed["tool_guards"] is True and deployed["state_reminders"] is True


def test_a_regate_start_measures_the_candidate_again_then_picks_the_remedy_from_that():
    chain_config = roomy_config()
    chain_config.update(start={"from_round": 11, "candidate": "configs/systems/r11-r7-fixes.yaml", "regate": True},
                        recipe={"tool_guards": True, "general_replay": 100, "corrections_rounds": 1})
    fakes = Fakes(gate_decisions=[decision(False, failed=("non_target",)), decision(True)])
    result = run(fakes, rounds=2, chain_config=chain_config, start_adapter_round=7)
    assert result.status == controller.PROMOTED
    assert stages_of(fakes, 10) == ["gate"] and fakes.states[0]["adapter_round"] == 7
    assert stages_of(fakes, 11) == controller.CORRECTION_STAGES
    assert fakes.states[1]["recipe"].corrections_rounds == 2 and fakes.states[1]["adapter_round"] == 11


def test_the_r12_config_reaches_its_last_gate_inside_the_cap_at_realistic_costs():
    """Round 12 gate $1.42 (gate 10's real cost), then a corrections round at past real costs: the final gate must still
    be allowed to start, or the money would be spent on a round that never gets its verdict."""
    chain_config = yaml.safe_load((ROOT / "configs" / "rounds" / "r12-final.yaml").read_text())
    real = {"candidate_train": {"exit_code": 0, "actual_usd": 0.8, "num_runs": 50, "num_infra_errors": 0},
            "classify": {"exit_code": 0, "actual_usd": 0.05}, "corrections": {"exit_code": 0, "actual_usd": 0.3},
            "data": {"exit_code": 0, "actual_usd": 0.0, "examples": 600},
            "train": {"exit_code": 0, "actual_usd": 1.25}}
    gates = [decision(False, failed=("non_target",)), decision(True)]
    fakes = Fakes({stage: [entry] for stage, entry in real.items()}, gate_decisions=gates)
    original = fakes.entry

    def entry(round_number, stage):
        result = original(round_number, stage)
        if stage == "gate":
            result["actual_usd"] = 1.42
        return result

    fakes.entry = entry
    result = run(fakes, rounds=2, chain_config=chain_config, start_adapter_round=7)
    assert result.status == controller.PROMOTED and stages_of(fakes, 13)[-1] == "gate"
    assert result.spent_usd <= chain_config["spend"]["total"]

"""The one-command round (round_chain): stage order, stop rules, one retry, spend caps, later rounds. Fakes only."""

from pathlib import Path

import pytest
import yaml

from shipwright.heal import promotion, round_chain

ROOT = Path(__file__).resolve().parents[2]


def config(**changes) -> dict:
    loaded = yaml.safe_load((ROOT / "configs" / "rounds" / "default.yaml").read_text())
    for key, value in changes.items():
        loaded[key] = value
    return loaded


def decision(approve: bool, failed=("pass_k",)) -> dict:
    checks = [{"name": name, "passed": approve} for name in failed]
    return {"decision": "promote" if approve else "reject", "checks": checks,
            "metrics_current": {"label_rates": {}}, "metrics_candidate": {"label_rates": {}}}


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
        if stage == "data":
            return {"exit_code": 0, "actual_usd": 0.0, "examples": 100}
        return {"exit_code": 0, "actual_usd": 0.5, "num_runs": 10, "num_infra_errors": 0}

    def hooks(self, saved_manifests: dict | None = None) -> round_chain.ChainHooks:
        def stage_functions(round_number, state):
            self.states.append({**state, "recipe": state["recipe"]})
            return {stage: (lambda stage=stage: self.entry(round_number, stage)) for stage in round_chain.STAGES}

        def save(round_number, manifest):
            self.manifests[round_number] = manifest

        def promote(round_number, gate_decision):
            self.promoted.append(round_number)

        return round_chain.ChainHooks(
            stage_functions=stage_functions, save_manifest=save,
            load_manifest=lambda round_number: (saved_manifests or {}).get(round_number),
            promote=promote, write_next_round=lambda n, d, r: self.next_rounds.append((n, r)), log=lambda message: None)


def run(fakes: Fakes, rounds: int = 1, **changes) -> round_chain.ChainResult:
    chain_config = config(**changes)
    return round_chain.run_chain(chain_config, rounds, fakes.hooks(), {"resolved": True})


def test_the_default_config_is_valid():
    assert round_chain.validate_config(config()) == []


def test_a_round_runs_every_stage_in_order_and_a_refusal_writes_the_next_round_input():
    fakes = Fakes()
    result = run(fakes)
    assert [stage for _, stage in fakes.calls] == round_chain.STAGES
    assert result.status == round_chain.REJECTED and "pass_k" in result.reason
    assert fakes.next_rounds and fakes.next_rounds[0][1].tool_guards  # the recipe the refusal leads to
    manifest = fakes.manifests[10]
    assert manifest["status"] == "rejected" and manifest["config"] == {"resolved": True}
    assert result.spent_usd == pytest.approx(0.5 * 4 + 1.0)


def test_an_approval_is_promoted():
    fakes = Fakes(gate_decisions=[decision(True)])
    assert run(fakes).status == round_chain.PROMOTED
    assert fakes.promoted == [10]


def test_a_failed_promotion_stops_the_chain_and_is_recorded():
    fakes = Fakes(gate_decisions=[decision(True)])
    hooks = fakes.hooks()
    hooks.promote = lambda n, d: (_ for _ in ()).throw(ValueError("no adapter"))
    result = round_chain.run_chain(config(), 1, hooks, {})
    assert result.status == round_chain.STOPPED and "promotion failed" in result.reason


def test_a_failed_stage_is_tried_once_more_and_both_attempts_are_paid_for():
    fakes = Fakes({"baseline": [{"exit_code": 1, "actual_usd": 0.3, "note": "crash"},
                                {"exit_code": 0, "actual_usd": 0.4, "num_runs": 10, "num_infra_errors": 0}]})
    result = run(fakes)
    assert [stage for _, stage in fakes.calls].count("baseline") == 2
    entry = fakes.manifests[10]["stages"]["baseline"]
    assert entry["ok"] and entry["actual_usd"] == pytest.approx(0.7) and len(entry["attempts"]) == 2
    assert result.status == round_chain.REJECTED


def test_a_stage_that_fails_twice_stops_the_round():
    fakes = Fakes({"teacher": [{"exit_code": 2, "actual_usd": 0.1, "note": "Vertex 500"}]})
    result = run(fakes)
    assert result.status == round_chain.STOPPED and result.reason == "teacher: Vertex 500"
    assert [stage for _, stage in fakes.calls] == ["baseline", "classify", "teacher", "teacher"]


def test_a_crashing_stage_is_a_failed_attempt_not_a_dead_chain():
    fakes = Fakes({"gate": [RuntimeError("hub 503"), SystemExit("launch failed")]})
    result = run(fakes)
    assert result.status == round_chain.STOPPED and "SystemExit: launch failed" in result.reason


def test_too_many_infra_errors_are_retried_then_stop_the_round():
    fakes = Fakes({"baseline": [{"exit_code": 0, "actual_usd": 0.5, "num_runs": 74, "num_infra_errors": 20}]})
    result = run(fakes)
    assert result.status == round_chain.STOPPED and "20 of 74 runs were infra errors" in result.reason
    assert [stage for _, stage in fakes.calls] == ["baseline", "baseline"]


def test_too_few_sft_examples_stop_without_a_retry():
    fakes = Fakes({"data": [{"exit_code": 0, "actual_usd": 0.0, "examples": 12}]})
    result = run(fakes)
    assert result.status == round_chain.STOPPED and "12 examples" in result.reason
    assert [stage for _, stage in fakes.calls].count("data") == 1


def test_a_no_from_the_retraining_trigger_is_not_retried():
    fakes = Fakes({"train": [{"exit_code": 1, "actual_usd": 0.0, "no_retry": True, "note": "AgentTune says no"}]})
    result = run(fakes)
    assert result.status == round_chain.STOPPED and [s for _, s in fakes.calls].count("train") == 1


def test_a_stage_whose_cap_does_not_fit_is_not_started():
    chain_config = config()
    chain_config["spend"]["total"] = 4.5  # 2.0 spent before the gate: its cap 3.0 does not fit, train's 2.0 did
    fakes = Fakes()
    result = round_chain.run_chain(chain_config, 1, fakes.hooks(), {})
    assert result.status == round_chain.OVER_BUDGET and result.reason.startswith("gate:")
    assert "gate" not in [stage for _, stage in fakes.calls]


def test_a_retry_is_not_started_when_its_cap_no_longer_fits():
    chain_config = config()
    chain_config["spend"]["total"] = 2.5
    fakes = Fakes({"baseline": [{"exit_code": 1, "actual_usd": 0.6, "note": "crash"}]})
    result = round_chain.run_chain(chain_config, 1, fakes.hooks(), {})
    assert result.status == round_chain.OVER_BUDGET
    assert fakes.manifests[10]["stages"]["baseline"]["actual_usd"] == pytest.approx(0.6)


def test_a_rerun_resumes_after_the_finished_stages():
    saved = {10: {"round": 10, "stages": {
        "baseline": {"ok": True, "actual_usd": 1.5}, "classify": {"ok": True, "actual_usd": 0.05},
        "teacher": {"ok": False, "actual_usd": 0.2}}}}
    fakes = Fakes()
    result = round_chain.run_chain(config(), 1, fakes.hooks(saved), {})
    assert [stage for _, stage in fakes.calls] == ["teacher", "data", "train", "gate"]
    assert result.spent_usd == pytest.approx(1.5 + 0.05 + 0.5 + 0.5 + 1.0)  # the failed earlier teacher try is replaced


def test_later_rounds_follow_the_recipe_and_stop_at_the_first_promotion():
    # Round 10 refused on pass_k: round 11 turns the guards on and gates the same adapter (no training); approved.
    fakes = Fakes(gate_decisions=[decision(False), decision(True)])
    result = run(fakes, rounds=3)
    assert result.status == round_chain.PROMOTED and [r.round_number for r in result.rounds] == [10, 11]
    assert [call for call in fakes.calls if call[0] == 11] == [(11, "gate")]
    round_11_state = fakes.states[1]
    assert round_11_state["recipe"].tool_guards and round_11_state["adapter_round"] == 10
    assert fakes.manifests[11]["failures_run"] == "r10-baseline"


def test_a_bfcl_refusal_retrains_from_the_first_rounds_data():
    fakes = Fakes(gate_decisions=[decision(False, failed=("bfcl",)), decision(True)])
    run(fakes, rounds=2)
    assert [stage for n, stage in fakes.calls if n == 11] == round_chain.RETRAIN_STAGES
    assert fakes.states[1]["adapter_round"] == 11 and fakes.states[1]["recipe"].general_replay == 100


def test_the_chain_hands_off_when_the_recipe_needs_corrections():
    # pass_k refused twice: guards (round 11), then corrections, which need a candidate-on-train run (run_loop.py).
    fakes = Fakes(gate_decisions=[decision(False), decision(False)])
    result = run(fakes, rounds=3)
    assert result.status == round_chain.HANDOFF and len(result.rounds) == 2


def test_the_total_cap_holds_across_rounds():
    chain_config = config()
    chain_config["spend"]["total"] = 6.0
    fakes = Fakes(gate_decisions=[decision(False), decision(True)])
    chain_config["spend"]["gate"] = 1.5
    result = round_chain.run_chain(chain_config, 2, fakes.hooks(), {})
    # Round 10 spends 3.0; round 11's gate (cap 1.5) fits in what is left; spend over both rounds stays under the cap.
    assert result.status == round_chain.PROMOTED and result.spent_usd <= 6.0
    chain_config["spend"]["total"] = 4.0
    fakes = Fakes(gate_decisions=[decision(False), decision(True)])
    result = round_chain.run_chain(chain_config, 2, fakes.hooks(), {})
    assert result.status == round_chain.OVER_BUDGET and result.rounds[-1].round_number == 11


@pytest.mark.parametrize("change, problem", [
    (lambda c: c["train"].update(flavors=["a100-large"], timeout="1h"), "train: worst case $2.50"),
    (lambda c: c["gate"].update(timeout="40m"), "gate.timeout: under 60m"),
    (lambda c: c["train"].update(flavors=["h100"]), "no price for h100"),
    (lambda c: c["spend"].pop("teacher"), "spend.teacher: missing"),
    (lambda c: c.pop("seed"), "seed: missing"),
])
def test_a_config_that_cannot_work_is_refused(change, problem):
    chain_config = config()
    change(chain_config)
    assert any(problem in found for found in round_chain.validate_config(chain_config))


def test_a_promotion_keeps_the_tool_guards_of_the_gated_candidate(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    candidate = {"name": "r11-weight", "adapter": {"repo": "x/y", "revision": "round-10"}, "tool_guards": True}
    promotion.promote(ledger, {"decision": "promote", "round": 11, "systems": {"candidate": candidate}}, "d.json")
    assert promotion.deployed_system(promotion.read_ledger(ledger))["tool_guards"] is True

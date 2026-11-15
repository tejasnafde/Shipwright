import pytest

from shipwright.heal import promotion

DECISION = {"decision": "promote", "round": 2, "systems": {"candidate": {
    "name": "r2-weight", "adapter": {"repo": "me/lora", "revision": "round-2"}}}}


def test_nothing_deployed_means_the_base_model(tmp_path):
    system = promotion.deployed_system(promotion.read_ledger(tmp_path / "ledger.jsonl"))
    assert system["name"] == "base" and "adapter" not in system


def test_promote_then_rollback_restores_the_previous_system(tmp_path):
    ledger = tmp_path / "promotions" / "ledger.jsonl"
    promotion.promote(ledger, DECISION, "gate_decision.json")
    second = {**DECISION, "round": 3, "systems": {"candidate": {
        "name": "r3-weight", "adapter": {"repo": "me/lora", "revision": "round-3"}}}}
    promotion.promote(ledger, second, "gate_decision.json")
    assert promotion.deployed_system(promotion.read_ledger(ledger))["adapter"]["revision"] == "round-3"
    promotion.rollback(ledger)
    assert promotion.deployed_system(promotion.read_ledger(ledger))["adapter"]["revision"] == "round-2"
    promotion.rollback(ledger)
    assert promotion.deployed_system(promotion.read_ledger(ledger))["name"] == "base"
    assert len(promotion.read_ledger(ledger)) == 4  # rows are never deleted
    with pytest.raises(ValueError, match="nothing to roll back"):
        promotion.rollback(ledger)


def test_only_an_approved_adapter_is_promoted(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="not 'promote'"):
        promotion.promote(ledger, {**DECISION, "decision": "reject"}, "x")
    with pytest.raises(ValueError, match="no adapter"):
        promotion.promote(ledger, {"decision": "promote", "round": 1, "systems": {"candidate": {"name": "t"}}}, "x")
    assert not ledger.exists()

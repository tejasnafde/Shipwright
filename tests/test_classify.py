import json
from pathlib import Path

import pytest

from shipwright import classify as c

FIX = Path(__file__).parent / "fixtures"
EXAMPLES = {"#W0000000", "sara_doe_496", "1008292230", "6086499569", "gift_card_0000000"}


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def labels(trace):
    return {h["label"] for h in c.rule_labels(trace, EXAMPLES)}


def test_passed_run_is_not_a_failure():
    assert c.classify_run(load("passed_run")) is None


def test_wrong_tool_on_delivered_order():
    t = load("wrong_tool_delivered_order")
    got = labels(t)
    assert "tool_error_ignored" in got  # modify failed, then cancel with no lookup
    assert "invented_id" in got and "tool_example_copied" in got  # item id 1008292230 from the tool example
    row = c.classify_run(t, example_values=EXAMPLES)
    assert row["route"] == "tool_fix" and row["source"] == "rules"
    assert row["first_wrong_turn"] == 2


def test_invented_user_id_and_loop():
    t = load("invented_user_id_and_loop")
    hits = c.rule_labels(t, EXAMPLES)
    assert any(h["label"] == "invented_id" and "sara_doe_496" in h["detail"] for h in hits)
    assert any(h["label"] == "loop" and "modify_pending_order_items" in h["detail"] for h in hits)
    # yusuf_rossi_9620 came from a tool result, so it is not invented
    assert not any("yusuf_rossi_9620" in h["detail"] for h in hits)


def test_repeated_lookups_are_a_loop():
    t = load("premature_transfer")
    assert any(h["label"] == "loop" for h in c.rule_labels(t))


def test_without_examples_invented_id_routes_to_weights():
    row = c.classify_run(load("invented_user_id_and_loop"))
    assert "tool_example_copied" not in row["labels"]
    assert row["route"] == "weight_fix"


def _synthetic(calls, extra_turn=None):
    """A failed trace from (name, args, error) tuples."""
    messages, turns = [{"role": "user", "content": "my order is #W1111111"}], []
    for i, (name, args, err) in enumerate(calls):
        idx = len(messages)
        cid = f"c{i}"
        messages.append({"role": "assistant", "tool_calls": [{"id": cid, "name": name, "arguments": args}]})
        messages.append({"role": "tool", "id": cid, "content": "Error" if err else "{}", "error": err})
        turns.append({"message_index": idx, "parsed_tool_calls": [
            {"id": cid, "name": name, "arguments": args, "malformed_arguments": "__malformed_arguments__" in args,
             "unknown_tool": name.startswith("nope")}],
            "tool_results": [{"id": cid, "name": name, "content": "", "error": err}], "errors": []})
    if extra_turn:
        turns.append({"message_index": len(messages), "parsed_tool_calls": [], "tool_results": [],
                      "errors": [], **extra_turn})
    return {"task_id": "x", "reward": 0.0, "termination_reason": "user_stop",
            "messages": messages, "agent_turns": turns, "reward_info": {}}


def test_rules_on_synthetic_traces():
    order = {"order_id": "#W1111111"}
    assert labels(_synthetic([("get_order_details", order, False)])) == set()
    assert "loop" in labels(_synthetic([("cancel_pending_order", order, True), ("cancel_pending_order", order, True)]))
    assert "loop" in labels(_synthetic([("get_order_details", order, False)] * 3))
    assert "loop" not in labels(_synthetic([("get_order_details", order, False)] * 2))
    assert "unknown_tool" in labels(_synthetic([("nope_tool", {}, True)]))
    assert "malformed_args" in labels(_synthetic([("get_order_details", {"__malformed_arguments__": "{"}, True)]))
    assert "malformed_args" in labels(_synthetic([], {"tool_call_parse": "parse_failed"}))
    assert "context_overflow" in labels(_synthetic([], {"context_trimmed": True}))
    # error then a lookup is fine; error then a write is not
    assert "tool_error_ignored" not in labels(
        _synthetic([("cancel_pending_order", order, True), ("get_order_details", order, False)]))
    assert "tool_error_ignored" in labels(
        _synthetic([("cancel_pending_order", order, True), ("transfer_to_human_agents", {"summary": "s"}, False)]))
    assert "invented_id" in labels(_synthetic([("get_order_details", {"order_id": "#W9999999"}, True)]))


def test_infra_error_is_environment():
    t = {"task_id": "3", "reward": None, "termination_reason": "infrastructure_error",
         "error": "too many tokens", "messages": [], "agent_turns": []}
    row = c.classify_run(t)
    assert row["labels"] == ["environment"] and row["route"] == "no_fix"


@pytest.mark.parametrize("layer,expected", [("tool", "tool_fix"), ("model", "weight_fix"),
                                             ("environment", "no_fix"), ("bogus", "no_fix")])
def test_router(layer, expected):
    assert c.route(layer) == expected


def test_judge_parse_retry_and_routing():
    t = _synthetic([("get_order_details", {"order_id": "#W1111111"}, False)])
    replies = iter([
        "I think it is wrong_tool",
        '```json\n{"first_wrong_turn": "1", "label": "invented_answer", "fault_layer": "model", "explanation": "x"}\n```',
    ])
    seen = []

    def chat(messages):
        seen.append(messages)
        return next(replies)

    row = c.classify_run(t, "goal", chat)
    assert len(seen) == 2 and "Invalid reply" in seen[1][-1]["content"]
    assert row["source"] == "judge" and row["labels"] == ["invented_answer"]
    assert row["first_wrong_turn"] == 1 and row["route"] == "weight_fix"
    assert "goal" in seen[0][1]["content"] and "[1] agent calls get_order_details" in seen[0][1]["content"]


def test_judge_failure_falls_back():
    t = _synthetic([("get_order_details", {"order_id": "#W1111111"}, False)])
    row = c.classify_run(t, None, lambda m: '{"label": "nonsense"}')
    assert row["source"] == "unclassified" and "judge_error" in row


def test_judge_skipped_when_rules_explain():
    t = load("invented_user_id_and_loop")
    row = c.classify_run(t, None, lambda m: pytest.fail("judge must not be called"))
    assert row["source"] == "rules"


def test_summary_counts():
    rows = [c.classify_run(load(n), example_values=EXAMPLES)
            for n in ("wrong_tool_delivered_order", "invented_user_id_and_loop", "premature_transfer")]
    s = c.summarize_failures(rows, 4)
    assert s["num_failed"] == 3 and s["num_runs"] == 4
    assert sum(s["per_route"].values()) == 3
    assert s["per_label"]["loop"] >= 2

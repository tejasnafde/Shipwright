import json

import pytest

from shipwright.llm import prompted_tools as pt
from shipwright.heal import tool_fix as tf


def _tool(name, desc, params):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "required": list(params),
                       "properties": {k: {"type": "string", "description": v} for k, v in params.items()}}}}


TOOLS = [
    _tool("get_order_details", "Get the status and details of an order.",
          {"order_id": "The order id, such as '#W0000000'. Be careful there is a '#' symbol."}),
    _tool("get_user_details", "Get the details of a user.", {"user_id": "The user id, such as 'sara_doe_496'."}),
    _tool("modify_pending_order_payment", "Modify the payment of a pending order.",
          {"payment_method_id": "such as 'gift_card_0000000' or 'credit_card_0000000'.",
           "item_id": "The item id, such as '1008292230'."}),
]
BY_NAME = tf.tool_map(TOOLS)


def test_concrete_ids_and_placeholders():
    text = "ids: #W0000000, sara_doe_496, gift_card_0000000, credit_card_1234, 1008292230"
    assert len(tf.concrete_ids(text)) == 5
    out = tf.replace_ids(text)
    assert tf.concrete_ids(out) == []
    assert "#W<7 digits>" in out and "<first>_<last>_<digits>" in out
    assert "gift_card_<digits>" in out and "<10 digits>" in out
    assert tf.concrete_ids("Ask the user for the order id (#W<7 digits>).") == []


def test_deterministic_patches_cover_every_example():
    patches = tf.deterministic_patches(BY_NAME)
    fields = {(p["tool"], p["field"]) for p in patches}
    assert fields == {
        ("get_order_details", "parameters.order_id.description"),
        ("get_user_details", "parameters.user_id.description"),
        ("modify_pending_order_payment", "parameters.payment_method_id.description"),
        ("modify_pending_order_payment", "parameters.item_id.description"),
    }
    assert all(tf.validate_patch(p, BY_NAME) == [] for p in patches)


@pytest.mark.parametrize("patch,reason", [
    ({"tool": "get_order_details", "field": "description", "new_text": "Use for #W1234567."}, "concrete"),
    ({"tool": "nope", "field": "description", "new_text": "x"}, "unknown tool"),
    ({"tool": "get_order_details", "field": "name", "new_text": "x"}, "not an editable"),
    ({"tool": "get_order_details", "field": "parameters.order_id.type", "new_text": "int"}, "not an editable"),
    ({"tool": "get_order_details", "field": "parameters.missing.description", "new_text": "x"}, "not an editable"),
    ({"tool": "get_order_details", "field": "description", "new_text": "  "}, "empty"),
])
def test_validate_rejects(patch, reason):
    errors = tf.validate_patch(patch, BY_NAME)
    assert errors and any(reason in e for e in errors)


def _ledger_with(patches, round_=1):
    rows = []
    for p in patches:
        tools, _ = tf.apply_patches(TOOLS, rows)
        before = tf.get_field(tf.tool_map(tools)[p["tool"]], p["field"])
        rows.append(tf.make_row(p, before, round_, "test", ["sim1"]))
    return rows


def test_apply_and_rollback():
    p = {"tool": "get_order_details", "field": "description",
         "new_text": "Get an order. Call it before any change to check the order status."}
    rows = _ledger_with([p])
    assert rows[0]["status"] == "proposed"
    assert rows[0]["sha256_before"] == tf.sha256("Get the status and details of an order.")
    # proposed patches do nothing
    tools, active = tf.apply_patches(TOOLS, rows)
    assert active == [] and tools == TOOLS

    rows = tf.set_status(rows, rows[0]["id"], "applied", TOOLS)
    tools, active = tf.apply_patches(TOOLS, rows)
    assert active == [rows[0]["id"]]
    assert tf.get_field(tf.tool_map(tools)["get_order_details"], "description") == p["new_text"]
    assert TOOLS[0]["function"]["description"] == "Get the status and details of an order."  # no mutation
    # names, params and types are untouched
    assert tf.tool_map(tools)["get_order_details"]["function"]["parameters"] == TOOLS[0]["function"]["parameters"]

    rows = tf.set_status(rows, rows[0]["id"], "rolled_back", TOOLS)
    tools, active = tf.apply_patches(TOOLS, rows)
    assert active == [] and tools == TOOLS
    assert [h["status"] for h in rows[0]["history"]] == ["proposed", "applied", "rolled_back"]


def test_stacked_patches_and_rollback_conflict():
    p1 = {"tool": "get_order_details", "field": "description", "new_text": "Version one."}
    rows = _ledger_with([p1])
    rows = tf.set_status(rows, rows[0]["id"], "applied", TOOLS)
    p2 = {"tool": "get_order_details", "field": "description", "new_text": "Version two."}
    tools, _ = tf.apply_patches(TOOLS, rows)
    rows.append(tf.make_row(p2, tf.get_field(tf.tool_map(tools)["get_order_details"], "description"), 2, "t", []))
    rows = tf.set_status(rows, rows[1]["id"], "applied", TOOLS)
    tools, active = tf.apply_patches(TOOLS, rows)
    assert active == [rows[0]["id"], rows[1]["id"]]
    # rolling back the first while the second (built on it) is applied is a conflict
    with pytest.raises(ValueError):
        tf.set_status(rows, rows[0]["id"], "rolled_back", TOOLS)
    rows = tf.set_status(rows, rows[1]["id"], "rolled_back", TOOLS)
    rows = tf.set_status(rows, rows[0]["id"], "rolled_back", TOOLS)
    assert tf.apply_patches(TOOLS, rows)[1] == []


def test_ledger_roundtrip(tmp_path):
    rows = _ledger_with(tf.deterministic_patches(BY_NAME))
    path = tmp_path / "patches" / "ledger.jsonl"
    tf.save_ledger(path, rows)
    assert tf.load_ledger(path) == rows
    assert tf.next_round(rows) == 2
    with pytest.raises(KeyError):
        tf.set_status(rows, "tp-missing", "applied")


def _failure(sim_id, labels, route="tool_fix"):
    return {"task_id": "1", "simulation_id": sim_id, "labels": labels, "route": route,
            "first_wrong_turn": 2, "explanation": "called get_order_details with an invented id",
            "rule_hits": [{"label": "invented_id", "turn": 2, "detail": "get_order_details: #W0000000"}]}


TRACE = {"simulation_id": "s1", "agent_turns": [{"message_index": 2, "parsed_tool_calls": [
    {"id": "c", "name": "get_order_details", "arguments": {"order_id": "#W0000000"}}],
    "tool_results": [{"id": "c", "content": "Error: order not found", "error": True}]}]}


def test_propose_rules_only():
    rows, notes = tf.propose([_failure("s1", ["tool_example_copied", "invented_id"])],
                             {"s1": TRACE}, TOOLS, [], None)
    assert len(rows) == 4 and notes == []
    assert all(r["source"] == "rules" and r["status"] == "proposed" and r["round"] == 1 for r in rows)
    assert all(r["source_failures"] == ["s1"] for r in rows)
    # proposing again gives nothing new
    again, _ = tf.propose([_failure("s1", ["tool_example_copied"])], {"s1": TRACE}, TOOLS, rows, None)
    assert again == []


def test_propose_llm_rejects_ids_then_retries():
    replies = iter([
        json.dumps({"patches": [{"tool": "get_order_details", "field": "parameters.order_id.description",
                                 "new_text": "Order id like #W1234567.", "reason": "r"}]}),
        json.dumps({"patches": [{"tool": "get_order_details", "field": "parameters.order_id.description",
                                 "new_text": "Order id from the user or get_user_details, format #W<7 digits>. "
                                             "Never guess it.", "reason": "invented ids"}]}),
    ])
    calls = []

    def chat(messages):
        calls.append(messages)
        return next(replies)

    rows, notes = tf.propose([_failure("s1", ["tool_example_copied", "wrong_tool"])],
                             {"s1": TRACE}, TOOLS, [], chat)
    assert len(calls) == 2 and "concrete example ids" in calls[1][-1]["content"]
    assert any("concrete example ids" in n for n in notes)
    order = [r for r in rows if r["field"] == "parameters.order_id.description"]
    assert len(order) == 1 and order[0]["source"] == "llm"  # LLM text replaced the placeholder patch
    assert "Never guess" in order[0]["after"]
    assert "called get_order_details" in calls[0][1]["content"]


def test_prompt_uses_applied_patches():
    p = {"tool": "get_order_details", "field": "description",
         "new_text": "Get an order. Always call it before changing an order."}
    rows = _ledger_with([p])
    rows = tf.set_status(rows, rows[0]["id"], "applied", TOOLS)
    try:
        pt.set_tool_patches(rows)
        kwargs, info = pt.prepare_request({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                           "tools": TOOLS})
        system = kwargs["messages"][0]["content"]
        assert "Always call it before changing an order." in system  # full patched text shown
        assert info["tool_patches"] == [rows[0]["id"]]
    finally:
        pt.set_tool_patches([])
    kwargs, info = pt.prepare_request({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                       "tools": TOOLS})
    assert "Always call it" not in kwargs["messages"][0]["content"] and "tool_patches" not in info

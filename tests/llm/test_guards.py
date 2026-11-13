"""Tool-call guards (src/shipwright/llm/guards.py) and how patches.py applies them. Offline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from shipwright.llm import guards, patches
from shipwright.llm.prompted_tools import TRACE_FIELD

ROOT = Path(__file__).resolve().parents[2]


def customer(text):
    return {"role": "user", "content": text}


def agent_text(text):
    return {"role": "assistant", "content": text}


def agent_call(name, arguments, call_id):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def call(name, **arguments):
    return guards.ToolCall(name=name, arguments=arguments)


GREETING = [agent_text("Hi! How can I help you today?"), customer("I want to cancel order #W2378156.")]
AUTHENTICATED = GREETING + [
    agent_text("Sure. What is your email?"),
    customer("yusuf@example.com"),
    agent_call("find_user_id_by_email", {"email": "yusuf@example.com"}, "c1"),
    tool_result("c1", "yusuf_rossi_9620"),
]


def test_authentication_comes_first_even_with_a_known_user_id():
    violation = guards.first_violation(GREETING, call("get_order_details", order_id="#W2378156"))
    assert violation.rule == "not_authenticated"
    assert guards.first_violation(GREETING, call("find_user_id_by_email", email="yusuf@example.com")) is None
    assert guards.first_violation(AUTHENTICATED, call("get_order_details", order_id="#W2378156")) is None
    failed_lookup = GREETING + [agent_call("find_user_id_by_email", {"email": "x@y.z"}, "c9"),
                                tool_result("c9", "Error: User not found")]
    assert guards.first_violation(failed_lookup, call("get_user_details", user_id="x")).rule == "not_authenticated"


def test_made_up_ids_are_rejected_and_hash_prefix_is_ignored():
    assert guards.first_violation(AUTHENTICATED, call("get_order_details", order_id="#W1234567")).rule == "made_up_id"
    without_hash = AUTHENTICATED[:-1] + [tool_result("c1", "yusuf_rossi_9620"), customer("It is W2378156")]
    assert guards.first_violation(without_hash, call("get_order_details", order_id="#W2378156")) is None
    assert guards.first_violation(AUTHENTICATED, call("get_product_details", product_id="T-Shirt")).rule == "made_up_id"


def test_early_transfer_is_blocked_until_authentication_was_tried():
    assert guards.first_violation(GREETING, call("transfer_to_human_agents", summary="x")).rule == "early_transfer"
    # The customer cannot be found (round 1 teacher, task 57): transferring is then allowed.
    cannot_find = GREETING + [agent_call("find_user_id_by_name_zip", {"first_name": "a", "last_name": "", "zip": "1"}, "c2"),
                              tool_result("c2", "Error: User not found")]
    assert guards.first_violation(cannot_find, call("transfer_to_human_agents", summary="x")) is None


def test_changes_need_an_explicit_yes_to_a_question():
    ready = AUTHENTICATED + [agent_call("get_order_details", {"order_id": "#W2378156"}, "c3"),
                             tool_result("c3", '{"order_id": "#W2378156", "status": "pending"}')]
    cancel = call("cancel_pending_order", order_id="#W2378156", reason="no longer needed")
    assert guards.first_violation(ready, cancel).rule == "no_confirmation"
    asked = ready + [agent_text("I will cancel #W2378156 (reason: no longer needed). Shall I proceed?"),
                     customer("Yes, please go ahead.")]
    assert guards.first_violation(asked, cancel) is None
    refused = ready + [agent_text("Shall I cancel it?"), customer("No, wait.")]
    assert guards.first_violation(refused, cancel).rule == "no_confirmation"


def test_repeating_a_failed_call_is_rejected():
    failed = AUTHENTICATED + [customer("order W2378156"),
                              agent_call("get_order_details", {"order_id": "#W2378156"}, "c4"),
                              tool_result("c4", "Error: Order not found")]
    violation = guards.first_violation(failed, call("get_order_details", order_id="#W2378156"))
    assert violation.rule == "repeated_failed_call" and "Order not found" in violation.message


# ---- patches.py: asking again ----


def reply(text, tool_name=None, arguments=None):
    tool_calls = None
    if tool_name:
        tool_calls = [SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments or {})))]
    message = SimpleNamespace(content=None if tool_calls else text, tool_calls=tool_calls,
                              provider_specific_fields={TRACE_FIELD: {"raw_text": text, "parse": "tool_call"}})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_policy_violation_is_asked_again_and_recorded():
    requests = []

    def ask_model(request):
        requests.append(request)
        return reply('{"name": "find_user_id_by_email"}', "find_user_id_by_email", {"email": "yusuf@example.com"})

    first = reply('{"name": "cancel_pending_order"}', "cancel_pending_order", {"order_id": "#W2378156"})
    final = patches.ask_again_until_policy_ok(GREETING, {"messages": [{"role": "system", "content": "S"}]}, first, ask_model)
    assert len(requests) == 1
    sent = requests[0]["messages"]
    assert sent[-2] == {"role": "assistant", "content": '{"name": "cancel_pending_order"}'}
    assert sent[-1]["role"] == "user" and "not authenticated" in sent[-1]["content"]
    fields = final.choices[0].message.provider_specific_fields[TRACE_FIELD]
    assert fields["guard_rejections"][0]["rule"] == "not_authenticated"
    assert fields["guard_allowed_after_retries"] is None


def test_retries_are_capped_and_the_call_then_runs():
    stubborn = reply("transfer", "transfer_to_human_agents", {"summary": "x"})
    asked = []
    final = patches.ask_again_until_policy_ok(GREETING, {"messages": []}, stubborn,
                                              lambda request: asked.append(request) or stubborn)
    assert len(asked) == guards.MAX_GUARD_RETRIES
    fields = final.choices[0].message.provider_specific_fields[TRACE_FIELD]
    assert len(fields["guard_rejections"]) == guards.MAX_GUARD_RETRIES
    assert fields["guard_allowed_after_retries"] == "early_transfer"


def test_good_call_and_text_reply_are_not_touched():
    def must_not_ask(request):
        raise AssertionError("asked again")

    good = reply("x", "find_user_id_by_email", {"email": "a@b.c"})
    assert patches.ask_again_until_policy_ok(GREETING, {"messages": []}, good, must_not_ask) is good
    text = reply("What is your email?")
    assert patches.ask_again_until_policy_ok(GREETING, {"messages": []}, text, must_not_ask) is text


def test_transient_errors_are_retried_then_raised():
    class RateLimitError(Exception):
        status_code = 429

    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise RateLimitError()
        return "ok"

    assert patches.call_with_transient_retries(flaky, {"model": "m"}, sleep=lambda seconds: None) == "ok"
    assert len(attempts) == 3

    def always_fails(**kwargs):
        raise RateLimitError()

    with pytest.raises(RateLimitError):
        patches.call_with_transient_retries(always_fails, {}, sleep=lambda seconds: None)

    def bad_request(**kwargs):
        error = ValueError("bad")
        error.status_code = 400
        raise error

    with pytest.raises(ValueError):  # not transient: no retry
        patches.call_with_transient_retries(bad_request, {}, sleep=lambda seconds: pytest.fail("waited"))


def test_empty_user_reply_is_detected():
    assert patches.is_empty_reply(reply("   "))
    assert not patches.is_empty_reply(reply("Hi"))


def test_dry_run_with_guards_records_rejections(tmp_path):
    pytest.importorskip("tau2")
    output_directory = tmp_path / "guarded"
    result = subprocess.run([sys.executable, "scripts/baseline.py", "--dry-run", "--num-tasks", "1",
                             "--tool-guards", "--output-dir", str(output_directory)],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    summary = json.loads((output_directory / "summary.json").read_text())
    assert summary["tool_guards"] is True
    assert sum(summary["guard_rejections_by_rule"].values()) >= 1


def test_order_status_must_fit_the_tool():
    looked_up = AUTHENTICATED + [agent_call("get_order_details", {"order_id": "#W2378156"}, "c5"),
                                 tool_result("c5", '{"order_id": "#W2378156", "status": "delivered"}'),
                                 agent_text("Cancel #W2378156?"), customer("yes")]
    violation = guards.first_violation(looked_up, call("cancel_pending_order", order_id="#W2378156", reason="x"))
    assert violation.rule == "wrong_order_status" and "delivered" in violation.message
    exchange = call("exchange_delivered_order_items", order_id="#W2378156", item_ids=[], new_item_ids=[],
                    payment_method_id="")
    assert guards.check_order_status_fits_tool(looked_up, exchange) is None


# ---- the guards added after rounds 3 to 5 ----

ORDER_RESULT = json.dumps({"order_id": "#W1", "status": "delivered", "items": [
    {"name": "Bookshelf", "product_id": "P-SHELF", "item_id": "I-SHELF", "price": 100.0}]})
PRODUCT_RESULT = json.dumps({"name": "Digital Camera", "product_id": "P-CAMERA", "variants": {
    "I-CAM": {"item_id": "I-CAM", "available": True}}})
LOOKED_UP = AUTHENTICATED + [
    agent_call("get_order_details", {"order_id": "#W1"}, "c2"), tool_result("c2", ORDER_RESULT),
    agent_call("get_product_details", {"product_id": "P-CAMERA"}, "c3"), tool_result("c3", PRODUCT_RESULT),
    agent_text("I can exchange it. Shall I go ahead?"), customer("yes please")]


def test_an_exchange_to_another_product_type_is_rejected_with_a_text_fallback():
    exchange = call("exchange_delivered_order_items", order_id="#W1", item_ids=["I-SHELF"], new_item_ids=["I-CAM"])
    violation = guards.first_violation(LOOKED_UP, exchange)
    assert violation.rule == "exchange_other_product_type" and "Bookshelf" in violation.message
    assert "isn't possible" in violation.fallback_reply
    same_product = ORDER_RESULT.replace("P-SHELF", "P-CAMERA")  # the same product in another option is allowed
    allowed = LOOKED_UP[:-6] + [agent_call("get_order_details", {"order_id": "#W1"}, "c2"), tool_result("c2", same_product),
                                agent_call("get_product_details", {"product_id": "P-CAMERA"}, "c3"),
                                tool_result("c3", PRODUCT_RESULT), agent_text("Go ahead?"), customer("yes")]
    assert guards.first_violation(allowed, exchange) is None
    unknown = call("exchange_delivered_order_items", order_id="#W1", item_ids=["I-SHELF"], new_item_ids=["I-NEVER-SEEN"])
    assert guards.check_same_product_type(LOOKED_UP, unknown) is None  # an unknown item is the made-up-id guard's job


def test_one_change_per_order():
    first = LOOKED_UP + [agent_call("exchange_delivered_order_items", {"order_id": "#W1", "item_ids": ["I-SHELF"],
                         "new_item_ids": ["I-OTHER"]}, "c4"), tool_result("c4", '{"status": "exchange requested"}')]
    second = call("exchange_delivered_order_items", order_id="#W1", item_ids=["I-SHELF"], new_item_ids=["I-OTHER"])
    assert guards.check_one_change_per_order(first, second).rule == "second_change_to_one_order"
    assert guards.check_one_change_per_order(LOOKED_UP, second) is None


def test_a_repeated_lookup_is_rejected_unless_the_data_changed_in_between():
    repeat = call("get_order_details", order_id="#W1")
    assert guards.check_repeated_call(LOOKED_UP, repeat).rule == "repeated_call"
    after_a_change = LOOKED_UP + [agent_call("cancel_pending_order", {"order_id": "#W1", "reason": "x"}, "c5"),
                                  tool_result("c5", "{}")]
    assert guards.check_repeated_call(after_a_change, repeat) is None  # checking the result of a change is fine
    assert guards.check_repeated_call(LOOKED_UP, call("get_order_details", order_id="#W2")) is None


def test_a_stubborn_repeat_is_replaced_by_the_fallback_text_instead_of_running():
    stubborn = reply("again", "get_order_details", {"order_id": "#W1"})
    asked = []
    final = patches.ask_again_until_policy_ok(LOOKED_UP, {"messages": []}, stubborn,
                                              lambda request: asked.append(request) or stubborn)
    message = final.choices[0].message
    assert len(asked) == guards.MAX_GUARD_RETRIES
    assert message.tool_calls is None and "already looked that up" in message.content
    fields = message.provider_specific_fields[TRACE_FIELD]
    assert fields["guard_replaced_with_text"] == "repeated_call" and fields["guard_allowed_after_retries"] is None


def order_and_listing_messages():
    """An order whose item 111 is of product P1, and a product listing of P2 with variant 222."""
    return [
        {"role": "user", "content": "email a@b.c"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
            "name": "get_order_details", "arguments": '{"order_id": "#W1"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": json.dumps({"order_id": "#W1", "items": [
            {"name": "Camera", "product_id": "P1", "item_id": "111"}]})},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {
            "name": "get_product_details", "arguments": '{"product_id": "P2"}'}}]},
        {"role": "tool", "tool_call_id": "b", "content": json.dumps({"name": "Shelf", "product_id": "P2", "variants": {
            "222": {"item_id": "222", "available": True}}})},
    ]


def test_an_item_id_given_to_get_product_details_becomes_get_item_details():
    messages = order_and_listing_messages()
    for item_id in ("222", "111"):  # a variant from a listing, an item from an order
        fixed = guards.repaired_call(messages, guards.ToolCall("get_product_details", {"product_id": item_id}))
        assert fixed == guards.ToolCall("get_item_details", {"item_id": item_id})


def test_a_real_product_id_or_an_unknown_id_is_left_alone():
    messages = order_and_listing_messages()
    assert guards.repaired_call(messages, guards.ToolCall("get_product_details", {"product_id": "P2"})) is None
    assert guards.repaired_call(messages, guards.ToolCall("get_product_details", {"product_id": "999"})) is None
    assert guards.repaired_call(messages, guards.ToolCall("get_order_details", {"order_id": "222"})) is None


def test_the_repeat_warning_carries_the_earlier_result():
    messages = order_and_listing_messages()
    violation = guards.check_repeated_call(messages, guards.ToolCall("get_product_details", {"product_id": "P2"}))
    assert violation is not None and '"variants"' in violation.message and violation.rule == "repeated_call"

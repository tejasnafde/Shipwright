"""Shortening old tool results and replies before any message is dropped (shipwright.llm.context)."""

import json

from shipwright.llm import context

LISTING = {"name": "Shelf", "product_id": "P2", "variants": {
    "111": {"item_id": "111", "options": {"color": "red"}, "available": True, "price": 10.0},
    "222": {"item_id": "222", "options": {"color": "blue"}, "available": False, "price": 12.0}}}
ORDER = {"order_id": "#W1", "user_id": "u_1", "address": {"city": "X" * 80}, "items": [{"item_id": "111", "product_id": "P2"}],
         "status": "pending", "fulfillments": [], "payment_history": [], "cancel_reason": None}


def tool_turn(name, payload):
    return {"role": "user", "content": f"Tool result for {name}: " + json.dumps(payload, separators=(",", ":"))}


def test_a_product_listing_keeps_only_what_can_be_ordered():
    short = context.shortened_result("get_product_details", LISTING)
    assert set(short["available_variants"]) == {"111"} and short["unavailable_variants"] == 1 and short["shortened"]
    assert short["available_variants"]["111"] == {"options": {"color": "red"}, "price": 10.0}


def test_an_order_keeps_its_items_and_status_and_loses_the_address():
    short = context.shortened_result("get_order_details", ORDER)
    assert short["items"] == ORDER["items"] and short["status"] == "pending" and "address" not in short


def test_an_unknown_tool_result_is_left_alone():
    payload = {"item_id": "1", "price": 2}
    assert context.shortened_result("get_item_details", payload) is payload


def test_the_newest_results_stay_whole_and_nothing_is_shortened_when_it_fits():
    messages = [tool_turn("get_product_details", LISTING) for _ in range(4)]
    shortened, count = context.shorten_old_tool_results(messages, budget=0)  # nothing can fit: shorten all that are allowed
    assert count == 2 and "available_variants" in shortened[0]["content"] and "available_variants" not in shortened[3]["content"]
    untouched, none = context.shorten_old_tool_results(messages, budget=10**6)
    assert none == 0 and untouched == messages


def test_a_tool_call_reply_is_never_cut_but_a_long_text_reply_is():
    text = {"role": "assistant", "content": "word " * 200}
    call = {"role": "assistant", "content": "```json\n" + "x" * 500 + "\n```"}
    messages = [text, call, text, text, text]
    shortened, count = context.shorten_old_replies(messages, budget=0)
    assert shortened[1] == call and shortened[0]["content"].endswith("[shortened]") and count == 2


def test_fit_messages_shortens_before_it_drops_anything():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hi"}]
    for _ in range(6):
        messages += [tool_turn("get_product_details", LISTING), {"role": "assistant", "content": "ok"}]
    budget = context.count_messages(messages) - 10
    fitted, info = context.fit_messages(messages, budget=budget)
    assert info["results_shortened"] >= 1 and info["dropped_messages"] == 0 and info["truncated_chars"] == 0
    assert context.count_messages(fitted) <= budget


def test_a_product_listing_is_shortened_the_moment_it_reaches_the_student_and_other_results_are_not():
    from shipwright.llm import prompted_tools

    listing = prompted_tools.observation_text("get_product_details", json.dumps(LISTING))
    assert '"available_variants"' in listing and '"222"' not in listing and "unavailable_variants" in listing
    order = prompted_tools.observation_text("get_order_details", json.dumps(ORDER))
    assert '"address"' in order  # an order is shown whole when it arrives
    assert prompted_tools.observation_text("get_product_details", "Error: Product not found") == "Error: Product not found"


def test_training_rows_and_live_requests_show_the_same_listing():
    from shipwright.llm import prompted_tools

    history = [{"role": "system", "content": "policy"}, {"role": "user", "content": "hi"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {
                   "name": "get_product_details", "arguments": '{"product_id": "P2"}'}}]},
               {"role": "tool", "tool_call_id": "a", "content": json.dumps(LISTING)}]
    converted = prompted_tools.convert_messages(history, [])
    assert "available_variants" in converted[-1]["content"]

import json

from litellm import ModelResponse

from shipwright.llm import prompted_tools as pt

TOOLS = [{"type": "function", "function": {"name": "get_order_details", "description": "d",
                                            "parameters": {"type": "object", "properties": {}}}}]


def test_parse_fenced():
    text = 'Sure.\n```json\n{"name": "get_order_details", "arguments": {"order_id": "#W1"}}\n```\nthanks'
    assert pt.parse_tool_call(text) == ("get_order_details", {"order_id": "#W1"})


def test_parse_no_fence_trailing_text_and_nested_braces():
    text = '{"name": "f", "arguments": {"a": {"b": "}"}}} and then more {text}'
    assert pt.parse_tool_call(text) == ("f", {"a": {"b": "}"}})


def test_parse_single_quotes_and_parameters_key_and_string_args():
    assert pt.parse_tool_call("{'name': 'f', 'arguments': {'x': 1}}") == ("f", {"x": 1})
    assert pt.parse_tool_call('{"name": "f", "parameters": {"x": 1}}') == ("f", {"x": 1})
    assert pt.parse_tool_call('{"name": "f", "arguments": "{\\"x\\": 1}"}') == ("f", {"x": 1})
    assert pt.parse_tool_call('{"name": "f"}') == ("f", {})


def test_parse_failures():
    assert pt.parse_tool_call("Hello, how can I help?") is None
    assert pt.parse_tool_call('```json\n{"name": "f", "arguments": {"a": 1}\n```') is None
    assert pt.parse_tool_call('{"order": "x"}') is None
    assert pt.parse_tool_call('{"name": "f", "arguments": [1]}') is None
    assert pt.looks_like_tool_attempt('```json\n{"name": "f"')
    assert not pt.looks_like_tool_attempt("Hello")


def test_convert_messages():
    history = [
        {"role": "system", "content": "Policy."},
        {"role": "user", "content": "Where is #W1?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_order_details", "arguments": '{"order_id": "#W1"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"status": "pending"}'},
    ]
    out = pt.convert_messages(history, TOOLS)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[0]["content"].startswith("Policy.") and "get_order_details" in out[0]["content"]
    assert pt.parse_tool_call(out[2]["content"]) == ("get_order_details", {"order_id": "#W1"})
    assert out[3]["content"] == 'Tool result for get_order_details: {"status":"pending"}'
    assert all("tool_calls" not in m for m in out)


def test_convert_messages_merges_user_turns_and_adds_system():
    history = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": "and?"},
    ]
    out = pt.convert_messages(history, TOOLS)
    assert out[0]["role"] == "system"
    assert [m["role"] for m in out] == ["system", "assistant", "user"]
    assert out[2]["content"] == "Tool result for f: ok\n\nand?"


def test_prepare_request_and_convert_response():
    req, info = pt.prepare_request({"model": "openai/CohereLabs/tiny-aya-global:cohere",
                              "messages": [{"role": "user", "content": "hi"}],
                              "tools": TOOLS, "tool_choice": "auto", "temperature": 0})
    assert "tools" not in req and "tool_choice" not in req and req["temperature"] == 0
    assert req["messages"][0]["role"] == "system"
    assert req["max_tokens"] == pt.REPLY_TOKENS and info["dropped_messages"] == 0

    raw = '```json\n{"name": "get_order_details", "arguments": {"order_id": "#W1"}}\n```'
    resp = ModelResponse(model="m", choices=[{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": raw}}])
    pt.convert_response(resp)
    msg = resp.choices[0].message
    assert msg.content is None
    assert msg.tool_calls[0].function.name == "get_order_details"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"order_id": "#W1"}
    d = resp.to_dict()["choices"][0]["message"]["provider_specific_fields"]["shipwright"]
    assert d == {"raw_text": raw, "parse": "tool_call"}

    resp2 = ModelResponse(model="m", choices=[{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": "Hello!"}}])
    pt.convert_response(resp2)
    assert resp2.choices[0].message.content == "Hello!" and not resp2.choices[0].message.tool_calls
    assert resp2.choices[0].message.provider_specific_fields["shipwright"]["parse"] == "text"


def test_is_prompted_model():
    assert pt.is_prompted_model("openai/CohereLabs/tiny-aya-global:cohere")
    assert not pt.is_prompted_model("openai/openai/gpt-oss-120b:cerebras")


def test_compact_tool_line():
    tool = {"type": "function", "function": {
        "name": "cancel_pending_order", "description": "Cancel a pending order. More text here.",
        "parameters": {"type": "object", "required": ["order_id"], "properties": {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "reason": {"type": "string", "description": "either 'no longer needed' or 'ordered by mistake'"},
            "item_ids": {"type": "array", "items": {"type": "string"}}}}}}
    assert pt.compact_tool_line(tool) == (
        f"cancel_pending_order(order_id: str {pt.NOT_AN_EXAMPLE_HINT}, "
        "reason?: str one of 'no longer needed'|'ordered by mistake', item_ids?: list[str])"
        " - Cancel a pending order."
    )


def test_example_hint_hides_real_looking_examples_only():
    for real_looking in ["'#W0000000'", "'sara_doe_496'", "'6086499569'", "'something@example.com'"]:
        assert pt.example_hint(real_looking) == pt.NOT_AN_EXAMPLE_HINT
    assert pt.example_hint("'John'") == "e.g. 'John'"
    assert "#W0000000" not in pt.render_tools_prompt([{"type": "function", "function": {
        "name": "get_order_details", "description": "Get an order.",
        "parameters": {"type": "object", "required": ["order_id"], "properties": {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."}}}}}])


def test_minify_content():
    assert pt.minify_content('{\n  "a": [1, 2]\n}') == '{"a":[1,2]}'
    assert pt.minify_content("not json") == "not json"


def test_convert_messages_drops_the_opening_greeting_so_roles_alternate():
    history = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "cancel my order"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "find_user_id_by_email", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "u_1"},
        {"role": "assistant", "content": "Found you."},
    ]
    converted = pt.convert_messages(history, [])
    roles = [message["role"] for message in converted]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert "Hi! How can I help you today?" not in json.dumps(converted)


def test_order_and_user_records_arrive_compact():
    order = {"order_id": "#W1", "user_id": "u1", "status": "delivered",
             "address": {"address1": "313 Chestnut Street", "address2": "Suite 537", "city": "Jacksonville",
                         "country": "USA", "state": "FL", "zip": "32286"},
             "items": [{"name": "Hose", "product_id": "p1", "item_id": "i1", "price": 96.35, "options": {"length": "25ft"}}],
             "fulfillments": [], "payment_history": [{"transaction_type": "payment", "amount": 96.35,
                                                      "payment_method_id": "paypal_1"}],
             "cancel_reason": None, "exchange_items": None, "return_items": None}
    text = pt.observation_text("get_order_details", json.dumps(order))
    compact = json.loads(text)
    assert compact["address"] == "313 Chestnut Street, Suite 537, Jacksonville, FL, 32286, USA"
    assert "cancel_reason" not in compact and "fulfillments" not in compact
    assert compact["items"] == order["items"] and compact["payment_history"] == order["payment_history"]
    assert len(text) < len(json.dumps(order, separators=(",", ":")))
    # a write tool returns the changed order: compacted the same way
    assert json.loads(pt.observation_text("cancel_pending_order", json.dumps(order))) == compact
    # other results are only minified
    assert pt.observation_text("calculate", '{"result": 2}') == '{"result":2}'
    assert pt.observation_text("transfer_to_human_agents", "Transfer successful") == "Transfer successful"


def test_the_condensed_policy_replaces_only_the_policy_and_keeps_the_rules_that_decide_a_task():
    from shipwright.llm import prompted_tools

    system = ("<instructions>\nFollow the <policy> provided below.\nBe helpful.\n</instructions>\n<policy>\n# Retail agent policy\n\nLong text.\n\n"
              "## Cancel\n\nMore text.\n\n\nRespond in English using Latin script.\n</policy>")
    short = prompted_tools.condensed_system(system)
    assert short.startswith("<instructions>\nFollow the <policy> provided below.\nBe helpful.\n</instructions>\n<policy>\n#")
    assert "Long text" not in short
    assert short.endswith("Respond in English using Latin script.\n</policy>")
    for rule in ("'no longer needed' or 'ordered by mistake'", "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
                 "explicit \"yes\"", "'pending'", "'delivered'", "existing gift card", "same product"):
        assert rule in short
    assert prompted_tools.condensed_system("no policy here") == "no policy here"
    prompted_tools.set_condensed_policy(True)
    try:
        converted = prompted_tools.convert_messages([{"role": "system", "content": system}], [])
    finally:
        prompted_tools.set_condensed_policy(False)
    assert "Long text" not in converted[0]["content"] and "Retail agent policy (short)" in converted[0]["content"]

import json

from litellm import ModelResponse

from shipwright import prompted_tools as pt

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
    assert out[3]["content"] == 'Tool result for get_order_details: {"status": "pending"}'
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
    req = pt.prepare_request({"model": "openai/CohereLabs/tiny-aya-global:cohere",
                              "messages": [{"role": "user", "content": "hi"}],
                              "tools": TOOLS, "tool_choice": "auto", "temperature": 0})
    assert "tools" not in req and "tool_choice" not in req and req["temperature"] == 0
    assert req["messages"][0]["role"] == "system"

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

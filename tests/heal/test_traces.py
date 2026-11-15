import json
from types import SimpleNamespace

from shipwright.llm.patches import sanitize_tool_calls
from shipwright.heal.traces import MALFORMED_KEY, build_trace, infra_error_kind, summarize, write_jsonl

TOOLS = {"get_user_details", "find_user_id_by_email"}


def _assistant(content, calls, raw_args):
    raw_calls = [
        {"id": c["id"], "function": {"name": c["name"], "arguments": a}}
        for c, a in zip(calls, raw_args)
    ]
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": calls or None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "raw_data": {"choices": [{"message": {"content": content, "tool_calls": raw_calls},
                                  "finish_reason": "tool_calls"}]},
    }


def _sim():
    bad = json.dumps({MALFORMED_KEY: '{"user_id": '})
    return {
        "id": "s1", "task_id": "7", "trial": 0, "termination_reason": "user_stop",
        "reward_info": {"reward": 1.0},
        "messages": [
            {"role": "user", "content": "hi", "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
            _assistant(None, [{"id": "a", "name": "get_user_details", "arguments": {MALFORMED_KEY: '{"user_id": '}}], [bad]),
            {"role": "tool", "id": "a", "content": "Error: bad", "error": True},
            _assistant(None, [{"id": "b", "name": "nuke_db", "arguments": {}}], ["{}"]),
            {"role": "tool", "id": "b", "content": "Error: unknown", "error": True},
            _assistant('{"name": "get_user_details", "arguments": {}}', [], []),
        ],
    }


def test_build_trace_and_summary(tmp_path):
    t = build_trace(_sim(), TOOLS)
    assert t["passed"] and t["task_id"] == "7"
    turns = t["agent_turns"]
    assert len(turns) == 3
    assert turns[0]["parsed_tool_calls"][0]["malformed_arguments"]
    assert turns[0]["raw_tool_calls"][0]["arguments"] == '{"user_id": '  # original text restored
    assert turns[0]["tool_results"][0]["error"]
    assert turns[1]["parsed_tool_calls"][0]["unknown_tool"]
    assert turns[2]["text_tool_call"]
    assert all("raw_data" not in m for m in t["messages"])

    s = summarize([t])
    assert s["pass_rate"] == 1.0
    assert s["tool_calls"] == 2
    assert s["malformed_tool_calls"] == 1
    assert s["unknown_tool_calls"] == 1 and s["unknown_tool_names"] == {"nuke_db": 1}
    assert s["tool_errors"] == 2 and s["tool_calls_in_text"] == 1
    assert s["agent_tokens"]["prompt_tokens"] == 30
    assert s["user_tokens"]["prompt_tokens"] == 5

    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [t, t])
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["simulation_id"] == "s1"


def test_sanitize_tool_calls():
    def call(args):
        return SimpleNamespace(id=None, function=SimpleNamespace(name="f", arguments=args))

    calls = [call('{"a": 1}'), call('{"a": '), call("[1]"), call("")]
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))])
    assert sanitize_tool_calls(resp) == 2
    assert calls[0].function.arguments == '{"a": 1}'
    assert json.loads(calls[1].function.arguments) == {MALFORMED_KEY: '{"a": '}
    assert json.loads(calls[2].function.arguments) == {MALFORMED_KEY: "[1]"}
    assert calls[3].function.arguments == "{}"
    assert calls[0].id == ""


def test_parse_status_in_trace():
    msg = _assistant("```json\n{\"name\": \"x\"", [], [])
    msg["raw_data"]["choices"][0]["message"]["provider_specific_fields"] = {
        "shipwright": {"raw_text": msg["content"], "parse": "parse_failed"}
    }
    sim = {"id": "s", "task_id": "1", "reward_info": {"reward": 0.0}, "messages": [msg]}
    t = build_trace(sim, TOOLS)
    turn = t["agent_turns"][0]
    assert turn["tool_call_parse"] == "parse_failed"
    assert turn["raw_output"].startswith("```json")
    assert "tool_call_parse_failed" in turn["errors"] and not turn["text_tool_call"]
    assert summarize([t])["tool_call_parse_failures"] == 1


def test_infra_error_kind_names_the_cause():
    kinds = {
        "litellm.Timeout: APITimeoutError - Request timed out.": "timeout",
        "litellm.BadRequestError: OpenAIException - Error code: 400": "bad_request_400",
        "litellm.RateLimitError: 429": "rate_limit_429",
        "UserMessage must have either content or tool_calls. Got UserMessage": "empty_user_reply",
        "run stopped: spent the --max-cost-usd limit of $2.00": "cost_cap",
        "something else": "other",
        None: "other",
    }
    for error_text, expected_kind in kinds.items():
        assert infra_error_kind(error_text) == expected_kind

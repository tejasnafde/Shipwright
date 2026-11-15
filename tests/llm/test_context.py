from shipwright.llm import context as ctx
from shipwright.llm.patches import ModelRerouter
from shipwright.heal.traces import build_trace, summarize


def _conv(n_pairs, size=400):
    msgs = [{"role": "system", "content": "S" * 700}, {"role": "user", "content": "first user"}]
    for i in range(n_pairs):
        msgs.append({"role": "assistant", "content": f"a{i} " + "x" * size})
        msgs.append({"role": "user", "content": f"u{i} " + "y" * size})
    return msgs


def test_under_budget_unchanged():
    msgs = _conv(2)
    out, info = ctx.fit_messages(msgs, budget=10_000)
    assert out == msgs
    assert info["dropped_messages"] == 0 and info["truncated_chars"] == 0
    assert info["token_count_method"] == "estimate"


def test_drops_oldest_middle_turns():
    msgs = _conv(10)
    budget = 900
    out, info = ctx.fit_messages(msgs, budget=budget)
    assert info["dropped_messages"] > 0 and info["truncated_chars"] == 0
    assert ctx.count_messages(out) <= budget
    assert out[0] == msgs[0] and out[1] == msgs[1]  # system and first user kept
    assert out[-1] == msgs[-1]  # latest turn kept
    assert out[2]["content"].startswith(ctx.OMITTED_NOTE)
    roles = [m["role"] for m in out]
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))  # still alternating
    assert info["prompt_tokens_est"] == ctx.count_messages(out)


def test_truncates_latest_when_it_alone_is_too_big():
    msgs = _conv(1) + [{"role": "assistant", "content": "call"},
                       {"role": "user", "content": "Tool result for f: " + "z" * 20_000}]
    budget = 1500
    out, info = ctx.fit_messages(msgs, budget=budget)
    assert ctx.count_messages(out) <= budget
    assert info["truncated_chars"] > 0
    assert "Tool result for f: zzz" in out[-1]["content"]
    assert info["truncated_chars"] < 20_000
    assert out[-1]["content"].endswith(f"[truncated {info['truncated_chars']} chars]")


def test_budget_reserves_reply():
    assert ctx.prompt_budget() == ctx.CONTEXT_LIMIT - ctx.REPLY_TOKENS - ctx.ESTIMATE_MARGIN


def test_rerouter():
    r = ModelRerouter({"agent", "user"}, "user")
    assert r({"model": "agent"})["model"] == "agent"
    out = r({"model": "openai/responses/gpt-5.4-mini", "allowed_openai_params": ["tool_choice"]})
    assert out["model"] == "user" and "allowed_openai_params" not in out
    assert r.counts == {"openai/responses/gpt-5.4-mini": 1}


def test_summary_infra_and_trim():
    ok = {"id": "a", "task_id": "1", "termination_reason": "user_stop", "reward_info": {"reward": 1.0},
          "messages": [{"role": "assistant", "content": "hi", "raw_data": {"choices": [{"message": {
              "content": "hi", "provider_specific_fields": {"shipwright": {
                  "raw_text": "hi", "parse": "text", "prompt_tokens_est": 7000, "dropped_messages": 4,
                  "truncated_chars": 0}}}}]}}]}
    bad = {"id": "b", "task_id": "2", "termination_reason": "infrastructure_error", "reward_info": None,
           "messages": [], "info": {"error": "too many tokens: size limit exceeded by 13 tokens" + "!" * 300}}
    traces = [build_trace(s, set()) for s in (ok, bad, dict(bad, id="c"))]
    assert traces[0]["agent_turns"][0]["context_trimmed"]
    assert traces[0]["agent_turns"][0]["prompt_tokens_est"] == 7000
    s = summarize(traces)
    assert s["pass_rate"] == 1 / 3
    assert s["pass_rate_excluding_infra_errors"] == 1.0
    assert s["num_infra_errors"] == 2
    [(msg, count)] = s["infra_error_messages"].items()
    assert count == 2 and len(msg) == 150
    assert s["context_trimmed_turns"] == 1


def test_strip_json_fences():
    from types import SimpleNamespace

    from shipwright.llm.patches import is_nl_judge_request, strip_json_fences

    msg = SimpleNamespace(content='```json\n{"results": []}\n```')
    strip_json_fences(SimpleNamespace(choices=[SimpleNamespace(message=msg)]))
    assert msg.content == '{"results": []}'
    assert is_nl_judge_request({"messages": [{"role": "system", "content": "... metExpectation ..."}]})
    assert not is_nl_judge_request({"messages": [{"role": "system", "content": "You are a user."}]})


def test_is_json_reply():
    from types import SimpleNamespace as NS

    from shipwright.llm.patches import is_json_reply

    def reply(text):
        return NS(choices=[NS(message=NS(content=text))])

    assert is_json_reply(reply('{"results": []}'))
    assert not is_json_reply(reply(""))
    assert not is_json_reply(reply("I think the agent did well."))
    assert not is_json_reply(NS(choices=[]))


def test_dropped_turns_leave_a_note_with_the_authenticated_user_and_their_orders():
    """Round 13: once the find_user_id result was dropped, the agent asked an identified customer to authenticate again."""
    from shipwright.llm.context import OMITTED_NOTE, fit_messages

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "I want to return my lamp."},
        {"role": "assistant", "content": '```json\n{"name": "find_user_id_by_name_zip"}\n```'},
        {"role": "user", "content": 'Tool result for find_user_id_by_name_zip: "noah_ito_3850"'},
        {"role": "assistant", "content": '```json\n{"name": "get_user_details"}\n```'},
        {"role": "user", "content": 'Tool result for get_user_details: {"user_id":"noah_ito_3850","orders":["#W1","#W2"]}'},
        {"role": "assistant", "content": "x " * 300},
        {"role": "user", "content": "y " * 300},
        {"role": "assistant", "content": "Which order?"},
        {"role": "user", "content": "The first one."},
    ]
    fitted, info = fit_messages(messages, budget=400)
    assert info["dropped_messages"] > 0
    note = next(message["content"] for message in fitted if OMITTED_NOTE in message["content"])
    assert "authenticated as user id noah_ito_3850" in note and "#W1, #W2" in note
    unauthenticated = messages[:2] + messages[6:]
    fitted, _ = fit_messages(unauthenticated, budget=400)
    assert not any("authenticated as user id" in message["content"] for message in fitted)

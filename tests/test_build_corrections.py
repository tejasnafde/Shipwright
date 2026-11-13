"""Corrective examples: which turns are wrong, what a kept reply is, and that the budget cap holds."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_corrections as corrections  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "get_order_details", "description": "Get an order.",
          "parameters": {"type": "object", "required": ["order_id"], "properties": {"order_id": {"type": "string"}}}}}]
PROMPT = {"system": "policy", "tools": TOOLS}


def tau2_message(role, content=None, tool_name=None, arguments=None, call_id=None):
    message = {"role": role, "content": content}
    if role == "assistant" and tool_name:
        message["tool_calls"] = [{"id": call_id, "name": tool_name, "arguments": arguments}]
    if role == "tool":
        message["id"] = call_id
    return message


def looping_trace():
    """greeting, customer, authentication, then the same order lookup three times."""
    messages = [
        tau2_message("assistant", "Hi!"), tau2_message("user", "Where is order #W1? My email is a@b.c"),
        tau2_message("assistant", None, "find_user_id_by_email", {"email": "a@b.c"}, "c1"),
        tau2_message("tool", "u_1", call_id="c1"),
    ]
    for number in (2, 3, 4):
        messages += [tau2_message("assistant", None, "get_order_details", {"order_id": "#W1"}, f"c{number}"),
                     tau2_message("tool", '{"order_id": "#W1", "status": "pending", "items": []}', call_id=f"c{number}")]
    return {"task_id": "7", "simulation_id": "s7", "passed": False, "termination_reason": "max_steps", "messages": messages}


def test_the_repeated_lookups_are_found_and_capped_per_rule():
    wrong = corrections.find_wrong_turns(looping_trace(), per_rule_limit=1)
    assert [turn.rule for turn in wrong] == ["repeated_call"]  # the 3rd and 4th identical lookup are not extra rows
    assert len(corrections.find_wrong_turns(looping_trace(), per_rule_limit=5)) == 2  # the 1st lookup is fine


def test_a_reply_that_breaks_the_guards_again_is_not_kept():
    before = corrections.find_wrong_turns(looping_trace())[0].conversation_before
    assert corrections.reply_is_acceptable(before, "Your order is pending. Anything else I can do?")
    assert not corrections.reply_is_acceptable(before, '```json\n{"name": "get_order_details", "arguments": {"order_id": "#W1"}}\n```')
    assert not corrections.reply_is_acceptable(before, "   ")


def test_rows_train_only_the_teachers_reply_and_the_budget_cap_holds():
    asked = []

    def teacher(model, messages):
        asked.append(messages)
        return "Your order is pending. Anything else?", {"prompt_tokens": 1_000_000, "completion_tokens": 0}  # $0.30 a call

    traces = [looping_trace(), looping_trace()]
    rows, stats = corrections.build_corrections(traces, lambda trace: PROMPT, teacher, "vertex/deepseek-ai/deepseek-v4.1-flash-maas",
                                                max_cost_usd=0.5, per_rule_limit=5)
    assert all(row["train_last_only"] and row["source"] == "correction" for row in rows)
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "Your order is pending. Anything else?"}
    assert rows[0]["messages"][0]["role"] == "system" and rows[0]["messages"][1]["role"] == "user"
    assert len(asked) == 2 and stats.stopped_on_budget  # the third and fourth wrong turns were not asked: $0.60 spent
    assert stats.cost_usd < 0.7


def test_passed_and_infrastructure_error_runs_are_skipped():
    passed = {**looping_trace(), "passed": True}
    infra = {**looping_trace(), "termination_reason": "infrastructure_error"}
    rows, stats = corrections.build_corrections([passed, infra], lambda trace: PROMPT, lambda model, messages: ("x", {}), "m", 1.0)
    assert rows == [] and stats.wrong_turns_found == {}


def test_a_tool_call_with_a_sentence_before_it_becomes_exactly_one_block():
    mixed = 'Let me look that up properly.\n\n```json\n{"name": "find_user_id_by_name_zip", "arguments": {"zip": "19122"}}\n```'
    assert corrections.canonical_reply(mixed) == (
        '```json\n{"name": "find_user_id_by_name_zip", "arguments": {"zip": "19122"}}\n```')
    assert corrections.canonical_reply("  Anything else I can do?  ") == "Anything else I can do?"


def guarded_trace():
    """A run with the guards on: the repeated lookup was rejected, so the history holds it once, and the rejected reply
    sits in the turn's guard_rejections."""
    messages = looping_trace()["messages"][:6] + [tau2_message("assistant", "Your order #W1 is pending.")]
    rejected = '```json\n{"name": "get_order_details", "arguments": {"order_id": "#W1"}}\n```'
    agent_turns = [{"message_index": 6, "guard_rejections": [{"rule": "repeated_call", "raw_text": rejected}]}]
    return {"task_id": "8", "simulation_id": "s8", "passed": False, "termination_reason": "user_stop",
            "messages": messages, "agent_turns": agent_turns}


def test_a_reply_the_guards_rejected_is_a_wrong_turn_at_the_state_the_student_was_in():
    wrong = corrections.find_wrong_turns(guarded_trace())
    assert [turn.rule for turn in wrong] == ["repeated_call"]
    assert len(wrong[0].conversation_before) == 6  # up to the turn, not including the reply the guards let through
    assert not corrections.reply_is_acceptable(
        wrong[0].conversation_before, '```json\n{"name": "get_order_details", "arguments": {"order_id": "#W1"}}\n```')


def test_a_long_conversation_is_trimmed_to_the_budget_not_dropped():
    trace = looping_trace()
    filler = "x " * 6000  # three of these pass 8192 tokens
    for _ in range(3):  # a long middle of the conversation, as in a looping run
        trace["messages"].insert(2, tau2_message("user", filler))
        trace["messages"].insert(2, tau2_message("assistant", "Let me check."))
    wrong = corrections.find_wrong_turns(trace)[0]
    row = corrections.correction_row(wrong, PROMPT, "Your order is pending.", "teacher")
    assert corrections.count_messages(row["messages"]) <= corrections.CONTEXT_LIMIT
    assert row["messages"][-1] == {"role": "assistant", "content": "Your order is pending."}


def test_a_failed_call_is_skipped_and_saved_answers_are_not_paid_for_again():
    saved = {}
    calls = []

    def flaky_teacher(model, messages):
        calls.append(messages)
        if len(calls) == 2:
            raise RuntimeError("429")
        return "Your order is pending.", {"prompt_tokens": 1000, "completion_tokens": 10}

    def save(key, answer):
        saved[key] = answer

    traces = [looping_trace(), {**looping_trace(), "simulation_id": "s9"}]
    rows, stats = corrections.build_corrections(traces, lambda trace: PROMPT, flaky_teacher, "m", 1.0, 5,
                                                saved_answers={}, save_answer=save)
    assert stats.teacher_errors == 1 and len(rows) == 3 and len(saved) == 3

    calls.clear()
    rows_again, stats_again = corrections.build_corrections(traces, lambda trace: PROMPT, flaky_teacher, "m", 1.0, 5,
                                                            saved_answers=dict(saved), save_answer=save)
    assert stats_again.answers_reused == 3 and len(calls) == 1 and len(rows_again) == 4  # only the failed turn is asked

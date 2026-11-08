import json
import sys
from pathlib import Path

import pytest

from shipwright import prompted_tools as pt
from shipwright import sft_data as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_job  # noqa: E402
import train_lora  # noqa: E402

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_details", "description": "Get the status and details of an order.",
        "parameters": {"type": "object", "required": ["order_id"], "properties": {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."}}}}},
]
PROMPT = {"system": "<instructions>Be helpful.</instructions>", "tools": TOOLS}
MODEL = "openai/CohereLabs/tiny-aya-global:cohere"


def tau2_trace(order_status: str = "pending", reward: float = 1.0) -> dict:
    """A tau2-style trace (the JSON form traces.jsonl keeps)."""
    msgs = [
        {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": None},
        {"role": "user", "content": "Where is my order #W1?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "name": "get_order_details", "arguments": {"order_id": "#W1"}, "requestor": "assistant"}]},
        {"role": "tool", "id": "c1", "content": json.dumps({"status": order_status}, indent=2),
         "requestor": "assistant", "error": False},
        {"role": "assistant", "content": f"Your order is {order_status}.", "tool_calls": None},
        {"role": "user", "content": "Thanks ###STOP###"},
    ]
    return {"task_id": "7", "simulation_id": "s", "reward": reward, "passed": reward == 1.0,
            "messages": msgs, "agent_turns": []}


def test_rendered_example_equals_student_prompt_at_every_turn():
    trace = tau2_trace()
    [rec] = sd.trace_records(trace, PROMPT, "teacher")
    msgs = rec["messages"]
    assert [m["role"] for m in msgs] == ["system", "assistant", "user", "assistant", "user", "assistant"]
    # What Tiny Aya is sent before each assistant turn (llm_patch -> prepare_request).
    history = sd.trace_to_openai(trace["messages"])
    full = [{"role": "system", "content": PROMPT["system"]}, *history]
    for idx in (3, 5):  # assistant turns after a user turn and after a tool result
        # Here the converted index equals the raw index: no user turns are merged.
        req, _ = pt.prepare_request({"model": MODEL, "messages": full[:idx], "tools": TOOLS})
        assert req["messages"] == msgs[:idx]
    assert pt.parse_tool_call(msgs[3]["content"]) == ("get_order_details", {"order_id": "#W1"})
    assert msgs[4]["content"] == 'Tool result for get_order_details: {"status":"pending"}'
    assert msgs[0]["content"] == f"{PROMPT['system']}\n\n{pt.render_tools_prompt(TOOLS)}"


def test_prepare_request_records_prompt_for_training():
    req, info = pt.prepare_request({"model": MODEL, "tools": TOOLS, "messages": [
        {"role": "system", "content": PROMPT["system"]}, {"role": "user", "content": "hi"}]})
    assert pt.recorded_prompts()[info["prompt_sha"]] == PROMPT


def test_force_prompted_teacher_gets_student_view_and_own_reply_budget():
    teacher = "openai/some/teacher-model"
    assert not pt.is_prompted_model(teacher)
    pt.force_prompted(teacher, 4096)
    req, _ = pt.prepare_request({"model": teacher, "tools": TOOLS, "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]})
    student, _ = pt.prepare_request({"model": MODEL, "tools": TOOLS, "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]})
    assert pt.is_prompted_model(teacher) and "tools" not in req
    assert req["messages"] == student["messages"]
    assert req["max_tokens"] == 4096 and student["max_tokens"] == 700


def test_filter_drops_over_8192_and_long_replies():
    recs = [
        {"tokens": 8192, "reply_tokens_max": 700},
        {"tokens": 8193, "reply_tokens_max": 10},
        {"tokens": 100, "reply_tokens_max": 701},
    ]
    kept, dropped = sd.filter_records(recs)
    assert kept == [recs[0]]
    assert dropped == {"over_max_tokens": 1, "reply_over_max_tokens": 1}


def test_dedupe_keeps_first_and_ignores_metadata():
    a = sd.trace_records(tau2_trace("pending"), PROMPT, "teacher")
    b = sd.trace_records(tau2_trace("pending"), PROMPT, "replay")
    c = sd.trace_records(tau2_trace("delivered"), PROMPT, "teacher")
    out = sd.dedupe(a + b + c)
    assert [r["source"] for r in out] == ["teacher", "teacher"]
    assert len(out) == 2


def _long_conv(n_turns: int, result_chars: int) -> list[dict]:
    conv = [{"role": "system", "content": "S" * 100}, {"role": "assistant", "content": "Hi!"}]
    for i in range(n_turns):
        conv.append({"role": "user", "content": f"Tool result for f: {'x' * result_chars} {i}"})
        conv.append({"role": "assistant", "content": f"reply {i}"})
    return conv


def test_window_mode_fits_budget_and_trains_each_turn_once():
    conv = _long_conv(n_turns=12, result_chars=3000)
    budget = 3000
    windows = sd.window_examples(conv, budget)
    assert len(windows) > 1
    targets = []
    for w in windows:
        assert w[0]["role"] == "system" and w[-1]["role"] == "assistant"
        assert sd.count_messages(w[:-1]) <= budget
        targets += [m["content"] for m in w if m["role"] == "assistant"]
    replies = [m["content"] for m in conv if m["role"] == "assistant"]
    assert set(targets) == set(replies)
    assert len(targets) == len(replies)


def test_short_conversation_is_one_example_and_drop_mode():
    conv = _long_conv(n_turns=2, result_chars=10)
    assert sd.conversation_examples(conv) == [conv]
    long = _long_conv(n_turns=12, result_chars=4000)
    assert sd.conversation_examples(long, "drop") == []
    assert len(sd.conversation_examples(long, "window")) > 1


# ---- train_lora pure helpers ----


def stable_render(msgs, gen):
    text = "<BOS>" + "".join(f"<{m['role']}>{m['content']}<end>" for m in msgs)
    return text + ("<assistant>" if gen else "")


def test_split_example_and_spans_and_labels():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "A"}]
    system, events = train_lora.split_example(msgs)
    assert system == "S" and events == [("observation", "q"), ("text", "A")]
    with pytest.raises(ValueError):
        train_lora.split_example([{"role": "tool", "content": "x"}])

    spans = train_lora.target_char_spans(stable_render, msgs, [True, True, False])
    full = stable_render(msgs, False)
    assert [full[s:e] for s, e in spans] == ["A<end>"]
    ids = [1, 2, 3]
    offsets = [(0, 5), (len(full) - 6, len(full) - 5), (len(full) - 5, len(full))]
    assert train_lora.labels_from_offsets(ids, offsets, spans) == [-100, 2, 3]

    def unstable(m, gen):  # generation prompt differs from the rendered assistant header
        return stable_render(m, False) + ("<bot>" if gen else "")

    with pytest.raises(ValueError):
        train_lora.target_char_spans(unstable, msgs, [True, True, False])


# ---- train_job ----


def test_train_job_plans_without_launch(capsys):
    assert train_job.main(["--timeout", "2h"]) == 0
    out = capsys.readouterr().out
    assert "max cost:  $1.60" in out and "Nothing was launched" in out
    assert train_job.estimate_cost("a10g-small", "90m") == (1.00, 1.5)
    with pytest.raises(SystemExit):
        train_job.estimate_cost("h200", "1h")
    assert train_job.estimate_cost("h200", "1h", 5.0) == (5.0, 5.0)

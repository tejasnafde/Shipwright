"""OTel GenAI spans -> trace records (the traces.jsonl shape), so the loop can take
standard OTel traces as input.

Input: span dicts as ``shipwright.observability.otel_export.JsonlFileSpanExporter`` writes them (name,
trace_id, span_id, parent_span_id, start_time_ns, attributes, status). One OTel trace
becomes one trace record. Spans from our own harness carry ``shipwright.*`` attributes
and convert without loss for the classifier. For other GenAI spans the converter uses
the semconv keys: ``chat`` spans are agent turns (text from ``gen_ai.output.messages``,
``gen_ai.completion`` or ``output.value``), ``execute_tool`` spans are tool calls
(``gen_ai.tool.name``, ``gen_ai.tool.call.arguments``, ``gen_ai.tool.call.result``,
error status). User messages exist only in our own spans.

The records go through ``traces.build_trace``, so derived fields (errors, malformed and
unknown flags) are computed the same way as for a live run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from shipwright.heal.traces import build_trace

P = "shipwright."
JSON_KEYS = {
    P + "reward_info", P + "tool_calls", P + "raw_tool_calls", P + "parsed_tool_calls",
    P + "usage", "gen_ai.output.messages", "gen_ai.tool.call.arguments",
}


def _attrs(span: dict) -> dict:
    out = {}
    for key, value in (span.get("attributes") or {}).items():
        if key in JSON_KEYS and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        out[key] = value
    return out


def _op(span: dict, a: dict) -> str:
    op = a.get("gen_ai.operation.name")
    if op:
        return op
    name = span.get("name") or ""
    kind = a.get("openinference.span.kind")
    if name == "user_message":
        return "user_message"
    if name.startswith(("chat", "text_completion", "generate_content")) or kind == "LLM":
        return "chat"
    if name.startswith("execute_tool") or kind == "TOOL":
        return "execute_tool"
    return ""


def _iso(ns: Any) -> str | None:
    return datetime.fromtimestamp(ns / 1e9).isoformat() if isinstance(ns, int) else None


def _text_of(a: dict) -> str | None:
    if P + "content" in a or P + "raw_output" in a:
        return a.get(P + "content")  # our spans: a missing content attribute means None
    for msg in a.get("gen_ai.output.messages") or []:
        texts = [p.get("content") for p in msg.get("parts") or [] if p.get("type") == "text"]
        if texts:
            return "".join(t or "" for t in texts)
    return a.get("gen_ai.completion") or a.get("output.value")


def _calls_of(a: dict) -> list[dict] | None:
    if P + "tool_calls" in a:
        return a[P + "tool_calls"]
    calls = [
        {"id": p.get("id"), "name": p.get("name"), "arguments": p.get("arguments") or {}}
        for msg in a.get("gen_ai.output.messages") or []
        for p in msg.get("parts") or [] if p.get("type") == "tool_call"
    ]
    return calls or None


def _assistant(span: dict, a: dict, tools: list[tuple[dict, dict]]) -> dict:
    calls = _calls_of(a)
    if calls is None and tools:
        calls = [
            {"id": t.get("gen_ai.tool.call.id") or ts["span_id"], "name": t.get("gen_ai.tool.name"),
             "arguments": t.get("gen_ai.tool.call.arguments") or {}}
            for ts, t in tools
        ]
    content = _text_of(a)
    raw_output = a.get(P + "raw_output", content)
    raw_calls = a.get(P + "raw_tool_calls") or [
        {"id": c.get("id"), "name": c.get("name"), "arguments": json.dumps(c.get("arguments"))}
        for c in calls or []
    ]
    shipwright = {
        "parse": a.get(P + "tool_call_parse"),
        "raw_text": raw_output,
        "prompt_tokens_est": a.get(P + "prompt_tokens_est"),
        "prompt_sha": a.get(P + "prompt_sha"),
        "dropped_messages": a.get(P + "context_dropped_messages") or 0,
        "truncated_chars": a.get(P + "context_truncated_chars") or 0,
    }
    reasons = a.get("gen_ai.response.finish_reasons") or [None]
    usage = a.get(P + "usage")
    if usage is None and ("gen_ai.usage.input_tokens" in a or "gen_ai.usage.output_tokens" in a):
        usage = {"prompt_tokens": a.get("gen_ai.usage.input_tokens"),
                 "completion_tokens": a.get("gen_ai.usage.output_tokens")}
    start, end = span.get("start_time_ns"), span.get("end_time_ns")
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": calls,
        "timestamp": _iso(end),
        "generation_time_seconds": (end - start) / 1e9 if isinstance(start, int) and isinstance(end, int) else None,
        "usage": usage,
        "raw_data": {"choices": [{
            "message": {
                "content": raw_output,
                "reasoning_content": a.get(P + "reasoning"),
                "tool_calls": [
                    {"id": c.get("id"), "type": "function",
                     "function": {"name": c.get("name"), "arguments": c.get("arguments")}}
                    for c in raw_calls
                ],
                "provider_specific_fields": {"shipwright": shipwright} if a.get(P + "tool_call_parse") or P + "raw_output" in a else {},
            },
            "finish_reason": reasons[0],
        }]},
    }


def _tool_message(span: dict, a: dict) -> dict:
    error = a.get(P + "tool_error")
    if error is None:
        error = span.get("status") == "ERROR"
    return {
        "id": a.get("gen_ai.tool.call.id") or span["span_id"],
        "role": "tool",
        "content": a.get("gen_ai.tool.call.result", a.get("output.value")),
        "requestor": a.get(P + "requestor", "assistant"),
        "error": bool(error),
        "timestamp": _iso(span.get("end_time_ns")),
    }


def _sort_key(span: dict, a: dict) -> tuple:
    idx = a.get(P + "message_index")
    return (span.get("start_time_ns") or 0, idx if isinstance(idx, int) else 0)


def spans_to_traces(spans: Iterable[dict]) -> list[dict]:
    """Group spans by trace id and build one trace record per OTel trace."""
    by_trace: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for s in spans:
        if s.get("trace_id") not in by_trace:
            order.append(s.get("trace_id"))
        by_trace[s.get("trace_id")].append(s)
    return [r for r in (_one_trace(by_trace[t]) for t in order) if r is not None]


def _one_trace(spans: list[dict]) -> dict | None:
    items = [(s, _attrs(s)) for s in spans]
    roots = [(s, a) for s, a in items if not s.get("parent_span_id")]
    root_s, root = next(((s, a) for s, a in roots if _op(s, a) == "invoke_agent"), roots[0] if roots else ({}, {}))
    kept = [(s, a) for s, a in items if _op(s, a) in ("chat", "execute_tool", "user_message")]
    if not any(_op(s, a) == "chat" for s, a in kept):
        return None

    have_index = all(isinstance(a.get(P + "message_index"), int) for s, a in kept if _op(s, a) != "execute_tool")
    if have_index:
        tool_idx = [a.get(P + "message_index") for s, a in kept if _op(s, a) == "execute_tool"]
        have_index = all(isinstance(i, int) or i is None for i in tool_idx)

    # Tool spans belong to the chat span that shares their parent (our agent_turn), or else
    # to the closest chat span that started before them.
    chats = sorted(((s, a) for s, a in kept if _op(s, a) == "chat"), key=lambda x: _sort_key(*x))
    tools_for: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for s, a in kept:
        if _op(s, a) != "execute_tool":
            continue
        owner = next((c for c, _ in chats if c.get("parent_span_id") == s.get("parent_span_id")
                      and c.get("parent_span_id") != root_s.get("span_id")), None)
        if owner is None:
            before = [c for c, _ in chats if (c.get("start_time_ns") or 0) <= (s.get("start_time_ns") or 0)]
            owner = before[-1] if before else chats[0][0]
        tools_for[owner["span_id"]].append((s, a))

    entries: list[tuple[tuple, dict]] = []
    seq = 0
    for s, a in sorted(kept, key=lambda x: _sort_key(*x)):
        op = _op(s, a)
        if op == "execute_tool":
            continue
        if op == "user_message":
            msg = {"role": "user", "content": a.get(P + "content", a.get("output.value")),
                   "timestamp": _iso(s.get("start_time_ns"))}
        else:
            msg = _assistant(s, a, tools_for.get(s["span_id"], []))
        if a.get(P + "turn_idx") is not None:
            msg["turn_idx"] = a[P + "turn_idx"]
        key = (a[P + "message_index"],) if have_index else (seq,)
        entries.append((key, msg))
        seq += 1
        tools = sorted(tools_for.get(s["span_id"], []), key=lambda x: _sort_key(*x))
        for k, (ts, ta) in enumerate(tools):
            tmsg = _tool_message(ts, ta)
            if ta.get(P + "turn_idx") is not None:
                tmsg["turn_idx"] = ta[P + "turn_idx"]
            if not ta.get(P + "executed", True):
                continue  # the call never ran (e.g. the run stopped first)
            idx = ta.get(P + "message_index")
            tkey = (idx,) if have_index and isinstance(idx, int) else (key[0] + (k + 1) * 1e-3,) if have_index else (seq,)
            entries.append((tkey, tmsg))
            seq += 1
    messages = [m for _, m in sorted(entries, key=lambda e: e[0])]

    tool_names = {
        a.get("gen_ai.tool.name") for s, a in kept
        if _op(s, a) == "execute_tool" and not a.get(P + "unknown_tool")
    }
    for m in messages:
        if m["role"] == "assistant" and not any(P + "unknown_tool" in a for _, a in kept):
            tool_names |= {c.get("name") for c in m.get("tool_calls") or []}

    reward = root.get(P + "reward")
    reward_info = root.get(P + "reward_info") or ({"reward": reward} if reward is not None else {})
    task_id = root.get(P + "task_id")
    sim = {
        "id": root.get(P + "simulation_id") or root.get("gen_ai.conversation.id") or root_s.get("trace_id"),
        "task_id": task_id,
        "trial": root.get(P + "trial"),
        "seed": root.get(P + "seed"),
        "reward_info": reward_info,
        "termination_reason": root.get(P + "termination_reason"),
        "info": {"error": root.get(P + "error"), "error_type": root.get(P + "error_type")},
        "duration": root.get(P + "duration"),
        "agent_cost": root.get(P + "agent_cost"),
        "user_cost": root.get(P + "user_cost"),
        "messages": messages,
    }
    return build_trace(sim, tool_names)


def read_spans(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

"""Turn tau2 simulation results into per-run traces and a summary.

Works on plain dicts (the JSON form of tau2 ``SimulationRun``), so it is testable
without tau2 or network access.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from shipwright.costs import cached_tokens, hit_rate

# Key that llm_patch puts into tool-call arguments when the model's arguments
# are not a valid JSON object. The original string is kept under this key.
MALFORMED_KEY = "__malformed_arguments__"

PASS_THRESHOLD = 1.0 - 1e-6
INFRA_ERROR = "infrastructure_error"


def _raw_message(msg: dict) -> dict:
    """The assistant message exactly as the provider returned it (from raw_data)."""
    raw = msg.get("raw_data") or {}
    choices = raw.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def _original_arguments(raw_args: str | None) -> str | None:
    """Undo llm_patch's wrapping so the trace shows what the model really wrote."""
    try:
        parsed = json.loads(raw_args) if raw_args else None
    except (TypeError, ValueError):
        return raw_args
    if isinstance(parsed, dict) and set(parsed) == {MALFORMED_KEY}:
        return parsed[MALFORMED_KEY]
    return raw_args


def _finish_reason(msg: dict) -> str | None:
    raw = msg.get("raw_data") or {}
    choices = raw.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def _usage(msg: dict) -> dict | None:
    """tau2's usage (prompt and completion tokens) plus cached prompt tokens from raw_data, if reported."""
    usage = msg.get("usage")
    cached = cached_tokens((msg.get("raw_data") or {}).get("usage"))
    if usage is not None and cached is not None:
        usage = {**usage, "cached_prompt_tokens": cached}
    return usage


def _looks_like_text_tool_call(text: str | None) -> bool:
    """Heuristic: the model wrote a tool call as text instead of a native call."""
    if not text:
        return False
    t = text.strip()
    return ('"name"' in t and ('"arguments"' in t or '"parameters"' in t)) or "<tool_call>" in t


def build_trace(sim: dict, tool_names: set[str]) -> dict:
    """One trace record for one simulation run."""
    messages = sim.get("messages") or []
    tool_results = {
        m.get("id"): m for m in messages if m.get("role") == "tool"
    }
    reward_info = sim.get("reward_info") or {}
    reward = reward_info.get("reward")

    turns = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        raw = _raw_message(msg)
        raw_tool_calls = [
            {
                "id": tc.get("id"),
                "name": (tc.get("function") or {}).get("name"),
                "arguments": _original_arguments((tc.get("function") or {}).get("arguments")),
            }
            for tc in (raw.get("tool_calls") or [])
        ]
        parsed, results, errors = [], [], []
        for tc in msg.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            malformed = isinstance(args, dict) and MALFORMED_KEY in args
            unknown = tc.get("name") not in tool_names
            parsed.append(
                {
                    "id": tc.get("id"),
                    "name": tc.get("name"),
                    "arguments": args,
                    "malformed_arguments": malformed,
                    "unknown_tool": unknown,
                }
            )
            if malformed:
                errors.append(f"malformed_arguments:{tc.get('name')}")
            if unknown:
                errors.append(f"unknown_tool:{tc.get('name')}")
            res = tool_results.get(tc.get("id"))
            if res is not None:
                results.append(
                    {
                        "id": res.get("id"),
                        "name": tc.get("name"),
                        "content": res.get("content"),
                        "error": bool(res.get("error")),
                    }
                )
                if res.get("error"):
                    errors.append(f"tool_error:{tc.get('name')}")
        text = msg.get("content")
        prompted = (raw.get("provider_specific_fields") or {}).get("shipwright") or {}
        parse_status = prompted.get("parse")  # tool_call | parse_failed | text | None (native)
        if parse_status == "parse_failed":
            errors.append("tool_call_parse_failed")
        text_tool_call = not parsed and parse_status != "parse_failed" and _looks_like_text_tool_call(text)
        if text_tool_call:
            errors.append("tool_call_in_text")
        turns.append(
            {
                "message_index": idx,
                "turn_idx": msg.get("turn_idx"),
                "raw_output": prompted["raw_text"] if "raw_text" in prompted else raw.get("content", text),
                "tool_call_parse": parse_status,
                "prompt_tokens_est": prompted.get("prompt_tokens_est"),
                "prompt_sha": prompted.get("prompt_sha"),
                "context_dropped_messages": prompted.get("dropped_messages", 0),
                "context_truncated_chars": prompted.get("truncated_chars", 0),
                "context_trimmed": bool(prompted.get("dropped_messages") or prompted.get("truncated_chars")),
                "reasoning": raw.get("reasoning_content") or raw.get("reasoning"),
                "raw_tool_calls": raw_tool_calls,
                "parsed_tool_calls": parsed,
                "tool_results": results,
                "errors": errors,
                "text_tool_call": text_tool_call,
                "finish_reason": _finish_reason(msg),
                "usage": _usage(msg),
            }
        )

    return {
        "simulation_id": sim.get("id"),
        "task_id": sim.get("task_id"),
        "trial": sim.get("trial"),
        "seed": sim.get("seed"),
        "reward": reward,
        "passed": reward is not None and reward >= PASS_THRESHOLD,
        "termination_reason": sim.get("termination_reason"),
        "error": (sim.get("info") or {}).get("error"),
        "error_type": (sim.get("info") or {}).get("error_type"),
        "reward_info": reward_info,
        "duration": sim.get("duration"),
        "agent_cost": sim.get("agent_cost"),
        "user_cost": sim.get("user_cost"),
        "num_messages": len(messages),
        "agent_turns": turns,
        # Full message list; raw_data is dropped here because agent_turns keeps it.
        "messages": [{**{k: v for k, v in m.items() if k != "raw_data"},
                      **({"usage": _usage(m)} if m.get("usage") is not None else {})} for m in messages],
    }


def _add_usage(total: Counter, usage: dict | None) -> None:
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens"):
        if isinstance(usage.get(key), (int, float)):
            total[key] += usage[key]


def summarize(traces: list[dict]) -> dict:
    """Aggregate numbers over all traces."""
    n = len(traces)
    agent_tokens: Counter = Counter()
    user_tokens: Counter = Counter()
    tool_calls = malformed = unknown = tool_errors = text_calls = parse_failed = trimmed = 0
    agent_turns = 0
    unknown_names: Counter = Counter()
    for t in traces:
        for m in t["messages"]:
            if m.get("role") == "user":
                _add_usage(user_tokens, m.get("usage"))
        for turn in t["agent_turns"]:
            agent_turns += 1
            _add_usage(agent_tokens, turn.get("usage"))
            text_calls += int(turn["text_tool_call"])
            parse_failed += int(turn.get("tool_call_parse") == "parse_failed")
            trimmed += int(bool(turn.get("context_trimmed")))
            for tc in turn["parsed_tool_calls"]:
                tool_calls += 1
                malformed += int(tc["malformed_arguments"])
                if tc["unknown_tool"]:
                    unknown += 1
                    unknown_names[tc["name"]] += 1
            tool_errors += sum(1 for r in turn["tool_results"] if r["error"])

    def _sum(key: str) -> float | None:
        vals = [t[key] for t in traces if isinstance(t.get(key), (int, float))]
        return sum(vals) if vals else None

    passed = sum(1 for t in traces if t["passed"])
    infra = [t for t in traces if t["termination_reason"] == INFRA_ERROR]
    clean = n - len(infra)
    infra_messages = Counter((t.get("error") or "")[:150] for t in infra)
    return {
        "num_runs": n,
        "num_tasks": len({t["task_id"] for t in traces}),
        "num_passed": passed,
        "pass_rate": passed / n if n else None,
        "num_infra_errors": len(infra),
        "pass_rate_excluding_infra_errors": (
            sum(1 for t in traces if t["passed"] and t["termination_reason"] != INFRA_ERROR) / clean
            if clean else None
        ),
        "infra_error_messages": dict(infra_messages.most_common()),
        "avg_reward": (
            sum(t["reward"] or 0.0 for t in traces) / n if n else None
        ),
        "tool_calls": tool_calls,
        "malformed_tool_calls": malformed,
        "unknown_tool_calls": unknown,
        "unknown_tool_names": dict(unknown_names),
        "tool_errors": tool_errors,
        "tool_calls_in_text": text_calls,
        "tool_call_parse_failures": parse_failed,
        "context_trimmed_turns": trimmed,
        "avg_agent_turns": agent_turns / n if n else None,
        "avg_messages": sum(t["num_messages"] for t in traces) / n if n else None,
        "termination_reasons": dict(Counter(t["termination_reason"] for t in traces)),
        "agent_tokens": dict(agent_tokens),
        "user_tokens": dict(user_tokens),
        "cached_prompt_tokens": {"agent": agent_tokens["cached_prompt_tokens"],
                                 "user": user_tokens["cached_prompt_tokens"]},
        "cache_hit_rate": {"agent": hit_rate(agent_tokens["prompt_tokens"], agent_tokens["cached_prompt_tokens"]),
                           "user": hit_rate(user_tokens["prompt_tokens"], user_tokens["cached_prompt_tokens"])},
        "agent_cost_litellm": _sum("agent_cost"),
        "user_cost_litellm": _sum("user_cost"),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

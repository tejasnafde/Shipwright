"""Turn tau2 simulation results into per-run traces and a summary.

Works on plain dicts (the JSON form of tau2 ``SimulationRun``), so it is testable
without tau2 or network access.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from shipwright.llm.costs import cached_tokens, hit_rate

# Key that llm_patch puts into tool-call arguments when the model's arguments
# are not a valid JSON object. The original string is kept under this key.
MALFORMED_KEY = "__malformed_arguments__"

PASS_THRESHOLD = 1.0 - 1e-6
INFRA_ERROR = "infrastructure_error"


def _raw_message(message: dict) -> dict:
    """The assistant message exactly as the provider returned it (from raw_data)."""
    raw_data = message.get("raw_data") or {}
    choices = raw_data.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def _original_arguments(raw_arguments: str | None) -> str | None:
    """Undo llm_patch's wrapping so the trace shows what the model really wrote."""
    try:
        parsed_arguments = json.loads(raw_arguments) if raw_arguments else None
    except (TypeError, ValueError):
        return raw_arguments
    if isinstance(parsed_arguments, dict) and set(parsed_arguments) == {MALFORMED_KEY}:
        return parsed_arguments[MALFORMED_KEY]
    return raw_arguments


def _finish_reason(message: dict) -> str | None:
    raw_data = message.get("raw_data") or {}
    choices = raw_data.get("choices") or []
    if choices:
        return choices[0].get("finish_reason")
    return None


def _usage(message: dict) -> dict | None:
    """tau2's usage (prompt and completion tokens) plus cached prompt tokens from raw_data, if reported."""
    usage = message.get("usage")
    cached_prompt_tokens = cached_tokens((message.get("raw_data") or {}).get("usage"))
    if usage is not None and cached_prompt_tokens is not None:
        usage = {**usage, "cached_prompt_tokens": cached_prompt_tokens}
    return usage


def _looks_like_text_tool_call(text: str | None) -> bool:
    """Heuristic: the model wrote a tool call as text instead of a native call."""
    if not text:
        return False
    stripped_text = text.strip()
    has_name_and_arguments = '"name"' in stripped_text and (
        '"arguments"' in stripped_text or '"parameters"' in stripped_text
    )
    return has_name_and_arguments or "<tool_call>" in stripped_text


def _build_agent_turn(
    message_index: int,
    message: dict,
    tool_result_by_call_id: dict,
    tool_names: set[str],
) -> dict:
    """One agent turn: what the model wrote, the tool calls parsed from it, and their results."""
    raw_message = _raw_message(message)
    raw_tool_calls = []
    for raw_call in raw_message.get("tool_calls") or []:
        raw_function = raw_call.get("function") or {}
        raw_tool_calls.append({
            "id": raw_call.get("id"),
            "name": raw_function.get("name"),
            "arguments": _original_arguments(raw_function.get("arguments")),
        })

    parsed_tool_calls = []
    tool_results = []
    errors = []
    for tool_call in message.get("tool_calls") or []:
        arguments = tool_call.get("arguments") or {}
        is_malformed = isinstance(arguments, dict) and MALFORMED_KEY in arguments
        is_unknown_tool = tool_call.get("name") not in tool_names
        parsed_tool_calls.append({
            "id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "arguments": arguments,
            "malformed_arguments": is_malformed,
            "unknown_tool": is_unknown_tool,
        })
        if is_malformed:
            errors.append(f"malformed_arguments:{tool_call.get('name')}")
        if is_unknown_tool:
            errors.append(f"unknown_tool:{tool_call.get('name')}")

        tool_result = tool_result_by_call_id.get(tool_call.get("id"))
        if tool_result is not None:
            tool_results.append({
                "id": tool_result.get("id"),
                "name": tool_call.get("name"),
                "content": tool_result.get("content"),
                "error": bool(tool_result.get("error")),
            })
            if tool_result.get("error"):
                errors.append(f"tool_error:{tool_call.get('name')}")

    text = message.get("content")
    shipwright_fields = (raw_message.get("provider_specific_fields") or {}).get("shipwright") or {}
    parse_status = shipwright_fields.get("parse")  # tool_call | parse_failed | text | None (native)
    if parse_status == "parse_failed":
        errors.append("tool_call_parse_failed")
    is_text_tool_call = (
        not parsed_tool_calls and parse_status != "parse_failed" and _looks_like_text_tool_call(text)
    )
    if is_text_tool_call:
        errors.append("tool_call_in_text")

    if "raw_text" in shipwright_fields:
        raw_output = shipwright_fields["raw_text"]
    else:
        raw_output = raw_message.get("content", text)
    was_context_trimmed = bool(
        shipwright_fields.get("dropped_messages") or shipwright_fields.get("truncated_chars")
    )
    return {
        "message_index": message_index,
        "turn_idx": message.get("turn_idx"),
        "raw_output": raw_output,
        "tool_call_parse": parse_status,
        "prompt_tokens_est": shipwright_fields.get("prompt_tokens_est"),
        "prompt_sha": shipwright_fields.get("prompt_sha"),
        "context_dropped_messages": shipwright_fields.get("dropped_messages", 0),
        "context_truncated_chars": shipwright_fields.get("truncated_chars", 0),
        "context_trimmed": was_context_trimmed,
        "reasoning": raw_message.get("reasoning_content") or raw_message.get("reasoning"),
        "raw_tool_calls": raw_tool_calls,
        "parsed_tool_calls": parsed_tool_calls,
        "tool_results": tool_results,
        "errors": errors,
        "text_tool_call": is_text_tool_call,
        "guard_rejections": shipwright_fields.get("guard_rejections") or [],
        "guard_allowed_after_retries": shipwright_fields.get("guard_allowed_after_retries"),
        "finish_reason": _finish_reason(message),
        "usage": _usage(message),
    }


def _message_without_raw_data(message: dict) -> dict:
    """A message for the trace's full message list. raw_data is dropped: agent_turns keeps it."""
    trimmed_message = {key: value for key, value in message.items() if key != "raw_data"}
    if message.get("usage") is not None:
        trimmed_message["usage"] = _usage(message)
    return trimmed_message


def build_trace(simulation: dict, tool_names: set[str]) -> dict:
    """One trace record for one simulation run."""
    messages = simulation.get("messages") or []
    tool_result_by_call_id = {
        message.get("id"): message for message in messages if message.get("role") == "tool"
    }
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")

    agent_turns = [
        _build_agent_turn(message_index, message, tool_result_by_call_id, tool_names)
        for message_index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    simulation_info = simulation.get("info") or {}
    return {
        "simulation_id": simulation.get("id"),
        "task_id": simulation.get("task_id"),
        "trial": simulation.get("trial"),
        "seed": simulation.get("seed"),
        "reward": reward,
        "passed": reward is not None and reward >= PASS_THRESHOLD,
        "termination_reason": simulation.get("termination_reason"),
        "error": simulation_info.get("error"),
        "error_type": simulation_info.get("error_type"),
        "reward_info": reward_info,
        "duration": simulation.get("duration"),
        "agent_cost": simulation.get("agent_cost"),
        "user_cost": simulation.get("user_cost"),
        "num_messages": len(messages),
        "agent_turns": agent_turns,
        "messages": [_message_without_raw_data(message) for message in messages],
    }


def _add_usage(token_totals: Counter, usage: dict | None) -> None:
    if not usage:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens"):
        if isinstance(usage.get(key), (int, float)):
            token_totals[key] += usage[key]


def _average(total: float, count: int) -> float | None:
    """total / count, or None when there is nothing to average."""
    if count:
        return total / count
    return None


def infra_error_kind(error_text: str | None) -> str:
    """A short name for why a run ended as an infra error, from tau2's error text."""
    text = error_text or ""
    if "Timeout" in text or "timed out" in text:
        return "timeout"
    if "400" in text or "BadRequest" in text:
        return "bad_request_400"
    if "429" in text or "RateLimit" in text:
        return "rate_limit_429"
    if "UserMessage must have either content" in text:
        return "empty_user_reply"
    if "max-cost-usd" in text or "BudgetExceeded" in text:
        return "cost_cap"
    return "other"


def summarize(traces: list[dict]) -> dict:
    """Aggregate numbers over all traces."""
    run_count = len(traces)
    agent_tokens: Counter = Counter()
    user_tokens: Counter = Counter()
    tool_call_count = malformed_count = unknown_count = tool_error_count = 0
    text_call_count = parse_failed_count = trimmed_count = agent_turn_count = 0
    unknown_tool_names: Counter = Counter()
    guard_rejections_by_rule: Counter = Counter()
    guard_allowed_by_rule: Counter = Counter()

    for trace in traces:
        for message in trace["messages"]:
            if message.get("role") == "user":
                _add_usage(user_tokens, message.get("usage"))
        for turn in trace["agent_turns"]:
            agent_turn_count += 1
            _add_usage(agent_tokens, turn.get("usage"))
            text_call_count += int(turn["text_tool_call"])
            parse_failed_count += int(turn.get("tool_call_parse") == "parse_failed")
            trimmed_count += int(bool(turn.get("context_trimmed")))
            for rejection in turn.get("guard_rejections") or []:
                guard_rejections_by_rule[rejection["rule"]] += 1
            if turn.get("guard_allowed_after_retries"):
                guard_allowed_by_rule[turn["guard_allowed_after_retries"]] += 1
            for tool_call in turn["parsed_tool_calls"]:
                tool_call_count += 1
                malformed_count += int(tool_call["malformed_arguments"])
                if tool_call["unknown_tool"]:
                    unknown_count += 1
                    unknown_tool_names[tool_call["name"]] += 1
            tool_error_count += sum(1 for result in turn["tool_results"] if result["error"])

    def _sum_of_numbers(key: str) -> float | None:
        numbers = [trace[key] for trace in traces if isinstance(trace.get(key), (int, float))]
        if numbers:
            return sum(numbers)
        return None

    passed_count = sum(1 for trace in traces if trace["passed"])
    infra_error_traces = [trace for trace in traces if trace["termination_reason"] == INFRA_ERROR]
    clean_run_count = run_count - len(infra_error_traces)
    infra_error_messages = Counter((trace.get("error") or "")[:150] for trace in infra_error_traces)
    passed_clean_count = sum(
        1 for trace in traces if trace["passed"] and trace["termination_reason"] != INFRA_ERROR
    )
    return {
        "num_runs": run_count,
        "num_tasks": len({trace["task_id"] for trace in traces}),
        "num_passed": passed_count,
        "pass_rate": _average(passed_count, run_count),
        "num_infra_errors": len(infra_error_traces),
        "pass_rate_excluding_infra_errors": _average(passed_clean_count, clean_run_count),
        "infra_error_messages": dict(infra_error_messages.most_common()),
        "infra_errors_by_kind": dict(Counter(infra_error_kind(trace.get("error")) for trace in infra_error_traces)),
        "infra_error_rate": _average(len(infra_error_traces), run_count),
        "avg_reward": _average(sum(trace["reward"] or 0.0 for trace in traces), run_count),
        "tool_calls": tool_call_count,
        "malformed_tool_calls": malformed_count,
        "unknown_tool_calls": unknown_count,
        "unknown_tool_names": dict(unknown_tool_names),
        "tool_errors": tool_error_count,
        "tool_calls_in_text": text_call_count,
        "tool_call_parse_failures": parse_failed_count,
        "context_trimmed_turns": trimmed_count,
        "guard_rejections_by_rule": dict(guard_rejections_by_rule),
        "guard_allowed_after_retries_by_rule": dict(guard_allowed_by_rule),
        "avg_agent_turns": _average(agent_turn_count, run_count),
        "avg_messages": _average(sum(trace["num_messages"] for trace in traces), run_count),
        "termination_reasons": dict(Counter(trace["termination_reason"] for trace in traces)),
        "agent_tokens": dict(agent_tokens),
        "user_tokens": dict(user_tokens),
        "cached_prompt_tokens": {
            "agent": agent_tokens["cached_prompt_tokens"],
            "user": user_tokens["cached_prompt_tokens"],
        },
        "cache_hit_rate": {
            "agent": hit_rate(agent_tokens["prompt_tokens"], agent_tokens["cached_prompt_tokens"]),
            "user": hit_rate(user_tokens["prompt_tokens"], user_tokens["cached_prompt_tokens"]),
        },
        "agent_cost_litellm": _sum_of_numbers("agent_cost"),
        "user_cost_litellm": _sum_of_numbers("user_cost"),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with open(path, "w") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, default=str) + "\n")

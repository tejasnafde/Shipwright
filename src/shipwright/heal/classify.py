"""DETECT, CLASSIFY and ROUTE failed runs (steps 03 and 04 in docs/system-design.html).

Input: trace records from ``traces.jsonl`` (see ``shipwright.heal.traces``).

1. DETECT: a run fails when reward < 1. Infra-error runs get the label
   ``environment`` and route ``no_fix``; they are not classified further.
2. Rule labels (deterministic, cheap): loop, unknown_tool, malformed_args,
   tool_error_ignored, context_overflow, invented_id, tool_example_copied
   (an invented id that is an example value from a tool description).
3. LLM judge (optional) for failed runs that no rule explains.
4. ROUTE: fault layer tool -> tool_fix, model -> weight_fix, environment -> no_fix.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable

from shipwright.llm.prompted_tools import _first_json_object
from shipwright.heal.traces import INFRA_ERROR, PASS_THRESHOLD

JUDGE_LABELS = (
    "wrong_tool", "bad_arguments", "invented_id", "wrong_order", "skipped_policy_step",
    "loop", "invented_answer", "premature_transfer", "tool_description_unclear", "environment",
)
FAULT_LAYERS = ("tool", "model", "environment")
ROUTES = {"tool": "tool_fix", "model": "weight_fix", "environment": "no_fix"}

# Fault layer for each rule label. Rule labels are listed in priority order:
# the first one found decides the route when no judge verdict exists.
RULE_LAYERS = {
    "tool_example_copied": "tool",  # agent used an example value from a tool description as a real id
    "context_overflow": "tool",  # our harness lost context: fix the harness, not the weights
    "unknown_tool": "model",
    "malformed_args": "model",
    "invented_id": "model",
    "loop": "model",
    "tool_error_ignored": "model",
}

READ_PREFIXES = ("get_", "find_", "list_", "calculate", "search_")
_ID_LIKE = re.compile(r"^(#?[A-Za-z]*\d[\w#-]*|[a-z]+(_[a-z0-9]+)+)$")


def is_failed(trace: dict) -> bool:
    reward = trace.get("reward")
    return reward is None or reward < PASS_THRESHOLD


def _is_read(name: str | None) -> bool:
    return bool(name) and name.startswith(READ_PREFIXES)


def _calls(trace: dict) -> list[dict]:
    """Agent tool calls in order, with message index and error flag."""
    calls = []
    for turn in trace.get("agent_turns") or []:
        result_by_call_id = {result["id"]: result for result in turn.get("tool_results") or []}
        for tool_call in turn.get("parsed_tool_calls") or []:
            tool_result = result_by_call_id.get(tool_call.get("id")) or {}
            calls.append({
                "index": turn["message_index"],
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments") or {},
                "key": json.dumps([tool_call.get("name"), tool_call.get("arguments")], sort_keys=True),
                "error": bool(tool_result.get("error")),
                "malformed": bool(tool_call.get("malformed_arguments")),
                "unknown": bool(tool_call.get("unknown_tool")),
            })
    return calls


def _id_values(arguments: Any, argument_name: str = "") -> list[str]:
    """String argument values that look like ids (order, user, item, payment ids)."""
    if isinstance(arguments, dict):
        id_values = []
        for name, value in arguments.items():
            id_values += _id_values(value, name)
        return id_values
    if isinstance(arguments, list):
        id_values = []
        for value in arguments:
            id_values += _id_values(value, argument_name)
        return id_values
    if isinstance(arguments, str):
        stripped_value = arguments.strip()
        name_marks_an_id = argument_name.endswith(("_id", "_ids"))
        if name_marks_an_id or (len(stripped_value) >= 5 and _ID_LIKE.match(stripped_value)):
            if stripped_value:
                return [stripped_value]
    return []


def tool_example_values(tools: list[dict]) -> set[str]:
    """Example values ("such as '#W0000000'") from tool parameter descriptions."""
    example_values = set()
    for tool in tools:
        function = tool.get("function", tool)
        parameter_schemas = ((function.get("parameters") or {}).get("properties") or {}).values()
        for schema in parameter_schemas:
            for match in re.finditer(r"such as '([^']+)'", schema.get("description") or ""):
                example_values.add(match.group(1))
    return example_values


def rule_labels(trace: dict, example_values: set[str] | None = None) -> list[dict]:
    """All rule hits as {label, turn, detail}, sorted by turn.

    ``example_values``: example ids from the tool descriptions. An invented id equal
    to one of them also gets ``tool_example_copied`` (a tool-description fault).
    """
    example_values = example_values or set()
    hits: list[dict] = []
    calls = _calls(trace)
    messages = trace.get("messages") or []

    for call in calls:
        if call["unknown"]:
            hits.append({"label": "unknown_tool", "turn": call["index"], "detail": call["name"]})
        if call["malformed"]:
            hits.append({"label": "malformed_args", "turn": call["index"], "detail": call["name"]})
    for turn in trace.get("agent_turns") or []:
        if turn.get("tool_call_parse") == "parse_failed":
            hits.append({"label": "malformed_args", "turn": turn["message_index"], "detail": "parse_failed"})
        if turn.get("context_trimmed"):
            hits.append({
                "label": "context_overflow",
                "turn": turn["message_index"],
                "detail": f"dropped {turn.get('context_dropped_messages', 0)} messages",
            })

    # loop: a call repeated after it failed, or the same call made 3 or more times.
    previous_call_by_key: dict[str, dict] = {}
    call_count_by_key: Counter = Counter()
    for call in calls:
        call_count_by_key[call["key"]] += 1
        previous_call = previous_call_by_key.get(call["key"])
        is_repeat_after_error = previous_call is not None and previous_call["error"]
        is_third_call = previous_call is not None and call_count_by_key[call["key"]] == 3
        if is_repeat_after_error or is_third_call:
            hits.append({
                "label": "loop",
                "turn": call["index"],
                "detail": f"{call['name']} repeated (x{call_count_by_key[call['key']]})",
            })
        previous_call_by_key[call["key"]] = call

    # tool_error_ignored: after a tool error the next call is a write or a transfer,
    # with no lookup in between.
    for previous_call, next_call in zip(calls, calls[1:]):
        if (
            previous_call["error"]
            and not _is_read(next_call["name"])
            and next_call["key"] != previous_call["key"]
        ):
            hits.append({
                "label": "tool_error_ignored",
                "turn": next_call["index"],
                "detail": f"{previous_call['name']} failed, then {next_call['name']}",
            })

    # invented_id: an id-like argument that never appeared earlier in the conversation.
    for call in calls:
        earlier_text = "\n".join(
            str(message.get("content") or "")
            for message in messages[: call["index"]]
            if message.get("role") in ("user", "tool")
        )
        for value in _id_values(call["arguments"]):
            if value not in earlier_text and value.lstrip("#") not in earlier_text:
                hits.append({
                    "label": "invented_id",
                    "turn": call["index"],
                    "detail": f"{call['name']}: {value}",
                })
                if value in example_values:
                    hits.append({
                        "label": "tool_example_copied",
                        "turn": call["index"],
                        "detail": f"{call['name']}: {value} is a tool description example",
                    })

    # One hit per (label, turn); keep the first detail.
    unique_hits: dict[tuple, dict] = {}
    for hit in hits:
        unique_hits.setdefault((hit["label"], hit["turn"]), hit)
    label_priority = list(RULE_LAYERS)
    return sorted(unique_hits.values(), key=lambda hit: (hit["turn"], label_priority.index(hit["label"])))


def route(fault_layer: str) -> str:
    return ROUTES.get(fault_layer, "no_fix")


def _rule_layer(labels: list[str]) -> str:
    for label in RULE_LAYERS:
        if label in labels:
            return RULE_LAYERS[label]
    return "model"


# ---------------------------------------------------------------- LLM judge

JUDGE_SYSTEM = """You review a failed run of a customer support agent that uses tools.
Find the first agent turn that went wrong and why.
Labels: {labels}.
fault_layer: "tool" if a tool description or tool behaviour misled the agent, "model" if the agent
reasoned or acted wrongly, "environment" if the user simulator or the infrastructure failed.
Reply with only a JSON object:
{{"first_wrong_turn": <turn number>, "label": "<one label>", "fault_layer": "<tool|model|environment>", "explanation": "<one or two sentences>"}}"""


def compact_trace(trace: dict, limit: int = 300) -> str:
    lines = []
    tool_name_by_call_id: dict[str, str] = {}
    for message_index, message in enumerate(trace.get("messages") or []):
        role = message.get("role")
        content = " ".join(str(message.get("content") or "").split())[:limit]
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                tool_name_by_call_id[tool_call.get("id", "")] = tool_call.get("name", "")
                arguments_text = json.dumps(tool_call.get("arguments"))[:limit]
                lines.append(f"[{message_index}] agent calls {tool_call.get('name')}({arguments_text})")
            if content:
                lines.append(f"[{message_index}] agent: {content}")
        elif role == "tool":
            error_mark = " (ERROR)" if message.get("error") else ""
            tool_name = tool_name_by_call_id.get(message.get("id", ""), "")
            lines.append(f"[{message_index}] tool {tool_name}{error_mark}: {content}")
        else:
            lines.append(f"[{message_index}] {role}: {content}")
    return "\n".join(lines)


def expected_actions(trace: dict) -> str:
    action_checks = (trace.get("reward_info") or {}).get("action_checks") or []
    lines = []
    for action_check in action_checks:
        action = action_check.get("action") or {}
        mark = "done" if action_check.get("action_match") else "MISSING"
        lines.append(f"- {action.get('name')}({json.dumps(action.get('arguments'))}) [{mark}]")
    return "\n".join(lines) or "(none)"


def judge_messages(trace: dict, instruction: str | None) -> list[dict]:
    user_message = (
        f"User goal:\n{instruction or '(unknown)'}\n\n"
        f"Expected agent actions:\n{expected_actions(trace)}\n\n"
        f"Run (turn numbers in brackets):\n{compact_trace(trace)}"
    )
    system_message = JUDGE_SYSTEM.format(labels=", ".join(JUDGE_LABELS))
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def parse_verdict(text: str | None) -> dict:
    """Validate a judge reply. Raises ValueError with a short reason."""
    verdict_text = _first_json_object(text or "")
    if not verdict_text:
        raise ValueError("no JSON object")
    verdict = json.loads(verdict_text)
    if verdict.get("label") not in JUDGE_LABELS:
        raise ValueError(f"label must be one of {', '.join(JUDGE_LABELS)}")
    if verdict.get("fault_layer") not in FAULT_LAYERS:
        raise ValueError("fault_layer must be tool, model or environment")

    first_wrong_turn = verdict.get("first_wrong_turn")
    if isinstance(first_wrong_turn, str) and first_wrong_turn.strip().lstrip("[").rstrip("]").isdigit():
        first_wrong_turn = int(first_wrong_turn.strip().strip("[]"))
    if not isinstance(first_wrong_turn, int):
        raise ValueError("first_wrong_turn must be an integer")
    return {
        "first_wrong_turn": first_wrong_turn,
        "label": verdict["label"],
        "fault_layer": verdict["fault_layer"],
        "explanation": str(verdict.get("explanation", "")),
    }


def run_judge(trace: dict, instruction: str | None, chat: Callable[[list[dict]], str]) -> dict:
    """Ask the judge; on an invalid reply, retry once with the reason."""
    messages = judge_messages(trace, instruction)
    reply = chat(messages)
    try:
        return parse_verdict(reply)
    except (ValueError, json.JSONDecodeError) as reply_error:
        messages += [
            {"role": "assistant", "content": reply or ""},
            {"role": "user", "content": f"Invalid reply ({reply_error}). Reply with only the JSON object."},
        ]
        return parse_verdict(chat(messages))


# ---------------------------------------------------------------- pipeline

def classify_run(
    trace: dict,
    instruction: str | None = None,
    chat: Callable[[list[dict]], str] | None = None,
    judge_all: bool = False,
    example_values: set[str] | None = None,
) -> dict | None:
    """One failures.jsonl row for a failed run, or None if the run passed."""
    is_infra_error = trace.get("termination_reason") == INFRA_ERROR
    if not is_infra_error and not is_failed(trace):
        return None

    row: dict[str, Any] = {
        "task_id": trace.get("task_id"),
        "trial": trace.get("trial"),
        "simulation_id": trace.get("simulation_id"),
        "reward": trace.get("reward"),
    }
    if is_infra_error:
        return {
            **row,
            "labels": ["environment"],
            "rule_hits": [],
            "first_wrong_turn": None,
            "fault_layer": "environment",
            "route": "no_fix",
            "source": "detect",
            "explanation": (trace.get("error") or "infrastructure error")[:300],
        }

    rule_hits = rule_labels(trace, example_values)
    labels = list(dict.fromkeys(hit["label"] for hit in rule_hits))

    verdict = None
    judge_error = None
    if chat is not None and (judge_all or not rule_hits):
        try:
            verdict = run_judge(trace, instruction, chat)
        except Exception as judge_exception:  # noqa: BLE001 - record and fall back to rules
            judge_error = str(judge_exception)[:300]

    if verdict:
        labels = list(dict.fromkeys(labels + [verdict["label"]]))
        fault_layer = verdict["fault_layer"]
        first_wrong_turn = verdict["first_wrong_turn"]
        source = "judge"
        explanation = verdict["explanation"]
    elif rule_hits:
        fault_layer = _rule_layer(labels)
        first_wrong_turn = rule_hits[0]["turn"]
        source = "rules"
        explanation = "; ".join(
            f"{hit['label']} at turn {hit['turn']}: {hit['detail']}" for hit in rule_hits[:5]
        )
    else:
        fault_layer = "model"
        first_wrong_turn = None
        source = "unclassified"
        labels = ["unclassified"]
        explanation = judge_error or "no rule matched and no judge verdict"

    row.update({
        "labels": labels,
        "rule_hits": rule_hits,
        "first_wrong_turn": first_wrong_turn,
        "fault_layer": fault_layer,
        "route": route(fault_layer),
        "source": source,
        "explanation": explanation,
    })
    if judge_error:
        row["judge_error"] = judge_error
    return row


def summarize_failures(rows: list[dict], num_runs: int) -> dict:
    """Counts over failures.jsonl rows. Label, route and layer counts cover failed runs only;
    passed runs that only AgentTune's detector flagged are counted in num_detector_only."""
    detector_only_rows = [row for row in rows if row.get("outcome_failed") is False]
    failed_rows = [row for row in rows if row.get("outcome_failed") is not False]

    label_counts: Counter = Counter()
    for row in failed_rows:
        label_counts.update(row["labels"])
    detected_by_counts = Counter(
        "+".join(row.get("detected_by") or ["outcome"]) for row in failed_rows + detector_only_rows
    )
    root_cause_counts = Counter(
        row["agenttune_root_cause"] for row in failed_rows if row.get("agenttune_root_cause")
    )
    return {
        "num_runs": num_runs,
        "num_failed": len(failed_rows),
        "num_detector_only": len(detector_only_rows),
        "per_detected_by": dict(detected_by_counts.most_common()),
        "per_agenttune_root_cause": dict(root_cause_counts.most_common()),
        "per_label": dict(label_counts.most_common()),
        "per_route": dict(Counter(row["route"] for row in failed_rows).most_common()),
        "per_fault_layer": dict(Counter(row["fault_layer"] for row in failed_rows).most_common()),
        "per_source": dict(Counter(row["source"] for row in failed_rows).most_common()),
    }

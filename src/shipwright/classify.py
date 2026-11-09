"""DETECT, CLASSIFY and ROUTE failed runs (steps 03 and 04 in docs/system-design.html).

Input: trace records from ``traces.jsonl`` (see ``shipwright.traces``).

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

from shipwright.prompted_tools import _first_json_object
from shipwright.traces import INFRA_ERROR, PASS_THRESHOLD

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
    out = []
    for turn in trace.get("agent_turns") or []:
        results = {r["id"]: r for r in turn.get("tool_results") or []}
        for tc in turn.get("parsed_tool_calls") or []:
            res = results.get(tc.get("id")) or {}
            out.append({
                "index": turn["message_index"],
                "name": tc.get("name"),
                "arguments": tc.get("arguments") or {},
                "key": json.dumps([tc.get("name"), tc.get("arguments")], sort_keys=True),
                "error": bool(res.get("error")),
                "malformed": bool(tc.get("malformed_arguments")),
                "unknown": bool(tc.get("unknown_tool")),
            })
    return out


def _id_values(arguments: Any, key: str = "") -> list[str]:
    """String argument values that look like ids (order, user, item, payment ids)."""
    if isinstance(arguments, dict):
        return [v for k, val in arguments.items() for v in _id_values(val, k)]
    if isinstance(arguments, list):
        return [v for val in arguments for v in _id_values(val, key)]
    if isinstance(arguments, str):
        s = arguments.strip()
        if key.endswith(("_id", "_ids")) or (len(s) >= 5 and _ID_LIKE.match(s)):
            return [s] if s else []
    return []


def tool_example_values(tools: list[dict]) -> set[str]:
    """Example values ("such as '#W0000000'") from tool parameter descriptions."""
    out = set()
    for tool in tools:
        fn = tool.get("function", tool)
        for schema in ((fn.get("parameters") or {}).get("properties") or {}).values():
            for m in re.finditer(r"such as '([^']+)'", schema.get("description") or ""):
                out.add(m.group(1))
    return out


def rule_labels(trace: dict, example_values: set[str] | None = None) -> list[dict]:
    """All rule hits as {label, turn, detail}, sorted by turn.

    ``example_values``: example ids from the tool descriptions. An invented id equal
    to one of them also gets ``tool_example_copied`` (a tool-description fault).
    """
    example_values = example_values or set()
    hits: list[dict] = []
    calls = _calls(trace)
    messages = trace.get("messages") or []

    for c in calls:
        if c["unknown"]:
            hits.append({"label": "unknown_tool", "turn": c["index"], "detail": c["name"]})
        if c["malformed"]:
            hits.append({"label": "malformed_args", "turn": c["index"], "detail": c["name"]})
    for turn in trace.get("agent_turns") or []:
        if turn.get("tool_call_parse") == "parse_failed":
            hits.append({"label": "malformed_args", "turn": turn["message_index"], "detail": "parse_failed"})
        if turn.get("context_trimmed"):
            hits.append({"label": "context_overflow", "turn": turn["message_index"],
                         "detail": f"dropped {turn.get('context_dropped_messages', 0)} messages"})

    # loop: a call repeated after it failed, or the same call made 3 or more times.
    seen: dict[str, dict] = {}
    count: Counter = Counter()
    for c in calls:
        count[c["key"]] += 1
        prev = seen.get(c["key"])
        if prev is not None and (prev["error"] or count[c["key"]] == 3):
            hits.append({"label": "loop", "turn": c["index"],
                         "detail": f"{c['name']} repeated (x{count[c['key']]})"})
        seen[c["key"]] = c

    # tool_error_ignored: after a tool error the next call is a write or a transfer,
    # with no lookup in between.
    for prev, nxt in zip(calls, calls[1:]):
        if prev["error"] and not _is_read(nxt["name"]) and nxt["key"] != prev["key"]:
            hits.append({"label": "tool_error_ignored", "turn": nxt["index"],
                         "detail": f"{prev['name']} failed, then {nxt['name']}"})

    # invented_id: an id-like argument that never appeared earlier in the conversation.
    for c in calls:
        earlier = "\n".join(
            str(m.get("content") or "") for m in messages[: c["index"]]
            if m.get("role") in ("user", "tool")
        )
        for value in _id_values(c["arguments"]):
            if value not in earlier and value.lstrip("#") not in earlier:
                hits.append({"label": "invented_id", "turn": c["index"],
                             "detail": f"{c['name']}: {value}"})
                if value in example_values:
                    hits.append({"label": "tool_example_copied", "turn": c["index"],
                                 "detail": f"{c['name']}: {value} is a tool description example"})

    # One hit per (label, turn); keep the first detail.
    unique: dict[tuple, dict] = {}
    for h in hits:
        unique.setdefault((h["label"], h["turn"]), h)
    return sorted(unique.values(), key=lambda h: (h["turn"], list(RULE_LAYERS).index(h["label"])))


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
    names: dict[str, str] = {}
    for i, m in enumerate(trace.get("messages") or []):
        role = m.get("role")
        content = " ".join(str(m.get("content") or "").split())[:limit]
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                names[tc.get("id", "")] = tc.get("name", "")
                lines.append(f"[{i}] agent calls {tc.get('name')}({json.dumps(tc.get('arguments'))[:limit]})")
            if content:
                lines.append(f"[{i}] agent: {content}")
        elif role == "tool":
            err = " (ERROR)" if m.get("error") else ""
            lines.append(f"[{i}] tool {names.get(m.get('id', ''), '')}{err}: {content}")
        else:
            lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines)


def expected_actions(trace: dict) -> str:
    checks = (trace.get("reward_info") or {}).get("action_checks") or []
    out = []
    for c in checks:
        a = c.get("action") or {}
        mark = "done" if c.get("action_match") else "MISSING"
        out.append(f"- {a.get('name')}({json.dumps(a.get('arguments'))}) [{mark}]")
    return "\n".join(out) or "(none)"


def judge_messages(trace: dict, instruction: str | None) -> list[dict]:
    user = (
        f"User goal:\n{instruction or '(unknown)'}\n\n"
        f"Expected agent actions:\n{expected_actions(trace)}\n\n"
        f"Run (turn numbers in brackets):\n{compact_trace(trace)}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM.format(labels=", ".join(JUDGE_LABELS))},
            {"role": "user", "content": user}]


def parse_verdict(text: str | None) -> dict:
    """Validate a judge reply. Raises ValueError with a short reason."""
    obj_text = _first_json_object(text or "")
    if not obj_text:
        raise ValueError("no JSON object")
    obj = json.loads(obj_text)
    if obj.get("label") not in JUDGE_LABELS:
        raise ValueError(f"label must be one of {', '.join(JUDGE_LABELS)}")
    if obj.get("fault_layer") not in FAULT_LAYERS:
        raise ValueError("fault_layer must be tool, model or environment")
    turn = obj.get("first_wrong_turn")
    if isinstance(turn, str) and turn.strip().lstrip("[").rstrip("]").isdigit():
        turn = int(turn.strip().strip("[]"))
    if not isinstance(turn, int):
        raise ValueError("first_wrong_turn must be an integer")
    return {"first_wrong_turn": turn, "label": obj["label"], "fault_layer": obj["fault_layer"],
            "explanation": str(obj.get("explanation", ""))}


def run_judge(trace: dict, instruction: str | None, chat: Callable[[list[dict]], str]) -> dict:
    """Ask the judge; on an invalid reply, retry once with the reason."""
    messages = judge_messages(trace, instruction)
    reply = chat(messages)
    try:
        return parse_verdict(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        messages += [{"role": "assistant", "content": reply or ""},
                     {"role": "user", "content": f"Invalid reply ({exc}). Reply with only the JSON object."}]
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
    if trace.get("termination_reason") != INFRA_ERROR and not is_failed(trace):
        return None
    row: dict[str, Any] = {"task_id": trace.get("task_id"), "trial": trace.get("trial"),
                           "simulation_id": trace.get("simulation_id"), "reward": trace.get("reward")}
    if trace.get("termination_reason") == INFRA_ERROR:
        return {**row, "labels": ["environment"], "rule_hits": [], "first_wrong_turn": None,
                "fault_layer": "environment", "route": "no_fix", "source": "detect",
                "explanation": (trace.get("error") or "infrastructure error")[:300]}

    hits = rule_labels(trace, example_values)
    labels = list(dict.fromkeys(h["label"] for h in hits))
    verdict = None
    judge_error = None
    if chat is not None and (judge_all or not hits):
        try:
            verdict = run_judge(trace, instruction, chat)
        except Exception as exc:  # noqa: BLE001 - record and fall back to rules
            judge_error = str(exc)[:300]

    if verdict:
        labels = list(dict.fromkeys(labels + [verdict["label"]]))
        layer, first, source = verdict["fault_layer"], verdict["first_wrong_turn"], "judge"
        explanation = verdict["explanation"]
    elif hits:
        layer, first, source = _rule_layer(labels), hits[0]["turn"], "rules"
        explanation = "; ".join(f"{h['label']} at turn {h['turn']}: {h['detail']}" for h in hits[:5])
    else:
        layer, first, source = "model", None, "unclassified"
        labels = ["unclassified"]
        explanation = judge_error or "no rule matched and no judge verdict"

    row.update({"labels": labels, "rule_hits": hits, "first_wrong_turn": first,
                "fault_layer": layer, "route": route(layer), "source": source,
                "explanation": explanation})
    if judge_error:
        row["judge_error"] = judge_error
    return row


def summarize_failures(rows: list[dict], num_runs: int) -> dict:
    """Counts over failures.jsonl rows. Label, route and layer counts cover failed runs only;
    passed runs that only AgentTune's detector flagged are counted in num_detector_only."""
    detected_only = [r for r in rows if r.get("outcome_failed") is False]
    rows = [r for r in rows if r.get("outcome_failed") is not False]
    labels: Counter = Counter()
    for r in rows:
        labels.update(r["labels"])
    return {
        "num_runs": num_runs,
        "num_failed": len(rows),
        "num_detector_only": len(detected_only),
        "per_detected_by": dict(Counter("+".join(r.get("detected_by") or ["outcome"])
                                        for r in rows + detected_only).most_common()),
        "per_agenttune_root_cause": dict(Counter(r["agenttune_root_cause"] for r in rows
                                                 if r.get("agenttune_root_cause")).most_common()),
        "per_label": dict(labels.most_common()),
        "per_route": dict(Counter(r["route"] for r in rows).most_common()),
        "per_fault_layer": dict(Counter(r["fault_layer"] for r in rows).most_common()),
        "per_source": dict(Counter(r["source"] for r in rows).most_common()),
    }

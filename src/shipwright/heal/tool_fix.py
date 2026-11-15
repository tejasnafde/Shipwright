"""Tool-fix branch: propose, check, apply and roll back patches to tool descriptions.

A patch edits one text field of one tool:
  ``description`` or ``parameters.<param>.description``.
Names, parameter names and types are never edited.

Patches live in an append-and-update ledger (``patches/ledger.jsonl``), one row per
patch, with status proposed / applied / rolled_back. Only ``applied`` patches take
effect, in ledger order, and each must find the exact ``before`` text (sha256 check).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from shipwright.llm.prompted_tools import _first_json_object

STATUSES = ("proposed", "applied", "rolled_back")

# Concrete id shapes in the retail domain, and the placeholder that replaces each.
ID_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"#W\d{7}\b"), "#W<7 digits>"),
    (re.compile(r"\b(gift_card|credit_card|paypal)_\d{3,}\b"), r"\1_<digits>"),
    (re.compile(r"\b[a-z]+_[a-z]+_\d{2,}\b"), "<first>_<last>_<digits>"),
    (re.compile(r"\b\d{10}\b"), "<10 digits>"),
]


def sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- fields

def _function_of(tool: dict) -> dict:
    """The function spec of a tool: tool["function"] for OpenAI-style tools, else the tool itself."""
    return tool.get("function", tool)


def tool_map(tools: list[dict]) -> dict[str, dict]:
    return {_function_of(tool)["name"]: tool for tool in tools}


def _parameter_description_name(field: str) -> str | None:
    """"parameters.<name>.description" -> <name>. None for any other field."""
    field_parts = field.split(".")
    if len(field_parts) == 3 and field_parts[0] == "parameters" and field_parts[2] == "description":
        return field_parts[1]
    return None


def get_field(tool: dict, field: str) -> str | None:
    function = _function_of(tool)
    if field == "description":
        return function.get("description")

    parameter_name = _parameter_description_name(field)
    if parameter_name is None:
        return None
    parameter_schema = ((function.get("parameters") or {}).get("properties") or {}).get(parameter_name)
    if parameter_schema is None:
        return None
    return parameter_schema.get("description", "")


def set_field(tool: dict, field: str, text: str) -> dict:
    """A copy of tool with one text field replaced. Only description fields."""
    tool_copy = copy.deepcopy(tool)
    function = _function_of(tool_copy)
    if field == "description":
        function["description"] = text
        return tool_copy

    parameter_name = _parameter_description_name(field)
    if parameter_name is not None:
        parameter_schemas = (function.get("parameters") or {}).get("properties") or {}
        if parameter_name in parameter_schemas:
            parameter_schemas[parameter_name]["description"] = text
            return tool_copy
    raise ValueError(f"field not editable: {field}")


def editable_fields(tool: dict) -> list[str]:
    parameter_schemas = (_function_of(tool).get("parameters") or {}).get("properties") or {}
    return ["description"] + [f"parameters.{name}.description" for name in parameter_schemas]


# ---------------------------------------------------------------- example ids

def concrete_ids(text: str | None) -> list[str]:
    found_ids = []
    for pattern, _ in ID_PATTERNS:
        found_ids += [match.group(0) for match in pattern.finditer(text or "")]
    return list(dict.fromkeys(found_ids))


def replace_ids(text: str) -> str:
    for pattern, placeholder in ID_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def validate_patch(patch: dict, tools: dict[str, dict]) -> list[str]:
    """Reasons to reject a patch (empty list means it is valid)."""
    tool = tools.get(patch.get("tool"))
    if tool is None:
        return [f"unknown tool {patch.get('tool')!r}"]

    problems = []
    if patch.get("field") not in editable_fields(tool):
        problems.append(f"field {patch.get('field')!r} is not an editable description field")

    new_text = patch.get("new_text")
    if not isinstance(new_text, str) or not new_text.strip():
        problems.append("new_text is empty")
    elif concrete_ids(new_text):
        problems.append(f"new_text contains concrete example ids: {', '.join(concrete_ids(new_text))}")
    elif get_field(tool, patch.get("field", "")) == new_text:
        problems.append("new_text is the same as the current text")
    return problems


def deterministic_patches(tools: dict[str, dict]) -> list[dict]:
    """Replace concrete example ids in every description field with placeholders."""
    patches = []
    for tool_name, tool in tools.items():
        for field in editable_fields(tool):
            text_before = get_field(tool, field) or ""
            text_after = replace_ids(text_before)
            if text_after != text_before:
                patches.append({
                    "tool": tool_name,
                    "field": field,
                    "new_text": text_after,
                    "reason": "Replace concrete example ids with placeholders: the agent "
                              "copied them as real ids (tool_example_copied).",
                })
    return patches


# ---------------------------------------------------------------- ledger

def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def next_round(rows: list[dict]) -> int:
    return max((row.get("round", 0) for row in rows), default=0) + 1


def make_row(patch: dict, before: str, round_number: int, source: str, failures: list[str]) -> dict:
    after = patch["new_text"]
    patch_id = "tp-" + sha256(f"{patch['tool']}|{patch['field']}|{before}|{after}")[:10]
    return {
        "id": patch_id,
        "round": round_number,
        "tool": patch["tool"],
        "field": patch["field"],
        "before": before,
        "after": after,
        "reason": patch.get("reason", ""),
        "source": source,
        "source_failures": failures,
        "sha256_before": sha256(before),
        "sha256_after": sha256(after),
        "status": "proposed",
        "history": [{"status": "proposed", "at": now()}],
    }


def apply_patches(tools: list[dict], rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Tools with every ``applied`` patch applied (ledger order), and the active ids."""
    patched_tool_by_name = {name: copy.deepcopy(tool) for name, tool in tool_map(tools).items()}
    active_patch_ids = []
    for row in rows:
        if row.get("status") != "applied":
            continue
        tool = patched_tool_by_name.get(row["tool"])
        if tool is None:
            raise ValueError(f"patch {row['id']}: unknown tool {row['tool']}")
        current_text = get_field(tool, row["field"])
        if sha256(current_text) != row["sha256_before"]:
            raise ValueError(f"patch {row['id']}: {row['tool']}.{row['field']} does not match its 'before' text")
        patched_tool_by_name[row["tool"]] = set_field(tool, row["field"], row["after"])
        active_patch_ids.append(row["id"])

    original_order = [_function_of(tool)["name"] for tool in tools]
    return [patched_tool_by_name[name] for name in original_order], active_patch_ids


def select_patches(rows: list[dict], ids: list[str] | str) -> list[dict]:
    """The ledger rows a system under test runs with, all marked ``applied``.

    ``ids`` is a list of patch ids (ledger order is kept, whatever their status), or the
    string ``"applied"`` for the rows whose status is applied now. Used by the gate, which
    must compare an exact patch set, not whatever the ledger says at run time.
    """
    if ids == "applied":
        chosen_rows = [row for row in rows if row.get("status") == "applied"]
    elif isinstance(ids, str):
        raise ValueError(f"patch ids must be a list or 'applied', got {ids!r}")
    else:
        known_ids = {row["id"] for row in rows}
        missing_ids = [patch_id for patch_id in ids if patch_id not in known_ids]
        if missing_ids:
            raise KeyError(f"patch ids not in the ledger: {missing_ids}")
        wanted_ids = set(ids)
        chosen_rows = [row for row in rows if row["id"] in wanted_ids]
    return [{**copy.deepcopy(row), "status": "applied"} for row in chosen_rows]


def set_status(rows: list[dict], patch_id: str, status: str, tools: list[dict] | None = None) -> list[dict]:
    """Change one patch's status. With tools given, check the applied set still applies."""
    if status not in STATUSES:
        raise ValueError(f"bad status {status}")
    rows = copy.deepcopy(rows)
    matching_rows = [row for row in rows if row["id"] == patch_id]
    if not matching_rows:
        raise KeyError(f"no patch {patch_id}")
    patch_row = matching_rows[0]
    patch_row["status"] = status
    patch_row.setdefault("history", []).append({"status": status, "at": now()})
    if tools is not None:
        apply_patches(tools, rows)  # raises on a conflict
    return rows


# ---------------------------------------------------------------- proposals

PROPOSER_SYSTEM = """You improve tool descriptions for a small customer support model.
You get one tool spec and short excerpts of failed runs where the model misused it.
Rewrite only text fields: "description" or "parameters.<name>.description".
Never change tool names, parameter names or types. Never write concrete ids
(like order, user, item or payment ids); use placeholders such as <order_id> or #W<7 digits>.
Keep texts short and concrete: say when to use the tool, what to check first, which tool to use instead.
Reply with only a JSON object:
{"patches": [{"tool": "<name>", "field": "<field>", "new_text": "<text>", "reason": "<why>"}]}"""


def failing_tools(row: dict, trace: dict | None) -> list[str]:
    """Tools involved in a failure: the call at first_wrong_turn and tools named in rule hits."""
    tool_names = []
    agent_turns = (trace or {}).get("agent_turns") or []
    first_wrong_turn = row.get("first_wrong_turn")
    if first_wrong_turn is not None:
        turns_with_calls = [
            turn for turn in agent_turns
            if turn["message_index"] >= first_wrong_turn and turn.get("parsed_tool_calls")
        ]
        if turns_with_calls:
            tool_names.append(turns_with_calls[0]["parsed_tool_calls"][0]["name"])
    for rule_hit in row.get("rule_hits") or []:
        tool_name = rule_hit.get("detail", "").split(":")[0].split(" ")[0]
        if tool_name:
            tool_names.append(tool_name)
    return list(dict.fromkeys(tool_names))


def excerpt(row: dict, trace: dict | None, tool: str, limit: int = 200) -> str:
    lines = [f"task {row.get('task_id')}, labels {', '.join(row.get('labels', []))}"]
    for turn in (trace or {}).get("agent_turns") or []:
        # Pad the results so a call without a result still pairs up.
        tool_results = (turn.get("tool_results") or []) + [{}] * 5
        for tool_call, tool_result in zip(turn.get("parsed_tool_calls") or [], tool_results):
            if tool_call["name"] == tool:
                arguments_text = json.dumps(tool_call["arguments"])[:limit]
                result_text = str(tool_result.get("content") or "")[:limit]
                lines.append(f"called {tool}({arguments_text}) -> {result_text}")
                break
        if len(lines) >= 3:
            break
    lines.append(f"judge: {str(row.get('explanation') or '')[:300]}")
    return "\n".join(lines)


def parse_proposals(text: str) -> list[dict]:
    proposals_text = _first_json_object(text or "")
    if not proposals_text:
        raise ValueError("no JSON object")
    proposals = json.loads(proposals_text)
    patches = proposals.get("patches") if isinstance(proposals, dict) else None
    if not isinstance(patches, list):
        raise ValueError('expected {"patches": [...]}')
    return [patch for patch in patches if isinstance(patch, dict)]


def propose_for_tool(
    tool: dict,
    excerpts: list[str],
    chat: Callable[[list[dict]], str],
    tools: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """LLM patches for one tool, validated; one retry with the reasons. Returns (valid, rejections)."""
    tool_name = _function_of(tool)["name"]
    tool_spec_text = json.dumps(_function_of(tool), indent=1)
    messages = [
        {"role": "system", "content": PROPOSER_SYSTEM},
        {"role": "user", "content": f"Tool spec:\n{tool_spec_text}\n\nFailures:\n" + "\n\n".join(excerpts)},
    ]

    rejections: list[str] = []
    for attempt in range(2):
        reply = chat(messages)
        try:
            patches = parse_proposals(reply)
            problems = []
        except (ValueError, json.JSONDecodeError) as reply_error:
            patches = []
            problems = [f"invalid reply: {reply_error}"]

        valid_patches = []
        for patch in patches:
            if patch.get("tool") != tool_name:
                problems.append(f"patch for {patch.get('tool')!r}, expected {tool_name!r}")
                continue
            patch_problems = validate_patch(patch, tools)
            if patch_problems:
                problems.append(f"{patch.get('field')}: {'; '.join(patch_problems)}")
            else:
                valid_patches.append(patch)

        rejections += problems
        is_last_attempt = attempt == 1
        if valid_patches or is_last_attempt:
            return valid_patches, rejections
        messages += [
            {"role": "assistant", "content": reply or ""},
            {"role": "user", "content": "Rejected: " + " | ".join(problems) + ". Reply with only the corrected JSON."},
        ]
    return [], rejections


def propose(
    failures: list[dict],
    traces: dict[str, dict],
    tools: list[dict],
    ledger: list[dict],
    chat: Callable[[list[dict]], str] | None,
    round_number: int | None = None,
    max_excerpts: int = 3,
) -> tuple[list[dict], list[str]]:
    """New ledger rows (status proposed) for the tool_fix failures, and rejection notes.

    Patches are made against the current effective tools (applied patches included).
    """
    current_tools, _ = apply_patches(tools, ledger)
    current_tool_by_name = tool_map(current_tools)
    round_number = round_number or next_round(ledger)
    tool_fix_failures = [failure for failure in failures if failure.get("route") == "tool_fix"]
    # (tool, field) -> (patch, source, ids of the failures behind it)
    proposals: dict[tuple[str, str], tuple[dict, str, list[str]]] = {}
    notes: list[str] = []

    example_copy_failure_ids = [
        str(failure.get("simulation_id") or failure.get("task_id"))
        for failure in failures
        if "tool_example_copied" in failure.get("labels", [])
    ]
    if example_copy_failure_ids:
        for patch in deterministic_patches(current_tool_by_name):
            proposals[(patch["tool"], patch["field"])] = (patch, "rules", example_copy_failure_ids)

    if chat is not None:
        failures_by_tool: dict[str, list[tuple[dict, dict | None]]] = {}
        for failure in tool_fix_failures:
            trace = traces.get(str(failure.get("simulation_id")))
            for tool_name in failing_tools(failure, trace):
                if tool_name in current_tool_by_name:
                    failures_by_tool.setdefault(tool_name, []).append((failure, trace))

        for tool_name, tool_failures in failures_by_tool.items():
            excerpts = [
                excerpt(failure, trace, tool_name) for failure, trace in tool_failures[:max_excerpts]
            ]
            failure_ids = [
                str(failure.get("simulation_id") or failure.get("task_id")) for failure, _ in tool_failures
            ]
            valid_patches, rejected_reasons = propose_for_tool(
                current_tool_by_name[tool_name], excerpts, chat, current_tool_by_name
            )
            notes += [f"{tool_name}: {reason}" for reason in rejected_reasons]
            for patch in valid_patches:
                key = (patch["tool"], patch["field"])
                patch_failure_ids = failure_ids
                if key in proposals:  # LLM text replaces the placeholder-only patch
                    patch_failure_ids = list(dict.fromkeys(failure_ids + proposals[key][2]))
                proposals[key] = (patch, "llm", patch_failure_ids)

    new_rows = []
    known_ids = {row["id"] for row in ledger}
    for (tool_name, field), (patch, source, failure_ids) in proposals.items():
        current_text = get_field(current_tool_by_name[tool_name], field) or ""
        row = make_row(patch, current_text, round_number, source, failure_ids)
        if row["id"] not in known_ids:
            new_rows.append(row)
    return new_rows, notes


def patch_summary(rows: list[dict]) -> list[dict[str, Any]]:
    summary_fields = ("id", "round", "tool", "field", "source", "status")
    return [{field: row[field] for field in summary_fields} for row in rows]

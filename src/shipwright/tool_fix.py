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

from shipwright.prompted_tools import _first_json_object

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

def _fn(tool: dict) -> dict:
    return tool.get("function", tool)


def tool_map(tools: list[dict]) -> dict[str, dict]:
    return {_fn(t)["name"]: t for t in tools}


def get_field(tool: dict, field: str) -> str | None:
    fn = _fn(tool)
    if field == "description":
        return fn.get("description")
    parts = field.split(".")
    if len(parts) == 3 and parts[0] == "parameters" and parts[2] == "description":
        prop = ((fn.get("parameters") or {}).get("properties") or {}).get(parts[1])
        return None if prop is None else prop.get("description", "")
    return None


def set_field(tool: dict, field: str, text: str) -> dict:
    """A copy of tool with one text field replaced. Only description fields."""
    tool = copy.deepcopy(tool)
    fn = _fn(tool)
    if field == "description":
        fn["description"] = text
        return tool
    parts = field.split(".")
    if len(parts) == 3 and parts[0] == "parameters" and parts[2] == "description":
        props = (fn.get("parameters") or {}).get("properties") or {}
        if parts[1] in props:
            props[parts[1]]["description"] = text
            return tool
    raise ValueError(f"field not editable: {field}")


def editable_fields(tool: dict) -> list[str]:
    props = ((_fn(tool).get("parameters") or {}).get("properties") or {})
    return ["description"] + [f"parameters.{p}.description" for p in props]


# ---------------------------------------------------------------- example ids

def concrete_ids(text: str | None) -> list[str]:
    found = []
    for pattern, _ in ID_PATTERNS:
        found += [m.group(0) for m in pattern.finditer(text or "")]
    return list(dict.fromkeys(found))


def replace_ids(text: str) -> str:
    for pattern, placeholder in ID_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def validate_patch(patch: dict, tools: dict[str, dict]) -> list[str]:
    """Reasons to reject a patch (empty list means it is valid)."""
    errors = []
    tool = tools.get(patch.get("tool"))
    if tool is None:
        return [f"unknown tool {patch.get('tool')!r}"]
    if patch.get("field") not in editable_fields(tool):
        errors.append(f"field {patch.get('field')!r} is not an editable description field")
    text = patch.get("new_text")
    if not isinstance(text, str) or not text.strip():
        errors.append("new_text is empty")
    elif concrete_ids(text):
        errors.append(f"new_text contains concrete example ids: {', '.join(concrete_ids(text))}")
    elif get_field(tool, patch.get("field", "")) == text:
        errors.append("new_text is the same as the current text")
    return errors


def deterministic_patches(tools: dict[str, dict]) -> list[dict]:
    """Replace concrete example ids in every description field with placeholders."""
    out = []
    for name, tool in tools.items():
        for field in editable_fields(tool):
            before = get_field(tool, field) or ""
            after = replace_ids(before)
            if after != before:
                out.append({"tool": name, "field": field, "new_text": after,
                            "reason": "Replace concrete example ids with placeholders: the agent "
                                      "copied them as real ids (tool_example_copied)."})
    return out


# ---------------------------------------------------------------- ledger

def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def next_round(rows: list[dict]) -> int:
    return max((r.get("round", 0) for r in rows), default=0) + 1


def make_row(patch: dict, before: str, round_: int, source: str, failures: list[str]) -> dict:
    after = patch["new_text"]
    pid = "tp-" + sha256(f"{patch['tool']}|{patch['field']}|{before}|{after}")[:10]
    return {"id": pid, "round": round_, "tool": patch["tool"], "field": patch["field"],
            "before": before, "after": after, "reason": patch.get("reason", ""),
            "source": source, "source_failures": failures,
            "sha256_before": sha256(before), "sha256_after": sha256(after),
            "status": "proposed", "history": [{"status": "proposed", "at": now()}]}


def apply_patches(tools: list[dict], rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Tools with every ``applied`` patch applied (ledger order), and the active ids."""
    by_name = {k: copy.deepcopy(v) for k, v in tool_map(tools).items()}
    active = []
    for r in rows:
        if r.get("status") != "applied":
            continue
        tool = by_name.get(r["tool"])
        if tool is None:
            raise ValueError(f"patch {r['id']}: unknown tool {r['tool']}")
        current = get_field(tool, r["field"])
        if sha256(current) != r["sha256_before"]:
            raise ValueError(f"patch {r['id']}: {r['tool']}.{r['field']} does not match its 'before' text")
        by_name[r["tool"]] = set_field(tool, r["field"], r["after"])
        active.append(r["id"])
    order = [_fn(t)["name"] for t in tools]
    return [by_name[n] for n in order], active


def set_status(rows: list[dict], patch_id: str, status: str, tools: list[dict] | None = None) -> list[dict]:
    """Change one patch's status. With tools given, check the applied set still applies."""
    if status not in STATUSES:
        raise ValueError(f"bad status {status}")
    rows = copy.deepcopy(rows)
    match = [r for r in rows if r["id"] == patch_id]
    if not match:
        raise KeyError(f"no patch {patch_id}")
    match[0]["status"] = status
    match[0].setdefault("history", []).append({"status": status, "at": now()})
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
    names = []
    turns = (trace or {}).get("agent_turns") or []
    first = row.get("first_wrong_turn")
    if first is not None:
        later = [t for t in turns if t["message_index"] >= first and t.get("parsed_tool_calls")]
        if later:
            names.append(later[0]["parsed_tool_calls"][0]["name"])
    for h in row.get("rule_hits") or []:
        head = h.get("detail", "").split(":")[0].split(" ")[0]
        if head:
            names.append(head)
    return list(dict.fromkeys(names))


def excerpt(row: dict, trace: dict | None, tool: str, limit: int = 200) -> str:
    lines = [f"task {row.get('task_id')}, labels {', '.join(row.get('labels', []))}"]
    for turn in (trace or {}).get("agent_turns") or []:
        for tc, res in zip(turn.get("parsed_tool_calls") or [], (turn.get("tool_results") or []) + [{}] * 5):
            if tc["name"] == tool:
                lines.append(f"called {tool}({json.dumps(tc['arguments'])[:limit]}) -> "
                             f"{str(res.get('content') or '')[:limit]}")
                break
        if len(lines) >= 3:
            break
    lines.append(f"judge: {str(row.get('explanation') or '')[:300]}")
    return "\n".join(lines)


def parse_proposals(text: str) -> list[dict]:
    obj_text = _first_json_object(text or "")
    if not obj_text:
        raise ValueError("no JSON object")
    obj = json.loads(obj_text)
    patches = obj.get("patches") if isinstance(obj, dict) else None
    if not isinstance(patches, list):
        raise ValueError('expected {"patches": [...]}')
    return [p for p in patches if isinstance(p, dict)]


def propose_for_tool(tool: dict, excerpts: list[str], chat: Callable[[list[dict]], str],
                     tools: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """LLM patches for one tool, validated; one retry with the reasons. Returns (valid, rejections)."""
    name = _fn(tool)["name"]
    messages = [{"role": "system", "content": PROPOSER_SYSTEM},
                {"role": "user", "content": f"Tool spec:\n{json.dumps(_fn(tool), indent=1)}\n\n"
                                            "Failures:\n" + "\n\n".join(excerpts)}]
    rejections: list[str] = []
    for attempt in range(2):
        reply = chat(messages)
        try:
            patches = parse_proposals(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            problems = [f"invalid reply: {exc}"]
            patches = []
        else:
            problems = []
        valid = []
        for p in patches:
            if p.get("tool") != name:
                problems.append(f"patch for {p.get('tool')!r}, expected {name!r}")
                continue
            errs = validate_patch(p, tools)
            if errs:
                problems.append(f"{p.get('field')}: {'; '.join(errs)}")
            else:
                valid.append(p)
        rejections += problems
        if valid or attempt == 1:
            return valid, rejections
        messages += [{"role": "assistant", "content": reply or ""},
                     {"role": "user", "content": "Rejected: " + " | ".join(problems) + ". Reply with only the corrected JSON."}]
    return [], rejections


def propose(
    failures: list[dict],
    traces: dict[str, dict],
    tools: list[dict],
    ledger: list[dict],
    chat: Callable[[list[dict]], str] | None,
    round_: int | None = None,
    max_excerpts: int = 3,
) -> tuple[list[dict], list[str]]:
    """New ledger rows (status proposed) for the tool_fix failures, and rejection notes.

    Patches are made against the current effective tools (applied patches included).
    """
    current, _ = apply_patches(tools, ledger)
    by_name = tool_map(current)
    round_ = round_ or next_round(ledger)
    rows_fix = [r for r in failures if r.get("route") == "tool_fix"]
    proposals: dict[tuple[str, str], tuple[dict, str, list[str]]] = {}
    notes: list[str] = []

    copied = [str(r.get("simulation_id") or r.get("task_id")) for r in failures
              if "tool_example_copied" in r.get("labels", [])]
    if copied:
        for p in deterministic_patches(by_name):
            proposals[(p["tool"], p["field"])] = (p, "rules", copied)

    if chat is not None:
        per_tool: dict[str, list[tuple[dict, dict | None]]] = {}
        for r in rows_fix:
            trace = traces.get(str(r.get("simulation_id")))
            for name in failing_tools(r, trace):
                if name in by_name:
                    per_tool.setdefault(name, []).append((r, trace))
        for name, items in per_tool.items():
            ex = [excerpt(r, t, name) for r, t in items[:max_excerpts]]
            ids = [str(r.get("simulation_id") or r.get("task_id")) for r, _ in items]
            valid, rejected = propose_for_tool(by_name[name], ex, chat, by_name)
            notes += [f"{name}: {x}" for x in rejected]
            for p in valid:
                key = (p["tool"], p["field"])
                if key in proposals:  # LLM text replaces the placeholder-only patch
                    ids = list(dict.fromkeys(ids + proposals[key][2]))
                proposals[key] = (p, "llm", ids)

    new_rows = []
    known = {r["id"] for r in ledger}
    for (name, field), (p, source, ids) in proposals.items():
        row = make_row(p, get_field(by_name[name], field) or "", round_, source, ids)
        if row["id"] not in known:
            new_rows.append(row)
    return new_rows, notes


def patch_summary(rows: list[dict]) -> list[dict[str, Any]]:
    return [{k: r[k] for k in ("id", "round", "tool", "field", "source", "status")} for r in rows]

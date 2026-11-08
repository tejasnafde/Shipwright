"""Prompted JSON tool calling for models without native tool support (Tiny Aya).

Tiny Aya's chat template knows only system, user and assistant turns, and the
HF router (cohere provider) drops the ``tools`` parameter. So at the litellm
layer we:

1. put the tool schemas and a short instruction in the system message,
2. rewrite history: assistant tool calls become a ```json block, tool results
   become user turns,
3. parse the first JSON block of the reply back into a native ``tool_calls``
   entry, so tau2 sees a normal tool call.

The raw reply text and the parse status go into
``message.provider_specific_fields["shipwright"]``, which tau2 keeps in raw_data.
Training data for the student model must use this same format.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any

TRACE_FIELD = "shipwright"

INSTRUCTION = """You can use these tools:
{tools}

To call a tool, reply with exactly one block and nothing else:
```json
{{"name": "<tool name>", "arguments": {{<arguments>}}}}
```
Call one tool at a time. You will get the result in the next message.
If you do not need a tool, reply to the user in plain text.

Example:
User: Where is my order #W1234567?
Assistant:
```json
{{"name": "get_order_details", "arguments": {{"order_id": "#W1234567"}}}}
```"""

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def is_prompted_model(model: str) -> bool:
    return "tiny-aya" in model.lower()


def _compact_tool(tool: dict) -> dict:
    fn = tool.get("function", tool)
    return {
        "name": fn.get("name"),
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {}),
    }


def render_tools_prompt(tools: list[dict]) -> str:
    lines = "\n".join(json.dumps(_compact_tool(t)) for t in tools)
    return INSTRUCTION.format(tools=lines)


def format_tool_call(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return "```json\n" + json.dumps({"name": name, "arguments": arguments}) + "\n```"


def convert_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Rewrite an OpenAI-style history into system/user/assistant turns only."""
    tool_prompt = render_tools_prompt(tools)
    names_by_id: dict[str, str] = {}
    out: list[dict] = []
    added = False
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            if not added:
                content = f"{content}\n\n{tool_prompt}" if content else tool_prompt
                added = True
            out.append({"role": "system", "content": content})
        elif role == "assistant":
            parts = [content] if content else []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                names_by_id[tc.get("id", "")] = fn.get("name", "")
                parts.append(format_tool_call(fn.get("name", ""), fn.get("arguments", {})))
            out.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            name = names_by_id.get(m.get("tool_call_id", ""), "tool")
            out.append({"role": "user", "content": f"Tool result for {name}: {content}"})
        else:
            out.append({"role": role, "content": content})
    if not added:
        out.insert(0, {"role": "system", "content": tool_prompt})
    # The template expects alternating turns; merge consecutive user turns.
    merged: list[dict] = []
    for m in out:
        if merged and m["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(m)
    return merged


def _first_json_object(text: str) -> str | None:
    """The first balanced {...} span in text (string-aware)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc, quote = 0, False, False, ""
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    in_str = False
            elif c in "\"'":
                in_str, quote = True, c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _load(obj_text: str) -> Any:
    try:
        return json.loads(obj_text)
    except ValueError:
        pass
    try:  # trivial single-quote / Python-literal case
        return ast.literal_eval(obj_text)
    except (ValueError, SyntaxError):
        return None


def looks_like_tool_attempt(text: str | None) -> bool:
    if not text:
        return False
    return "```json" in text or bool(re.search(r"""["']name["']\s*:""", text))


def parse_tool_call(text: str | None) -> tuple[str, dict] | None:
    """(name, arguments) from the first JSON block in text, or None."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        obj_text = _first_json_object(cand)
        if not obj_text:
            continue
        obj = _load(obj_text)
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            args = _load(args)
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        return obj["name"], args
    return None


def convert_response(response: Any) -> None:
    """Turn a prompted JSON tool call in the reply into a native tool_calls entry (in place)."""
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    for choice in getattr(response, "choices", None) or []:
        msg = choice.message
        raw = msg.content
        parsed = parse_tool_call(raw)
        if parsed:
            name, args = parsed
            msg.tool_calls = [
                ChatCompletionMessageToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=Function(name=name, arguments=json.dumps(args)),
                )
            ]
            msg.content = None
            status = "tool_call"
        else:
            status = "parse_failed" if looks_like_tool_attempt(raw) else "text"
        fields = dict(getattr(msg, "provider_specific_fields", None) or {})
        fields[TRACE_FIELD] = {"raw_text": raw, "parse": status}
        msg.provider_specific_fields = fields


def prepare_request(kwargs: dict) -> dict:
    """Remove native tools from a completion request and prompt for them instead."""
    kwargs = dict(kwargs)
    tools = kwargs.pop("tools", None)
    kwargs.pop("tool_choice", None)
    kwargs["messages"] = convert_messages(kwargs["messages"], tools)
    return kwargs

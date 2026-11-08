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

from shipwright.context import REPLY_TOKENS, fit_messages

TRACE_FIELD = "shipwright"

INSTRUCTION = """You can use these tools (name(arguments) - what it does; ? marks an optional argument):
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


_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "object": "dict"}
_CHOICE = re.compile(r"either ('[^']*') or ('[^']*')")
_EXAMPLE = re.compile(r"such as (\'[^\']*\'|\"[^\"]*\")")


def _type_name(schema: dict) -> str:
    t = schema.get("type", "any")
    if t == "array":
        return f"list[{_type_name(schema.get('items') or {})}]"
    return _TYPES.get(t, str(t))


def _first_sentence(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    m = re.search(r"(.+?\.)(\s|$)", text)
    text = m.group(1) if m else text
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compact_tool_line(tool: dict) -> str:
    """One line per tool: name(arg: type e.g. 'x', opt?: type) - first sentence of the description."""
    fn = tool.get("function", tool)
    params = fn.get("parameters") or {}
    required = set(params.get("required") or [])
    args = []
    for name, schema in (params.get("properties") or {}).items():
        arg = f"{name}{'' if name in required else '?'}: {_type_name(schema)}"
        desc = schema.get("description") or ""
        ex = _EXAMPLE.search(desc)
        choice = _CHOICE.search(desc)
        if choice:
            arg += f" one of {choice.group(1)}|{choice.group(2)}"
        elif ex:
            arg += f" e.g. {ex.group(1)}"
        args.append(arg)
    return f"{fn.get('name')}({', '.join(args)}) - {_first_sentence(fn.get('description', ''))}"


def render_tools_prompt(tools: list[dict]) -> str:
    lines = "\n".join(compact_tool_line(t) for t in tools)
    return INSTRUCTION.format(tools=lines)


def minify_content(content: str) -> str:
    """Compact JSON (no indentation) if the content is JSON, else unchanged."""
    try:
        return json.dumps(json.loads(content), separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return content


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
            out.append({"role": "user", "content": f"Tool result for {name}: {minify_content(content)}"})
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


def convert_response(response: Any, context: dict | None = None) -> None:
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
        fields[TRACE_FIELD] = {"raw_text": raw, "parse": status, **(context or {})}
        msg.provider_specific_fields = fields


def prepare_request(kwargs: dict) -> tuple[dict, dict]:
    """Remove native tools, prompt for them instead, and fit the context window.

    Returns (new kwargs, context info for the trace).
    """
    kwargs = dict(kwargs)
    tools = kwargs.pop("tools", None)
    kwargs.pop("tool_choice", None)
    messages, info = fit_messages(convert_messages(kwargs["messages"], tools))
    kwargs["messages"] = messages
    kwargs["max_tokens"] = REPLY_TOKENS
    return kwargs, info

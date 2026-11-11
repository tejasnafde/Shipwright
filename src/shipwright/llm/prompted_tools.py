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
import hashlib
import json
import re
import threading
import uuid
from typing import Any

from shipwright.llm.context import REPLY_TOKENS, fit_messages

TRACE_FIELD = "shipwright"

INSTRUCTION = """You can use these tools (name(arguments) - what it does; ? marks an optional argument):
{tools}

To call a tool, reply with exactly one block and nothing else:
```json
{{"name": "<tool name>", "arguments": {{<arguments>}}}}
```
Call one tool at a time. You will get the result in the next message.
If you do not need a tool, reply to the user in plain text.

Only use values the customer or an earlier tool result gave you. Never make up an id.
If you do not have a value yet, ask the customer or look it up first.

How to handle a request:
1. Identify the customer first: ask for their email, or their name and zip code, and look up their user id. Do this even if they give you a user id.
2. Look up what you need with the tools. Take every id (order, item, product, payment method) from a tool result or from the customer.
3. Before any change (cancel, modify, return, exchange), tell the customer exactly what will change and wait for them to say yes.
4. If a tool returns an error, read it, then fix the arguments or ask the customer. Never repeat the same call.
5. Transfer to a human only when no tool can do what the customer asks.

Example (angle brackets stand for a real value from the conversation, not text to copy):
User: Where is my order <order id the customer wrote>?
Assistant:
```json
{{"name": "get_order_details", "arguments": {{"order_id": "<that order id>"}}}}
```"""

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


# Models that get prompted tools although their name has no "tiny-aya" (the teacher),
# with the reply budget they get. The teacher reasons before it answers, so it needs
# more than REPLY_TOKENS; its prompt budget stays the same as the student's.
_forced: dict[str, int] = {}
# The un-converted agent prompt (system message and tool schemas) of each request,
# by prompt_sha. Training data is rendered from it with convert_messages.
_prompts: dict[str, dict] = {}
_prompts_lock = threading.Lock()


def force_prompted(model: str, reply_tokens: int = REPLY_TOKENS) -> None:
    """Give ``model`` the same prompted-tools view as Tiny Aya (used for the teacher)."""
    _forced[model] = reply_tokens


def is_prompted_model(model: str) -> bool:
    return model in _forced or "tiny-aya" in model.lower()


def reply_tokens_for(model: str) -> int:
    return _forced.get(model, REPLY_TOKENS)


def prompt_sha(system: str, tools: list[dict]) -> str:
    serialized_prompt = json.dumps({"system": system, "tools": tools}, sort_keys=True)
    return hashlib.sha256(serialized_prompt.encode()).hexdigest()[:16]


def recorded_prompts() -> dict[str, dict]:
    """{prompt_sha: {"system": ..., "tools": [...]}} for every prompted request so far."""
    with _prompts_lock:
        return dict(_prompts)


_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "object": "dict"}
_CHOICE = re.compile(r"either ('[^']*') or ('[^']*')")
_EXAMPLE = re.compile(r"such as (\'[^\']*\'|\"[^\"]*\")")


def _type_name(schema: dict) -> str:
    json_type = schema.get("type", "any")
    if json_type == "array":
        return f"list[{_type_name(schema.get('items') or {})}]"
    return _TYPES.get(json_type, str(json_type))


def _first_sentence(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    sentence_match = re.search(r"(.+?\.)(\s|$)", text)
    if sentence_match:
        text = sentence_match.group(1)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# Tool patches (rows with status "applied" from patches/ledger.jsonl), set by baseline.py.
_tool_patches: list[dict] = []


def set_tool_patches(rows: list[dict]) -> None:
    global _tool_patches
    _tool_patches = [row for row in rows if row.get("status") == "applied"]


def _patched_fields(tool_name: str) -> set[str]:
    """Fields to show in full. Placeholder-only patches (source "rules") already reach
    the model through the compact "e.g." form, so they stay compact."""
    return {
        row["field"]
        for row in _tool_patches
        if row["tool"] == tool_name and row.get("source") != "rules"
    }


def compact_tool_line(tool: dict) -> str:
    """One line per tool: name(arg: type e.g. 'x', opt?: type) - first sentence of the description.

    A field changed by a tool patch is shown in full, so the patch reaches the model.
    """
    function = tool.get("function", tool)
    patched_fields = _patched_fields(function.get("name", ""))
    parameters = function.get("parameters") or {}
    required_names = set(parameters.get("required") or [])

    argument_texts = []
    for name, schema in (parameters.get("properties") or {}).items():
        optional_mark = "" if name in required_names else "?"
        argument_text = f"{name}{optional_mark}: {_type_name(schema)}"
        parameter_description = schema.get("description") or ""
        example_match = _EXAMPLE.search(parameter_description)
        choice_match = _CHOICE.search(parameter_description)
        if f"parameters.{name}.description" in patched_fields:
            argument_text += f" ({' '.join(parameter_description.split())})"
        elif choice_match:
            argument_text += f" one of {choice_match.group(1)}|{choice_match.group(2)}"
        elif example_match:
            argument_text += f" e.g. {example_match.group(1)}"
        argument_texts.append(argument_text)

    description = function.get("description", "")
    if "description" in patched_fields:
        description = " ".join(description.split())
    else:
        description = _first_sentence(description)
    return f"{function.get('name')}({', '.join(argument_texts)}) - {description}"


def render_tools_prompt(tools: list[dict]) -> str:
    tool_lines = "\n".join(compact_tool_line(tool) for tool in tools)
    return INSTRUCTION.format(tools=tool_lines)


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
    tool_name_by_call_id: dict[str, str] = {}
    converted_messages: list[dict] = []
    tool_prompt_added = False
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            if not tool_prompt_added:
                content = f"{content}\n\n{tool_prompt}" if content else tool_prompt
                tool_prompt_added = True
            converted_messages.append({"role": "system", "content": content})
        elif role == "assistant":
            content_parts = [content] if content else []
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                tool_name_by_call_id[tool_call.get("id", "")] = function.get("name", "")
                content_parts.append(format_tool_call(function.get("name", ""), function.get("arguments", {})))
            converted_messages.append({"role": "assistant", "content": "\n".join(content_parts)})
        elif role == "tool":
            tool_name = tool_name_by_call_id.get(message.get("tool_call_id", ""), "tool")
            converted_messages.append(
                {"role": "user", "content": f"Tool result for {tool_name}: {minify_content(content)}"}
            )
        else:
            converted_messages.append({"role": role, "content": content})
    if not tool_prompt_added:
        converted_messages.insert(0, {"role": "system", "content": tool_prompt})

    # The template expects alternating turns; merge consecutive user turns.
    merged_messages: list[dict] = []
    for message in converted_messages:
        previous_message = merged_messages[-1] if merged_messages else None
        if previous_message and message["role"] == "user" and previous_message["role"] == "user":
            merged_content = previous_message["content"] + "\n\n" + message["content"]
            merged_messages[-1] = {"role": "user", "content": merged_content}
        else:
            merged_messages.append(message)
    return merged_messages


def _first_json_object(text: str) -> str | None:
    """The first balanced {...} span in text (string-aware)."""
    start_index = text.find("{")
    while start_index != -1:
        brace_depth = 0
        inside_string = False
        previous_was_backslash = False
        quote_character = ""
        for index in range(start_index, len(text)):
            character = text[index]
            if inside_string:
                if previous_was_backslash:
                    previous_was_backslash = False
                elif character == "\\":
                    previous_was_backslash = True
                elif character == quote_character:
                    inside_string = False
            elif character in "\"'":
                inside_string = True
                quote_character = character
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : index + 1]
        start_index = text.find("{", start_index + 1)
    return None


def _load(object_text: str) -> Any:
    try:
        return json.loads(object_text)
    except ValueError:
        pass
    try:  # trivial single-quote / Python-literal case
        return ast.literal_eval(object_text)
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
    candidate_texts = [match.group(1) for match in _FENCE.finditer(text)] + [text]
    for candidate_text in candidate_texts:
        object_text = _first_json_object(candidate_text)
        if not object_text:
            continue
        parsed_object = _load(object_text)
        if not isinstance(parsed_object, dict) or not isinstance(parsed_object.get("name"), str):
            continue
        arguments = parsed_object.get("arguments", parsed_object.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = _load(arguments)
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return None
        return parsed_object["name"], arguments
    return None


def convert_response(response: Any, context: dict | None = None) -> None:
    """Turn a prompted JSON tool call in the reply into a native tool_calls entry (in place)."""
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    for choice in getattr(response, "choices", None) or []:
        message = choice.message
        raw_text = message.content
        parsed_call = parse_tool_call(raw_text)
        if parsed_call:
            tool_name, arguments = parsed_call
            message.tool_calls = [
                ChatCompletionMessageToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=Function(name=tool_name, arguments=json.dumps(arguments)),
                )
            ]
            message.content = None
            parse_status = "tool_call"
        elif looks_like_tool_attempt(raw_text):
            parse_status = "parse_failed"
        else:
            parse_status = "text"
        provider_fields = dict(getattr(message, "provider_specific_fields", None) or {})
        provider_fields[TRACE_FIELD] = {"raw_text": raw_text, "parse": parse_status, **(context or {})}
        message.provider_specific_fields = provider_fields


def prepare_request(kwargs: dict) -> tuple[dict, dict]:
    """Remove native tools, prompt for them instead, and fit the context window.

    Returns (new kwargs, context info for the trace).
    """
    kwargs = dict(kwargs)
    tools = kwargs.pop("tools", None) or []
    kwargs.pop("tool_choice", None)
    active_patch_ids = []
    if _tool_patches:
        from shipwright.heal.tool_fix import apply_patches

        tools, active_patch_ids = apply_patches(tools, _tool_patches)

    original_messages = kwargs["messages"]
    system_prompt = next(
        (message.get("content") or "" for message in original_messages if message.get("role") == "system"),
        "",
    )
    sha = prompt_sha(system_prompt, tools)
    with _prompts_lock:
        _prompts.setdefault(sha, {"system": system_prompt, "tools": tools})

    fitted_messages, info = fit_messages(convert_messages(original_messages, tools))
    info["prompt_sha"] = sha
    if _tool_patches:
        info["tool_patches"] = active_patch_ids
    kwargs["messages"] = fitted_messages
    kwargs["max_tokens"] = reply_tokens_for(str(kwargs.get("model", "")))
    return kwargs, info

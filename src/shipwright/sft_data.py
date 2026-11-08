"""Turn successful tau2 runs into SFT examples, rendered exactly as Tiny Aya sees them.

The student's view of a conversation is ``prompted_tools.convert_messages`` (system
prompt + compact tool list + JSON block protocol; tool calls as ```json blocks; tool
results as user turns), then ``context.fit_messages`` (the 8192-token budget). This
module calls the same two functions, so a training example is byte-for-byte the prompt
the student would get, plus the assistant turns as targets.

One example = one list of system/user/assistant messages. Every assistant message is a
training target (AgentTune maps it to a TEXT event); system and user messages are
context only (OBSERVATION events, loss-masked).

Long runs: a run whose full conversation does not fit the budget is either dropped
(``long_mode="drop"``) or cut into windows (``long_mode="window"``). A window is the
trimmed prompt the student would see before an assistant turn (``fit_messages``) plus
that turn; windows are chosen from the end so that each assistant turn is a target once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from shipwright.context import (
    CONTEXT_LIMIT,
    OMITTED_NOTE,
    REPLY_TOKENS,
    count_messages,
    count_tokens,
    fit_messages,
    prompt_budget,
)
from shipwright.prompted_tools import convert_messages


def trace_to_openai(messages: list[dict]) -> list[dict]:
    """tau2 message dicts (traces.jsonl ``messages``) -> the litellm messages tau2 sends.

    Uses tau2's own ``to_litellm_messages`` so the history matches the live request.
    """
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.utils.llm_utils import to_litellm_messages

    by_role = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    parsed = [by_role[m["role"]].model_validate(m) for m in messages if m.get("role") in by_role]
    return to_litellm_messages(parsed)


def render_conversation(system: str, tools: list[dict], history: list[dict]) -> list[dict]:
    """The student view of a whole run, ending at the last assistant turn."""
    conv = convert_messages([{"role": "system", "content": system}, *history], tools)
    while conv and conv[-1]["role"] != "assistant":
        conv.pop()
    return conv


def _assistant_indices(conv: list[dict]) -> list[int]:
    return [i for i, m in enumerate(conv) if m["role"] == "assistant"]


def _move_note_off_assistant(prompt: list[dict]) -> list[dict]:
    """fit_messages may put the omitted-turns note on an assistant turn; that turn would
    become a target that teaches the note. Drop that turn too and put the note on the
    next user turn (the view fit_messages gives when it drops one more message)."""
    i = 1 if prompt and prompt[0]["role"] == "system" else 0
    if (len(prompt) > i + 1 and prompt[i]["role"] == "assistant"
            and prompt[i]["content"].startswith(OMITTED_NOTE) and prompt[i + 1]["role"] == "user"):
        nxt = prompt[i + 1]
        return prompt[:i] + [{**nxt, "content": f"{OMITTED_NOTE}\n{nxt['content']}"}] + prompt[i + 2:]
    return prompt


def window_examples(conv: list[dict], budget: int | None = None) -> list[list[dict]]:
    """Cut a long conversation into windows, each the student's trimmed view plus one turn.

    The last window ends at the last assistant turn. Its kept history covers some earlier
    assistant turns; the next window ends at the latest assistant turn not yet covered.
    """
    budget = prompt_budget() if budget is None else budget
    targets = _assistant_indices(conv)
    covered: set[int] = set()
    windows: list[list[dict]] = []
    for end in reversed(targets):
        if end in covered:
            continue
        prompt, _ = fit_messages(conv[:end], budget)
        prompt = _move_note_off_assistant(prompt)
        window = prompt + [conv[end]]
        kept_assistant = sum(1 for m in prompt if m["role"] == "assistant")
        earlier = [i for i in targets if i < end]
        covered.update(earlier[len(earlier) - kept_assistant:] if kept_assistant else [])
        covered.add(end)
        windows.append(window)
    windows.reverse()
    return windows


def conversation_examples(conv: list[dict], long_mode: str = "window") -> list[list[dict]]:
    targets = _assistant_indices(conv)
    if not targets:
        return []
    # The prompt before the last turn is the longest one; if it fits, no turn was trimmed.
    if count_messages(conv[: targets[-1]]) <= prompt_budget():
        return [conv]
    if long_mode == "drop":
        return []
    if long_mode != "window":
        raise ValueError(f"unknown long_mode {long_mode!r}")
    return window_examples(conv)


def example_key(messages: list[dict]) -> str:
    blob = json.dumps([[m["role"], m["content"]] for m in messages], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def dedupe(records: Iterable[dict]) -> list[dict]:
    """Drop records whose messages are identical to an earlier record's."""
    seen: set[str] = set()
    out = []
    for r in records:
        key = example_key(r["messages"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def longest_reply_tokens(messages: list[dict]) -> int:
    return max((count_tokens(m["content"]) for m in messages if m["role"] == "assistant"), default=0)


def trace_records(trace: dict, prompt: dict, source: str, long_mode: str = "window") -> list[dict]:
    """SFT records for one trace. ``prompt`` is {"system", "tools"} from prompts.json."""
    conv = render_conversation(prompt["system"], prompt["tools"], trace_to_openai(trace["messages"]))
    out = []
    for i, messages in enumerate(conversation_examples(conv, long_mode)):
        out.append(
            {
                "messages": messages,
                "source": source,
                "task_id": str(trace.get("task_id")),
                "simulation_id": trace.get("simulation_id"),
                "window": i,
                "tokens": count_messages(messages),
                "reply_tokens_max": longest_reply_tokens(messages),
            }
        )
    return out


def filter_records(records: list[dict], max_tokens: int = CONTEXT_LIMIT,
                   max_reply_tokens: int = REPLY_TOKENS) -> tuple[list[dict], dict]:
    """Drop examples over the context limit, or with a reply the student could not emit."""
    kept, dropped = [], {"over_max_tokens": 0, "reply_over_max_tokens": 0}
    for r in records:
        if r["tokens"] > max_tokens:
            dropped["over_max_tokens"] += 1
        elif r["reply_tokens_max"] > max_reply_tokens:
            dropped["reply_over_max_tokens"] += 1
        else:
            kept.append(r)
    return kept, dropped


def stats(records: list[dict], dropped: dict, extra: dict | None = None) -> dict:
    toks = [r["tokens"] for r in records]
    by_source: dict[str, int] = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    return {
        "examples": len(records),
        "by_source": by_source,
        "tasks": sorted({r["task_id"] for r in records}, key=lambda x: (len(x), x)),
        "assistant_turns": sum(1 for r in records for m in r["messages"] if m["role"] == "assistant"),
        "avg_tokens": round(sum(toks) / len(toks), 1) if toks else 0,
        "max_tokens": max(toks, default=0),
        "dropped": dropped,
        **(extra or {}),
    }

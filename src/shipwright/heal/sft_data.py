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

from shipwright.llm.context import (
    CONTEXT_LIMIT,
    OMITTED_NOTE,
    REPLY_TOKENS,
    count_messages,
    count_tokens,
    fit_messages,
    prompt_budget,
)
from shipwright.llm.prompted_tools import convert_messages


def trace_to_openai(messages: list[dict]) -> list[dict]:
    """tau2 message dicts (traces.jsonl ``messages``) -> the litellm messages tau2 sends.

    Uses tau2's own ``to_litellm_messages`` so the history matches the live request.
    """
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.utils.llm_utils import to_litellm_messages

    message_class_by_role = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    tau2_messages = [
        message_class_by_role[message["role"]].model_validate(message)
        for message in messages
        if message.get("role") in message_class_by_role
    ]
    return to_litellm_messages(tau2_messages)


def render_conversation(system: str, tools: list[dict], history: list[dict]) -> list[dict]:
    """The student view of a whole run, ending at the last assistant turn."""
    conversation = convert_messages([{"role": "system", "content": system}, *history], tools)
    while conversation and conversation[-1]["role"] != "assistant":
        conversation.pop()
    return conversation


def _assistant_indices(conversation: list[dict]) -> list[int]:
    return [index for index, message in enumerate(conversation) if message["role"] == "assistant"]


def _move_note_off_assistant(prompt: list[dict]) -> list[dict]:
    """fit_messages may put the omitted-turns note on an assistant turn; that turn would
    become a target that teaches the note. Drop that turn too and put the note on the
    next user turn (the view fit_messages gives when it drops one more message)."""
    first_turn_index = 1 if prompt and prompt[0]["role"] == "system" else 0
    has_note_on_assistant = (
        len(prompt) > first_turn_index + 1
        and prompt[first_turn_index]["role"] == "assistant"
        and prompt[first_turn_index]["content"].startswith(OMITTED_NOTE)
        and prompt[first_turn_index + 1]["role"] == "user"
    )
    if not has_note_on_assistant:
        return prompt
    next_user_message = prompt[first_turn_index + 1]
    user_message_with_note = {**next_user_message, "content": f"{OMITTED_NOTE}\n{next_user_message['content']}"}
    return prompt[:first_turn_index] + [user_message_with_note] + prompt[first_turn_index + 2:]


def window_examples(conversation: list[dict], budget: int | None = None) -> list[list[dict]]:
    """Cut a long conversation into windows, each the student's trimmed view plus one turn.

    The last window ends at the last assistant turn. Its kept history covers some earlier
    assistant turns; the next window ends at the latest assistant turn not yet covered.
    """
    if budget is None:
        budget = prompt_budget()
    target_indices = _assistant_indices(conversation)
    covered_indices: set[int] = set()
    windows: list[list[dict]] = []
    for end_index in reversed(target_indices):
        if end_index in covered_indices:
            continue
        prompt, _ = fit_messages(conversation[:end_index], budget)
        prompt = _move_note_off_assistant(prompt)
        windows.append(prompt + [conversation[end_index]])

        kept_assistant_count = sum(1 for message in prompt if message["role"] == "assistant")
        earlier_targets = [index for index in target_indices if index < end_index]
        if kept_assistant_count:
            covered_indices.update(earlier_targets[len(earlier_targets) - kept_assistant_count:])
        covered_indices.add(end_index)
    windows.reverse()
    return windows


def conversation_examples(conversation: list[dict], long_mode: str = "window") -> list[list[dict]]:
    target_indices = _assistant_indices(conversation)
    if not target_indices:
        return []
    # The prompt before the last turn is the longest one; if it fits, no turn was trimmed.
    if count_messages(conversation[: target_indices[-1]]) <= prompt_budget():
        return [conversation]
    if long_mode == "drop":
        return []
    if long_mode != "window":
        raise ValueError(f"unknown long_mode {long_mode!r}")
    return window_examples(conversation)


def example_key(messages: list[dict]) -> str:
    serialized_messages = json.dumps(
        [[message["role"], message["content"]] for message in messages], ensure_ascii=False
    )
    return hashlib.sha256(serialized_messages.encode()).hexdigest()


def dedupe(records: Iterable[dict]) -> list[dict]:
    """Drop records whose messages are identical to an earlier record's."""
    seen_keys: set[str] = set()
    unique_records = []
    for record in records:
        key = example_key(record["messages"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_records.append(record)
    return unique_records


def longest_reply_tokens(messages: list[dict]) -> int:
    reply_token_counts = [count_tokens(message["content"]) for message in messages if message["role"] == "assistant"]
    return max(reply_token_counts, default=0)


def trace_records(trace: dict, prompt: dict, source: str, long_mode: str = "window") -> list[dict]:
    """SFT records for one trace. ``prompt`` is {"system", "tools"} from prompts.json."""
    conversation = render_conversation(prompt["system"], prompt["tools"], trace_to_openai(trace["messages"]))
    records = []
    for window_number, messages in enumerate(conversation_examples(conversation, long_mode)):
        records.append({
            "messages": messages,
            "source": source,
            "task_id": str(trace.get("task_id")),
            "simulation_id": trace.get("simulation_id"),
            "window": window_number,
            "tokens": count_messages(messages),
            "reply_tokens_max": longest_reply_tokens(messages),
        })
    return records


def filter_records(
    records: list[dict],
    max_tokens: int = CONTEXT_LIMIT,
    max_reply_tokens: int = REPLY_TOKENS,
) -> tuple[list[dict], dict]:
    """Drop examples over the context limit, or with a reply the student could not emit."""
    kept_records = []
    dropped_counts = {"over_max_tokens": 0, "reply_over_max_tokens": 0}
    for record in records:
        if record["tokens"] > max_tokens:
            dropped_counts["over_max_tokens"] += 1
        elif record["reply_tokens_max"] > max_reply_tokens:
            dropped_counts["reply_over_max_tokens"] += 1
        else:
            kept_records.append(record)
    return kept_records, dropped_counts


def stats(records: list[dict], dropped: dict, extra: dict | None = None) -> dict:
    token_counts = [record["tokens"] for record in records]
    example_count_by_source: dict[str, int] = {}
    for record in records:
        source = record["source"]
        example_count_by_source[source] = example_count_by_source.get(source, 0) + 1

    task_ids = {record["task_id"] for record in records if record["task_id"] is not None}
    assistant_turn_count = sum(
        1 for record in records for message in record["messages"] if message["role"] == "assistant"
    )
    if token_counts:
        average_tokens = round(sum(token_counts) / len(token_counts), 1)
    else:
        average_tokens = 0
    return {
        "examples": len(records),
        "by_source": example_count_by_source,
        "tasks": sorted(task_ids, key=lambda task_id: (len(task_id), task_id)),
        "assistant_turns": assistant_turn_count,
        "avg_tokens": average_tokens,
        "max_tokens": max(token_counts, default=0),
        "dropped": dropped,
        **(extra or {}),
    }

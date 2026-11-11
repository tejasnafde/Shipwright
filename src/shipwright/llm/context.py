"""Keep a prompted Tiny Aya request inside its 8192-token context.

Tiny Aya: max_position_embeddings 8192; the cohere provider rejects anything
over 8192 tokens (prompt plus reply). We reserve REPLY_TOKENS for the reply,
and when the prompt is over budget we drop the oldest middle turns (keeping the
system message, the first user message and the latest turns). If it is still
over, we truncate the latest message with a "[truncated N chars]" marker.
"""

from __future__ import annotations

import math
import threading
from typing import Callable

STUDENT_REPO = "CohereLabs/tiny-aya-global"
CONTEXT_LIMIT = 8192
REPLY_TOKENS = 700
PER_MESSAGE_TOKENS = 8  # chat template overhead per turn
CHARS_PER_TOKEN = 3.5
ESTIMATE_MARGIN = 400  # extra safety when we only estimate
TOKENIZER_MARGIN = 100
OMITTED_NOTE = "[earlier turns omitted]"

_lock = threading.Lock()
_encode: Callable[[str], int] | None = None


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def load_tokenizer(repo_id: str, token: str | None) -> bool:
    """Use the real tokenizer if tokenizer.json downloads; else keep the estimate."""
    global _encode
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(repo_id, "tokenizer.json", token=token)
        tok = Tokenizer.from_file(path)
    except Exception:  # noqa: BLE001 - any failure means: estimate
        return False
    with _lock:
        _encode = lambda text: len(tok.encode(text, add_special_tokens=False).ids)  # noqa: E731
    return True


def count_method() -> str:
    return "tokenizer" if _encode else "estimate"


def count_tokens(text: str) -> int:
    return _encode(text) if _encode else estimate_tokens(text)


def prompt_budget() -> int:
    margin = TOKENIZER_MARGIN if _encode else ESTIMATE_MARGIN
    return CONTEXT_LIMIT - REPLY_TOKENS - margin


def count_messages(messages: list[dict]) -> int:
    return sum(count_tokens(message.get("content") or "") + PER_MESSAGE_TOKENS for message in messages)


def _merge_same_role(messages: list[dict]) -> list[dict]:
    merged_messages: list[dict] = []
    for message in messages:
        previous_message = merged_messages[-1] if merged_messages else None
        if previous_message and message["role"] == previous_message["role"] and message["role"] != "system":
            merged_content = previous_message["content"] + "\n\n" + message["content"]
            merged_messages[-1] = {"role": message["role"], "content": merged_content}
        else:
            merged_messages.append(dict(message))
    return merged_messages


def fit_messages(messages: list[dict], budget: int | None = None) -> tuple[list[dict], dict]:
    """Return (messages that fit the budget, info about what was trimmed)."""
    budget = prompt_budget() if budget is None else budget
    info = {"prompt_tokens_est": count_messages(messages), "dropped_messages": 0,
            "truncated_chars": 0, "token_count_method": count_method(), "budget": budget}
    if info["prompt_tokens_est"] <= budget:
        return messages, info

    head = 0
    if messages and messages[0]["role"] == "system":
        head = 1
    if len(messages) > head and messages[head]["role"] == "user":
        head += 1
    pinned, middle = messages[:head], list(messages[head:])

    def assemble(rest: list[dict], dropped: int) -> list[dict]:
        if dropped and rest:
            rest = [{**rest[0], "content": f"{OMITTED_NOTE}\n{rest[0]['content']}"}] + rest[1:]
        return _merge_same_role(pinned + rest)

    last_pinned_role = pinned[-1]["role"] if pinned else None

    def drop_one() -> int:
        dropped_count = 1
        middle.pop(0)
        # Keep turns alternating: do not start the kept turns with the pinned role.
        while len(middle) > 1 and middle[0]["role"] == last_pinned_role:
            middle.pop(0)
            dropped_count += 1
        return dropped_count

    dropped = 0
    fitted_messages = assemble(middle, 0)
    while len(middle) > 1 and count_messages(fitted_messages) > budget:
        dropped += drop_one()
        fitted_messages = assemble(middle, dropped)

    tokens_over_budget = count_messages(fitted_messages) - budget
    if tokens_over_budget > 0 and middle:
        last_message = middle[-1]
        content = last_message["content"]
        # Characters to cut: over-budget tokens, at a low chars-per-token rate, then grow.
        chars_to_cut = min(len(content), math.ceil(tokens_over_budget * 2.5) + 40)
        while True:
            chars_to_keep = len(content) - chars_to_cut
            truncated_content = f"{content[:chars_to_keep]} [truncated {chars_to_cut} chars]"
            trial_middle = middle[:-1] + [{**last_message, "content": truncated_content}]
            candidate_messages = assemble(trial_middle, dropped)
            if count_messages(candidate_messages) <= budget or chars_to_keep == 0:
                fitted_messages = candidate_messages
                info["truncated_chars"] = chars_to_cut
                break
            chars_to_cut = min(len(content), chars_to_cut + max(200, chars_to_cut // 4))
    info["dropped_messages"] = dropped
    info["prompt_tokens_est"] = count_messages(fitted_messages)
    return fitted_messages, info

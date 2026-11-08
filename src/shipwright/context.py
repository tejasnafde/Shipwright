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
    return sum(count_tokens(m.get("content") or "") + PER_MESSAGE_TOKENS for m in messages)


def _merge_same_role(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if out and m["role"] == out[-1]["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


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
        n = 1
        middle.pop(0)
        # Keep turns alternating: do not start the kept turns with the pinned role.
        while len(middle) > 1 and middle[0]["role"] == last_pinned_role:
            middle.pop(0)
            n += 1
        return n

    dropped = 0
    out = assemble(middle, 0)
    while len(middle) > 1 and count_messages(out) > budget:
        dropped += drop_one()
        out = assemble(middle, dropped)

    over = count_messages(out) - budget
    if over > 0 and middle:
        last = middle[-1]
        content = last["content"]
        # Characters to cut: over-budget tokens, at a low chars-per-token rate, then grow.
        cut = min(len(content), math.ceil(over * 2.5) + 40)
        while True:
            keep = len(content) - cut
            trial = middle[:-1] + [{**last, "content": f"{content[:keep]} [truncated {cut} chars]"}]
            candidate = assemble(trial, dropped)
            if count_messages(candidate) <= budget or keep == 0:
                out = candidate
                info["truncated_chars"] = cut
                break
            cut = min(len(content), cut + max(200, cut // 4))
    info["dropped_messages"] = dropped
    info["prompt_tokens_est"] = count_messages(out)
    return out, info

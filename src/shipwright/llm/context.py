"""Keep a prompted Tiny Aya request inside its 8192-token context.

Tiny Aya: max_position_embeddings 8192; the cohere provider rejects anything
over 8192 tokens (prompt plus reply). We reserve REPLY_TOKENS for the reply,
and when the prompt is over budget we drop the oldest middle turns (keeping the
system message, the first user message and the latest turns). If it is still
over, we truncate the latest message with a "[truncated N chars]" marker.
"""

from __future__ import annotations

import json
import math
import re
import threading
from typing import Callable

STUDENT_REPO = "CohereLabs/tiny-aya-global"
CONTEXT_LIMIT = 8192
# 700 until 2026-11-14. Only 1 to 2% of student replies exceed 450 tokens (gate 8), and every token reserved here is
# taken from the conversation on every turn: 512 gives the history about 190 more tokens.
REPLY_TOKENS = 512
PER_MESSAGE_TOKENS = 8  # chat template overhead per turn
CHARS_PER_TOKEN = 3.5
ESTIMATE_MARGIN = 400  # extra safety when we only estimate
# 100 was too small: a round 1 request was rejected 18 tokens over 8192 (chat template tokens we do
# not count). 2026-11-12.
TOKENIZER_MARGIN = 250
OMITTED_NOTE = "[earlier turns omitted]"

_lock = threading.Lock()
_encode: Callable[[str], int] | None = None

# The identity note (still_true_note) is off by default (2026-11-15): round 16 was gated and promoted without it, so the
# serving harness stays what was gated. A system turns it on with ``identity_note: true`` (baseline --identity-note).
_identity_note = False


def set_identity_note(enabled: bool) -> None:
    global _identity_note
    _identity_note = enabled


def identity_note_enabled() -> bool:
    return _identity_note


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


# "Tool result for <tool>: <json>" is how convert_messages shows a tool result to the student. Keep the newest few as
# they are: the agent is working from them.
TOOL_RESULT = re.compile(r"Tool result for (\w+): (\{.*\})")
RESULTS_KEPT_WHOLE = 2


def shortened_result(tool_name: str, result: dict) -> dict:
    """The same tool result with the parts the agent does not act on left out.

    A product listing is up to about 1k tokens and half of it is variants that cannot be ordered; with a prompt that
    already takes 3.4k of the 8.2k window, three lookups used to push the customer's details out (gate 7: 10 of 60
    runs lost history, gate 8: 14). Only an older result is shortened, and only when the request is over budget."""
    if tool_name == "get_product_details" and isinstance(result.get("variants"), dict):
        available = {item_id: {"options": variant.get("options"), "price": variant.get("price")}
                     for item_id, variant in result["variants"].items() if variant.get("available")}
        return {"name": result.get("name"), "product_id": result.get("product_id"), "available_variants": available,
                "unavailable_variants": len(result["variants"]) - len(available), "shortened": True}
    if tool_name == "get_order_details" and "items" in result:
        return {**{key: result.get(key) for key in ("order_id", "user_id", "status", "items", "payment_history")},
                "shortened": True}
    if tool_name == "get_user_details" and "payment_methods" in result:
        return {"user_id": result.get("user_id"), "name": result.get("name"), "orders": result.get("orders"),
                "payment_methods": list(result["payment_methods"]), "shortened": True}
    return result


def shorten_old_tool_results(messages: list[dict], budget: int) -> tuple[list[dict], int]:
    """Shorten tool results, oldest first and never the newest RESULTS_KEPT_WHOLE, until the messages fit.
    Returns (messages, how many results were shortened)."""
    spots = [(index, match) for index, message in enumerate(messages) if message["role"] == "user"
             for match in TOOL_RESULT.finditer(message["content"])]
    shortened_count = 0
    messages = [dict(message) for message in messages]
    for index, match in spots[:-RESULTS_KEPT_WHOLE] if RESULTS_KEPT_WHOLE else spots:
        if count_messages(messages) <= budget:
            break
        try:
            result = json.loads(match.group(2))
        except ValueError:
            continue
        if not isinstance(result, dict) or result.get("shortened"):
            continue
        replacement = shortened_result(match.group(1), result)
        if replacement is result:
            continue
        text = f"Tool result for {match.group(1)}: " + json.dumps(replacement, separators=(",", ":"), ensure_ascii=False)
        content = messages[index]["content"]
        messages[index]["content"] = content.replace(match.group(0), text, 1)
        shortened_count += 1
    return messages, shortened_count


OLD_REPLY_CHARS = 350
REPLIES_KEPT_WHOLE = 2


def shorten_old_replies(messages: list[dict], budget: int) -> tuple[list[dict], int]:
    """Cut the agent's older replies to their first OLD_REPLY_CHARS characters (oldest first, never the newest
    REPLIES_KEPT_WHOLE), until the messages fit. A tool call (a ```json block) is never cut: the next turn reads it as
    the agent's own history. Gate 8: the agent's replies were 27% of the requests that still did not fit."""
    replies = [index for index, message in enumerate(messages)
               if message["role"] == "assistant" and "```json" not in message["content"]
               and len(message["content"]) > OLD_REPLY_CHARS]
    messages = [dict(message) for message in messages]
    shortened_count = 0
    for index in replies[:-REPLIES_KEPT_WHOLE] if REPLIES_KEPT_WHOLE else replies:
        if count_messages(messages) <= budget:
            break
        messages[index]["content"] = messages[index]["content"][:OLD_REPLY_CHARS] + " [shortened]"
        shortened_count += 1
    return messages, shortened_count


AUTHENTICATION_RESULT = re.compile(r"Tool result for find_user_id_by_\w+: \"?([A-Za-z0-9_]+)\"?")


def still_true_note(dropped: list[dict]) -> str:
    """What the agent must not lose when old turns are dropped: the authenticated user id and that user's order ids.

    Round 13 (2026-11-15): once the turn holding the find_user_id result was dropped, the agent asked a customer it had
    already identified for their identity again (16 of 60 gate runs; 13 of 15 runs that lost history ended with the
    customer quitting), and 52 of the 176 correction rows taught that, because the teacher answered the same trimmed
    view. The note goes after OMITTED_NOTE, in live requests and training windows alike (both use fit_messages)."""
    user_id = None
    order_ids: list[str] = []
    for message in dropped:
        if message["role"] != "user":
            continue
        for match in AUTHENTICATION_RESULT.finditer(message["content"]):
            user_id = match.group(1)
        for match in TOOL_RESULT.finditer(message["content"]):
            if match.group(1) != "get_user_details":
                continue
            try:
                record = json.loads(match.group(2))
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("orders"), list):
                order_ids = [str(order_id) for order_id in record["orders"]]
    if not user_id:
        return ""
    note = f" Still true: the customer is already authenticated as user id {user_id}"
    if order_ids:
        note += f" (orders {', '.join(order_ids)})"
    return note + "; do not ask for their identity again."


def fit_messages(messages: list[dict], budget: int | None = None) -> tuple[list[dict], dict]:
    """Return (messages that fit the budget, info about what was trimmed)."""
    budget = prompt_budget() if budget is None else budget
    info = {"prompt_tokens_est": count_messages(messages), "dropped_messages": 0, "results_shortened": 0,
            "truncated_chars": 0, "token_count_method": count_method(), "budget": budget}
    if info["prompt_tokens_est"] <= budget:
        return messages, info
    messages, info["results_shortened"] = shorten_old_tool_results(messages, budget)
    if count_messages(messages) > budget:
        messages, replies_shortened = shorten_old_replies(messages, budget)
        info["results_shortened"] += replies_shortened
    if count_messages(messages) <= budget:
        info["prompt_tokens_est"] = count_messages(messages)
        return messages, info

    head = 0
    if messages and messages[0]["role"] == "system":
        head = 1
    if len(messages) > head and messages[head]["role"] == "user":
        head += 1
    pinned, middle = messages[:head], list(messages[head:])

    all_middle = list(middle)

    def assemble(rest: list[dict], dropped: int) -> list[dict]:
        if dropped and rest:
            note = OMITTED_NOTE
            if _identity_note:
                note += still_true_note(all_middle[:len(all_middle) - len(rest)])
            rest = [{**rest[0], "content": f"{note}\n{rest[0]['content']}"}] + rest[1:]
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

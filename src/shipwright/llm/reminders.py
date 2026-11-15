"""Per-turn status reminders for the student: a short note at the end of what the model reads.

Why: Tiny Aya reads the policy and the "How to handle a request" checklist once, at the top of a
2,700-token system prompt, and then mostly ignores it (gate 8 base: the first tool call was an order
lookup, a change or a transfer in 35 of 60 runs; 34 of 54 failures never made the change the
customer asked for). The guards (guards.py) only react after a wrong call, which costs a turn. A
reminder says, right before the model answers, where the conversation stands and what comes next.

The note is added to the last message of the request sent to the model only. tau2's conversation,
the trace's message list and the training data never contain it; the trace records it per turn
(``reminder`` in the agent turn's context info).

Off by default: ``scripts/baseline.py --state-reminders`` (system key ``state_reminders: true``).
Adapters trained before 2026-11-14 never saw a reminder, so gate them with and without it.
"""

from __future__ import annotations

import re

from shipwright.llm import guards

_enabled = False

# An agent text turn that asks the customer to approve a change.
CONFIRMATION_QUESTION = re.compile(
    r"\b(proceed|confirm|shall i|should i|would you like me to|do you want me to|go ahead|is that correct|"
    r"is this correct|are you sure|ok to)\b",
    re.IGNORECASE,
)
ERROR_PREVIEW_CHARACTERS = 120


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    return _enabled


def authenticated_user_id(messages: list[dict]) -> str | None:
    """The user id from the latest successful find_user_id_* call, or None."""
    user_id = None
    for earlier in guards.earlier_calls(messages):
        is_lookup = earlier.call.name.startswith(guards.AUTHENTICATION_TOOL_PREFIX)
        if is_lookup and earlier.result and not earlier.result.startswith("Error"):
            user_id = earlier.result.strip().strip('"')
    return user_id


def latest_tool_error(messages: list[dict]) -> str | None:
    """The error text if the very last message is a failed tool result."""
    if not messages or messages[-1].get("role") != "tool":
        return None
    content = (messages[-1].get("content") or "").strip()
    if content.startswith("Error"):
        return content[:ERROR_PREVIEW_CHARACTERS]
    return None


def customer_just_confirmed_a_change(messages: list[dict]) -> bool:
    """The latest customer message says yes to a change the agent proposed in its last text turn,
    and the agent has not made a tool call since."""
    if not guards.has_explicit_confirmation(messages):
        return False
    customer_turns = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    latest_customer_turn = customer_turns[-1]
    for message in messages[latest_customer_turn + 1:]:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return False
    agent_text_turns = [message for message in messages[:latest_customer_turn] if message.get("role") == "assistant"]
    question = agent_text_turns[-1].get("content") or ""
    return "?" in question and bool(CONFIRMATION_QUESTION.search(question))


def status_note(messages: list[dict]) -> str | None:
    """The reminder for the next agent turn, from the OpenAI-format conversation so far, or None."""
    if not any(message.get("role") == "user" for message in messages):
        return None
    lines = []
    user_id = authenticated_user_id(messages)
    if user_id is None:
        if guards.tried_to_authenticate(messages):
            lines.append("The customer is not identified yet: the last lookup found no user. Ask them to check "
                         "their email, or for their first name, last name and zip code.")
        else:
            lines.append("The customer is not identified yet. First ask for their email, or their first name, "
                         "last name and zip code, and look up their user id. Do nothing else before that.")
    else:
        lines.append(f"The customer is identified (user id {user_id}).")
    error = latest_tool_error(messages)
    if error:
        lines.append(f"Your last tool call failed: \"{error}\". Fix the arguments using earlier tool results, "
                     f"or ask the customer. Do not repeat the same call.")
    if user_id is not None and customer_just_confirmed_a_change(messages):
        lines.append("The customer just said yes to the change you described. Make that tool call now, with "
                     "exactly the details you listed.")
    return "[Status, for you only: " + " ".join(lines) + "]"


def add_to_request(converted_messages: list[dict], note: str) -> list[dict]:
    """Append the note to the last user-role message (a customer turn or a tool result)."""
    if not converted_messages or converted_messages[-1].get("role") != "user":
        return converted_messages
    last_message = dict(converted_messages[-1])
    last_message["content"] = f"{last_message.get('content') or ''}\n\n{note}"
    return [*converted_messages[:-1], last_message]

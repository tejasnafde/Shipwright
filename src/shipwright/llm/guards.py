"""Tool-call guards: policy checks on the agent's tool call before tau2 runs it.

This is the "add validation" half of the tool-fix branch (docs/system-design.html): the
round 1 baseline showed that tool-description patches alone do not stop Tiny Aya from
making up ids, skipping authentication, changing orders without a "yes", transferring to
a human at the first message, or repeating a call that already failed.

Each guard reads the conversation so far (OpenAI-format messages, as tau2 sends them) and
the tool call the agent just wrote. A violation comes back as a ``GuardViolation`` with a
short message for the agent. ``shipwright.llm.patches`` then asks the agent again with that
message (a few times at most), so tau2 only ever sees the corrected call. The rejected
replies are kept in the trace, never in the conversation, so training data stays clean.

The rules come from the tau2 retail policy, quoted in each guard below.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

AUTHENTICATION_TOOL_PREFIX = "find_user_id"
TRANSFER_TOOL = "transfer_to_human_agents"
READ_TOOL_PREFIXES = ("get_", "find_", "list_", "calculate", "search_", "think")
AFFIRMATIVE_ANSWER = re.compile(
    r"\b(yes|yeah|yep|yup|confirm|confirmed|go ahead|proceed|sure|correct|please do|ok|okay|do it|that's right)\b",
    re.IGNORECASE,
)
# How often the agent is asked again after a violation. After that the call runs as written,
# so a guard that is wrong for one task can slow the agent down but never block it for good.
MAX_GUARD_RETRIES = 2

# Off by default: a run turns the guards on with scripts/baseline.py --tool-guards (a system config
# with ``tool_guards: true`` passes it), so the gate can compare a system with and without them.
_enabled = False


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    return _enabled


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class GuardViolation:
    rule: str
    message: str
    # What the agent says to the customer instead, if it still makes the call after being asked again. Only for rules
    # where running the call is worse than moving the conversation on (a repeated lookup loops; a forbidden exchange
    # breaks the policy). The other rules let the call run, so a wrong guard slows the agent but never blocks it.
    fallback_reply: str | None = None


def is_write_tool(tool_name: str) -> bool:
    if tool_name == TRANSFER_TOOL:
        return False
    return not tool_name.startswith(READ_TOOL_PREFIXES)


def tool_call_from_message(tool_call: dict) -> ToolCall:
    """A tool call as tau2 sends it to litellm ({"function": {"name", "arguments": "<json>"}})."""
    function = tool_call.get("function") or tool_call
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(name=function.get("name") or "", arguments=arguments)


@dataclass
class EarlierCall:
    call: ToolCall
    result: str


def earlier_calls(messages: list[dict]) -> list[EarlierCall]:
    """Every tool call the agent made so far, with the text of its result."""
    result_by_call_id = {}
    for message in messages:
        if message.get("role") == "tool":
            result_by_call_id[message.get("tool_call_id", "")] = message.get("content") or ""
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            result = result_by_call_id.get(tool_call.get("id", ""), "")
            calls.append(EarlierCall(call=tool_call_from_message(tool_call), result=result))
    return calls


def evidence_text(messages: list[dict]) -> str:
    """Everything the customer said and every tool result: where a real id can come from."""
    parts = []
    for message in messages:
        if message.get("role") in ("user", "tool"):
            parts.append(message.get("content") or "")
    return "\n".join(parts)


def id_values(arguments: dict) -> list[tuple[str, str]]:
    """(argument name, value) for every id argument: names ending in _id or _ids."""
    values = []
    for name, value in arguments.items():
        if not (name.endswith("_id") or name.endswith("_ids")):
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and item.strip():
                values.append((name, item.strip()))
    return values


def is_authenticated(messages: list[dict]) -> bool:
    for earlier in earlier_calls(messages):
        is_lookup = earlier.call.name.startswith(AUTHENTICATION_TOOL_PREFIX)
        if is_lookup and earlier.result and not earlier.result.startswith("Error"):
            return True
    return False


def tried_to_authenticate(messages: list[dict]) -> bool:
    """The agent called an authentication tool at least once, whatever the result."""
    for earlier in earlier_calls(messages):
        if earlier.call.name.startswith(AUTHENTICATION_TOOL_PREFIX):
            return True
    return False


def has_explicit_confirmation(messages: list[dict]) -> bool:
    """The customer's latest message says yes, and it answers a text turn of the agent."""
    customer_turns = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if not customer_turns:
        return False
    latest = customer_turns[-1]
    if not AFFIRMATIVE_ANSWER.search(messages[latest].get("content") or ""):
        return False
    previous_agent_turns = [message for message in messages[:latest] if message.get("role") == "assistant"]
    if not previous_agent_turns:
        return False
    asked_in_text = bool((previous_agent_turns[-1].get("content") or "").strip())
    return asked_in_text and not previous_agent_turns[-1].get("tool_calls")


# ---- the guards ----


def check_made_up_ids(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: "You should not make up any information ... not provided by the user or the tools"."""
    evidence = evidence_text(messages)
    for argument_name, value in id_values(call.arguments):
        if value.lstrip("#") not in evidence:
            return GuardViolation(
                rule="made_up_id",
                message=(f"{argument_name} '{value}' does not appear in anything the customer said or any tool "
                         f"result. Do not guess ids. Ask the customer, or look the id up with a tool first "
                         f"(for example the user's details list their orders and payment methods, an order's "
                         f"details list its items, a product's details list its variant item ids)."),
            )
    return None


def check_authentication_first(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: "At the beginning of the conversation, you have to authenticate the user identity by
    locating their user id via email, or via name + zip code. This has to be done even when the user
    already provides the user id." (The transfer guard handles transfers.)"""
    if call.name.startswith(AUTHENTICATION_TOOL_PREFIX) or call.name == TRANSFER_TOOL:
        return None
    if call.name.startswith(("list_", "calculate", "think")):
        return None
    if is_authenticated(messages):
        return None
    return GuardViolation(
        rule="not_authenticated",
        message=("The customer is not authenticated yet. First ask for their email, or their first name, last "
                 "name and zip code, and find their user id with find_user_id_by_email or "
                 "find_user_id_by_name_zip. Do this even if they already gave a user id."),
    )


def check_confirmation_before_change(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: "Before taking any action that updates the database (cancel, modify, return, exchange),
    you must list the action details and obtain explicit user confirmation (yes) to proceed." """
    if not is_write_tool(call.name):
        return None
    if has_explicit_confirmation(messages):
        return None
    return GuardViolation(
        rule="no_confirmation",
        message=(f"{call.name} changes the customer's data. First tell the customer exactly what you will do "
                 f"(the order, the items, the payment method, any refund or price difference) and ask them to "
                 f"confirm with yes. Make this call only after they say yes."),
    )


def check_early_transfer(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: "You should transfer the user to a human agent if and only if the request cannot be
    handled within the scope of your actions." Before the agent has even tried to identify the
    customer, nothing has been tried yet. After a failed attempt (the customer cannot be found),
    a transfer can be right, so it is allowed."""
    if call.name != TRANSFER_TOOL:
        return None
    if tried_to_authenticate(messages):
        return None
    return GuardViolation(
        rule="early_transfer",
        message=("Do not transfer yet: you can handle most requests yourself with the tools. Authenticate the "
                 "customer first (email, or name and zip code), look up their order, and try to solve the "
                 "request. Transfer only if the request cannot be handled with your tools."),
    )


def check_repeated_failure(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """A call identical to one that already returned an error will fail again."""
    for earlier in earlier_calls(messages):
        same_call = earlier.call.name == call.name and earlier.call.arguments == call.arguments
        if same_call and earlier.result.startswith("Error"):
            return GuardViolation(
                rule="repeated_failed_call",
                message=(f"This exact call already failed with \"{earlier.result[:150]}\". Do not repeat it. "
                         f"Check the arguments against the tool results, or ask the customer."),
            )
    return None


def latest_order_status(messages: list[dict], order_id: str) -> str | None:
    """The order's status from the latest tool result that shows it, or None if never looked up."""
    status = None
    wanted = order_id.lstrip("#")
    for earlier in earlier_calls(messages):
        try:
            result = json.loads(earlier.result)
        except ValueError:
            continue
        if isinstance(result, dict) and str(result.get("order_id", "")).lstrip("#") == wanted:
            status = result.get("status", status)
    return status


def check_order_status_fits_tool(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: pending orders can be cancelled or modified; delivered orders can be returned or
    exchanged. The tool name says which kind it needs (``*_pending_*`` / ``*_delivered_*``)."""
    if "_pending_" in call.name:
        needed_status = "pending"
    elif "_delivered_" in call.name:
        needed_status = "delivered"
    else:
        return None
    order_id = call.arguments.get("order_id")
    if not isinstance(order_id, str):
        return None
    status = latest_order_status(messages, order_id)
    if status is None or status == needed_status:
        return None
    return GuardViolation(
        rule="wrong_order_status",
        message=(f"Order {order_id} is '{status}', but {call.name} only works on {needed_status} orders. "
                 f"Pending orders can be cancelled or modified; delivered orders can be returned or "
                 f"exchanged. Tell the customer what is possible for a {status} order."),
    )


# ---- guards added after rounds 3 to 5: the adapter over-acts and loops (docs/pipeline/gate.md, AGENTS.md) ----

CHANGE_TOOLS = ("exchange_delivered_order_items", "modify_pending_order_items")


def result_json(result: str) -> dict | None:
    try:
        parsed = json.loads(result)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def product_of_items(messages: list[dict]) -> dict[str, tuple[str, str]]:
    """item id -> (product id, product name), from the order details and product details the agent has looked at."""
    product_by_item: dict[str, tuple[str, str]] = {}
    for earlier in earlier_calls(messages):
        result = result_json(earlier.result)
        if not result:
            continue
        for item in result.get("items") or []:  # an order: its items carry their product
            if isinstance(item, dict) and item.get("item_id") and item.get("product_id"):
                product_by_item[str(item["item_id"])] = (str(item["product_id"]), str(item.get("name") or ""))
        for item_id in (result.get("variants") or {}):  # a product: its variants are item ids
            if result.get("product_id"):
                product_by_item[str(item_id)] = (str(result["product_id"]), str(result.get("name") or ""))
    return product_by_item


def check_same_product_type(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: "each item can be exchanged to an available new item of the same product but of different product
    option. There cannot be any change of product types, e.g. modify shirt to shoe." """
    if call.name not in CHANGE_TOOLS:
        return None
    old_ids, new_ids = call.arguments.get("item_ids"), call.arguments.get("new_item_ids")
    if not isinstance(old_ids, list) or not isinstance(new_ids, list):
        return None
    product_by_item = product_of_items(messages)
    for old_id, new_id in zip(old_ids, new_ids):
        old_product, new_product = product_by_item.get(str(old_id)), product_by_item.get(str(new_id))
        if old_product and new_product and old_product[0] != new_product[0]:
            return GuardViolation(
                rule="exchange_other_product_type",
                message=(f"Item {old_id} is a {old_product[1] or 'different product'} and item {new_id} is a "
                         f"{new_product[1] or 'different product'}: the policy only allows changing an item to the "
                         f"same product in another option (a keyboard for a keyboard), never to another product "
                         f"type. Tell the customer this is not possible and offer an alternative of the same product."),
                fallback_reply=("I'm sorry, but that change isn't possible: an item can only be exchanged for the same "
                                "product in a different option, not for a different kind of product. Is there "
                                "something else I can help you with?"),
            )
    return None


def check_one_change_per_order(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """Policy: "Exchange or modify order tools can only be called once per order." """
    if call.name not in CHANGE_TOOLS:
        return None
    order_id = str(call.arguments.get("order_id") or "").lstrip("#")
    for earlier in earlier_calls(messages):
        same_order = str(earlier.call.arguments.get("order_id") or "").lstrip("#") == order_id
        if earlier.call.name in CHANGE_TOOLS and same_order and not earlier.result.startswith("Error"):
            return GuardViolation(
                rule="second_change_to_one_order",
                message=(f"A change to order {call.arguments.get('order_id')} was already submitted, and the policy "
                         f"allows it only once per order. Tell the customer what was submitted instead of making "
                         f"another change."),
                fallback_reply=("That order already has a change submitted, and the policy only allows one per "
                                "order. Is there anything else I can help you with?"),
            )
    return None


def check_repeated_call(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """The same lookup again, with nothing changed since, only returns the same answer (round 5: the adapter repeated
    one order lookup 8 times and ran out of room). A repeat after a change to the data is a legitimate check."""
    calls = earlier_calls(messages)
    for position in range(len(calls) - 1, -1, -1):
        earlier = calls[position]
        if earlier.call.name == call.name and earlier.call.arguments == call.arguments and not earlier.result.startswith("Error"):
            changed_since = any(is_write_tool(later.call.name) for later in calls[position + 1:])
            if not changed_since:
                return GuardViolation(
                    rule="repeated_call",
                    message=(f"You already made this exact call ({call.name}) and its result is in the conversation "
                             f"above. Do not call it again. Use that result: tell the customer what you found and "
                             f"what you can do, or take the next step."),
                    fallback_reply=("I have already looked that up and have the details. How would you like me to "
                                    "proceed?"),
                )
            return None
    return None


GUARDS = [
    check_early_transfer,
    check_authentication_first,
    check_repeated_failure,
    check_repeated_call,
    check_made_up_ids,
    check_order_status_fits_tool,
    check_same_product_type,
    check_one_change_per_order,
    check_confirmation_before_change,
]


def first_violation(messages: list[dict], call: ToolCall) -> GuardViolation | None:
    """The first guard the call breaks, in the order of GUARDS, or None."""
    for guard in GUARDS:
        violation = guard(messages, call)
        if violation is not None:
            return violation
    return None


def feedback_turn(violation: GuardViolation) -> str:
    """The user-role message the agent gets instead of a tool result."""
    return f"Tool call not run (policy check): {violation.message} Reply with a corrected tool call or a message to the customer."

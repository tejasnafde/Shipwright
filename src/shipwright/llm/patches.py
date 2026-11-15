"""Wrap the litellm ``completion`` call that tau2 uses.

Why: tau2 runs ``json.loads`` on every tool-call argument string. One malformed
call from a small model raises and kills the whole simulation, and the raw text
is lost. The wrapper replaces bad arguments with
``{MALFORMED_KEY: "<original string>"}``. tau2 then executes the call, the
environment returns a tool error to the agent (as a real deployment would), and
the original text stays in the trace.

For Tiny Aya (no native tool calling) it also switches the call to prompted
JSON tool calls, see ``shipwright.llm.prompted_tools``. ``vertex/`` model ids go to
Vertex AI with a fresh token (``shipwright.llm.vertex``). A ``UsageMeter`` counts the
tokens of every call by role (agent, user, judge) for the cost estimate.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any, Callable

from shipwright.observability import instrument
from shipwright.llm import guards, vertex
from shipwright.llm.context import fit_messages
from shipwright.llm.costs import UsageMeter
from shipwright.llm.prompted_tools import TRACE_FIELD, convert_response, is_prompted_model, prepare_request
from shipwright.heal.traces import MALFORMED_KEY

_lock = threading.Lock()
_installed = False


def sanitize_tool_calls(response: Any) -> int:
    """Fix malformed tool-call arguments in place. Returns how many were fixed."""
    fixed_count = 0
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = tool_call.function
            raw_arguments = function.arguments
            try:
                arguments_are_an_object = isinstance(json.loads(raw_arguments) if raw_arguments else {}, dict)
            except (TypeError, ValueError):
                arguments_are_an_object = False

            if not raw_arguments:
                function.arguments = "{}"
            elif not arguments_are_an_object:
                function.arguments = json.dumps({MALFORMED_KEY: raw_arguments})
                fixed_count += 1
            if function.name is None:
                function.name = ""
            if tool_call.id is None:
                tool_call.id = ""
    return fixed_count


def is_nl_judge_request(kwargs: dict) -> bool:
    """tau2's NL-assertion judge (it json.loads the reply with no fence handling)."""
    messages = kwargs.get("messages") or []
    return bool(messages) and "metExpectation" in str(messages[0].get("content") or "")


def strip_json_fences(response: Any) -> None:
    """Remove a ```json fence around a reply, in place."""
    for choice in getattr(response, "choices", None) or []:
        text = (choice.message.content or "").strip()
        if text.startswith("```"):
            if "\n" in text:
                text = text.split("\n", 1)[1]
            else:
                text = ""
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            choice.message.content = text.strip()


# tau2's NL-assertion judge runs json.loads on the reply; gpt-oss sometimes answers with empty
# or non-JSON text, and the whole run became an infra error (2026-11-11). Ask again this often.
NL_JUDGE_JSON_RETRIES = 2


# The user simulator sometimes replies with no text and no tool call; tau2 then raises
# "UserMessage must have either content or tool_calls" and the run is an infra error (round 1).
EMPTY_USER_REPLY_RETRIES = 2


def is_empty_reply(response: Any) -> bool:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return True
    message = choices[0].message
    has_text = bool((getattr(message, "content", None) or "").strip())
    has_tool_call = bool(getattr(message, "tool_calls", None))
    return not (has_text or has_tool_call)


# gpt-oss as the user simulator (seen on Vertex) sometimes writes past its own turn: its reply holds
# "<|end|><|start|>assistant<|channel|>analysis..." and goes on to play both sides, often ending with
# "###STOP###". tau2 then ends the conversation right after the customer says "yes, go ahead",
# before the agent can act (round 3 ceiling: 16 of 31 teacher failures). Keep only the text before
# the first such marker.
SIMULATOR_LEAK_MARKERS = ("<|end|>", "<|start|>", "<|channel|>", "<|message|>", "<|return|>")


def cut_leaked_turn(response: Any) -> bool:
    """Cut a user-simulator reply at the first chat-format marker. True if something was cut."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    message = choices[0].message
    content = getattr(message, "content", None) or ""
    positions = [content.find(marker) for marker in SIMULATOR_LEAK_MARKERS if marker in content]
    if not positions:
        return False
    message.content = content[:min(positions)].rstrip()
    return True


# gpt-oss as the user simulator (HF router, DeepInfra, low reasoning effort) sometimes sends a bare "###STOP###"
# while its own reasoning says it is about to answer ("So we say Yes. We should wait for next."). tau2 then ends the
# conversation as if the customer hung up. r10-ceiling: 15 of the 20 bare stops had no intent to end in the reasoning,
# 15 of the teacher's 41 failures; gate 8: 15 student failures against 3 base ones (the student asks more questions).
# Such a stop is asked again, like an empty reply. A stop whose reasoning says the customer is done is kept.
STOP_SIGNAL = "###STOP###"
STOP_INTENT = re.compile(
    r"stop|\bend\b|done|complete|finish|goodbye|\bbye\b|that's all|nothing else|satisf|resolved|wrap up|conclude|"
    r"no further|transfer",
    re.IGNORECASE,
)


def reasoning_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = choices[0].message
    reasoning = getattr(message, "reasoning_content", None)
    if not reasoning:
        fields = getattr(message, "provider_specific_fields", None) or {}
        reasoning = fields.get("reasoning_content") if isinstance(fields, dict) else None
    return reasoning or ""


def is_unexplained_stop(response: Any) -> bool:
    """A bare stop signal whose reasoning shows no intent to end the conversation. Without reasoning: kept."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    content = (getattr(choices[0].message, "content", None) or "").strip()
    if content != STOP_SIGNAL:
        return False
    reasoning = reasoning_text(response)
    return bool(reasoning) and not STOP_INTENT.search(reasoning)


# The same fault as speaks_as_agent below, other wordings: a reply that opens like the agent's apology or service line
# ("I'm sorry, but I can't verify your account without ...", "I understand this is frustrating. I'll transfer you to a
# supervisor") and says nothing for the customer. Recorded runs: 4 gate 13 candidate runs, 3 teacher runs, 0 base runs;
# "I'm sorry, but I can't share my credit-card details" is the customer and is kept (CUSTOMER_VOICE). Asked again too.
AGENT_OPENING = re.compile(
    r"^(i'?m sorry, but i (can'?t|cannot|am not able|'m not able)|i understand (this is|that this is|your) frustrat|"
    r"unfortunately, i (can'?t|cannot|am unable)|thank you for your patience|"
    r"i apologi[sz]e for (the|any) (inconvenience|confusion)|how (can|may) i (help|assist) you|"
    r"is there anything else i can (help|assist))",
    re.IGNORECASE,
)
CUSTOMER_VOICE = re.compile(r"\b(my|mine|i have|i don'?t have|i'?d like|i want|i need)\b", re.IGNORECASE)


def is_agent_voiced_reply(response: Any) -> bool:
    """A user-simulator reply that speaks as the agent and not as the customer."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    content = (getattr(choices[0].message, "content", None) or "").strip().replace("\u2019", "'")
    return bool(AGENT_OPENING.search(content)) and not CUSTOMER_VOICE.search(content)


# gpt-oss as the user simulator sometimes starts speaking as the agent ("I've processed the cancellation", "I'm not
# able to transfer you to a supervisor", "Anything else I can help you with?") and stays that way; agent and "customer"
# then argue until the context runs out. Gate 13: 5 of the 13 failed runs that ran out of context, 0 of the other 38
# failures; all 55 flagged customer turns in gates 11 and 13 and the r11 ceiling were the simulator in the agent's
# voice. Such a reply is asked again, like an empty one.
AGENT_VOICE = re.compile(
    r"I(’|')ve (processed|initiated|cancell?ed|submitted|updated)|I have (processed|initiated|cancell?ed|submitted)|"
    r"I(’|')m (not able|unable) to (transfer|reverse|process)|I can(’|')?no?t (reverse|transfer|process)|"
    r"anything else I can help( you)? with|I(’|')ll (process|go ahead and (cancel|process))|"
    r"refund (has been|will be) (issued|processed) to your",
    re.IGNORECASE,
)


def speaks_as_agent(response: Any) -> bool:
    """A user-simulator reply written in the agent's voice."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    return bool(AGENT_VOICE.search(getattr(choices[0].message, "content", None) or ""))


def is_json_reply(response: Any) -> bool:
    """True when every choice's content parses as JSON (what tau2's NL judge expects)."""

    choices = getattr(response, "choices", None) or []
    for choice in choices:
        try:
            json.loads(choice.message.content or "")
        except ValueError:
            return False
    return bool(choices)


# Waits before retrying a transient error (rate limit, timeout, server error) on a call that does
# not go to Vertex (Vertex has its own retries in vertex.call_with_refresh). In round 1, Tiny Aya on
# the HF router had no retry at all: one 429 or timeout turned the whole run into an infra error.
TRANSIENT_ERROR_WAITS = (2.0, 5.0, 10.0, 20.0)
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_ERROR_NAMES = ("RateLimit", "Timeout", "ServiceUnavailable", "InternalServer", "APIConnection")


def is_transient_error(error: BaseException) -> bool:
    if getattr(error, "status_code", None) in TRANSIENT_STATUS_CODES:
        return True
    error_name = type(error).__name__
    for transient_name in TRANSIENT_ERROR_NAMES:
        if transient_name in error_name:
            return True
    return False


def router_gpt_oss_defaults(call_kwargs: dict) -> dict:
    """The settings a Vertex gpt-oss call already gets, for gpt-oss on the HF router: low reasoning effort, a 60 s
    timeout (a hung call is retried, not waited for) and a reply budget that leaves room after its reasoning."""
    model = str(call_kwargs.get("model", ""))
    if vertex.is_vertex_model(model) or "gpt-oss" not in model:
        return call_kwargs
    call_kwargs = dict(call_kwargs)
    call_kwargs.setdefault("reasoning_effort", "low")
    call_kwargs.setdefault("timeout", 60.0)
    max_tokens = call_kwargs.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens < 1024:
        call_kwargs["max_tokens"] = 1024
    return call_kwargs


def call_with_transient_retries(call: Callable[..., Any], call_kwargs: dict,
                                sleep: Callable[[float], None] = time.sleep) -> Any:
    """Run ``call(**call_kwargs)``; on a transient error wait and retry (TRANSIENT_ERROR_WAITS)."""
    remaining_waits = list(TRANSIENT_ERROR_WAITS)
    while True:
        try:
            return call(**call_kwargs)
        except Exception as call_error:  # noqa: BLE001
            if not (is_transient_error(call_error) and remaining_waits):
                raise
            sleep(remaining_waits.pop(0) * (1 + random.random() / 2))


def proposed_tool_call(response: Any) -> guards.ToolCall | None:
    """The first tool call in a (converted) agent reply, or None for a text reply."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    tool_calls = getattr(choices[0].message, "tool_calls", None) or []
    if not tool_calls:
        return None
    function = tool_calls[0].function
    return guards.tool_call_from_message({"function": {"name": function.name, "arguments": function.arguments}})


def trace_fields(response: Any) -> dict:
    """The shipwright fields convert_response put on the reply (raw text, parse status, context)."""
    message = response.choices[0].message
    provider_fields = getattr(message, "provider_specific_fields", None) or {}
    return provider_fields.get(TRACE_FIELD) or {}


def repair_tool_call(conversation: list[dict], response: Any, repairs: list[dict],
                     available_tools: set[str] | None = None) -> None:
    """Rewrite the reply's tool call in place when guards.repaired_call has a fix for it; note it in ``repairs``."""
    call = proposed_tool_call(response)
    fixed = guards.repaired_call(conversation, call, available_tools) if call else None
    if fixed is None:
        return
    function = response.choices[0].message.tool_calls[0].function
    function.name = fixed.name
    function.arguments = json.dumps(fixed.arguments)
    repairs.append({"from": call.name, "to": fixed.name, "arguments": fixed.arguments})


def ask_again_until_policy_ok(conversation: list[dict], request_kwargs: dict, response: Any,
                              ask_model: Callable[[dict], Any]) -> Any:
    """Tool-call guards (shipwright.llm.guards): while the agent's tool call breaks a policy check,
    show it the rejected reply and the reason, and ask again (at most MAX_GUARD_RETRIES times).

    ``conversation`` is the tau2 history (what the guards check against); ``request_kwargs`` is the
    prompted request that produced ``response``. Returns the reply tau2 should get. The rejected
    replies are recorded in the reply's trace fields (``guard_rejections``), not in the history.
    """
    rejections = []
    repairs = []
    retry_messages = list(request_kwargs["messages"])
    available_tools = guards.tool_names_in_prompt(retry_messages)
    repair_tool_call(conversation, response, repairs, available_tools)
    for _ in range(guards.MAX_GUARD_RETRIES):
        call = proposed_tool_call(response)
        if call is None:
            break
        violation = guards.first_violation(conversation, call)
        if violation is None:
            break
        rejected_text = trace_fields(response).get("raw_text") or ""
        rejections.append({"rule": violation.rule, "raw_text": rejected_text})
        retry_messages.append({"role": "assistant", "content": rejected_text})
        retry_messages.append({"role": "user", "content": guards.feedback_turn(violation)})
        fitted_messages, _ = fit_messages(retry_messages)
        retry_kwargs = dict(request_kwargs)
        retry_kwargs["messages"] = fitted_messages
        response = ask_model(retry_kwargs)
        repair_tool_call(conversation, response, repairs, available_tools)

    if rejections or repairs:
        final_call = proposed_tool_call(response)
        final_violation = guards.first_violation(conversation, final_call) if final_call else None
        fields = dict(trace_fields(response))
        fields["guard_rejections"] = rejections
        fields["guard_repairs"] = repairs
        message = response.choices[0].message
        if final_violation is not None and final_violation.fallback_reply:
            # Still the same bad call after being asked again: say the fallback to the customer instead of running it.
            message.content = final_violation.fallback_reply
            message.tool_calls = None
            fields["guard_replaced_with_text"] = final_violation.rule
            final_violation = None
        fields["guard_allowed_after_retries"] = final_violation.rule if final_violation else None
        provider_fields = dict(getattr(message, "provider_specific_fields", None) or {})
        provider_fields[TRACE_FIELD] = fields
        message.provider_specific_fields = provider_fields
    return response


def call_role(original: dict, kwargs: dict, agent_model: str | None, user_model: str | None) -> str:
    """Which role a call belongs to: judge (NL judge or a rerouted tau2 default), agent or user."""
    if is_nl_judge_request(kwargs) or original.get("model") != kwargs.get("model"):
        return "judge"
    model = kwargs.get("model")
    if model == agent_model and (model != user_model or kwargs.get("tools")):
        return "agent"
    return "user"


def install(
    fake_completion: Callable[..., Any] | None = None,
    model_map: Callable[[dict], dict] | None = None,
    meter: UsageMeter | None = None,
    agent_model: str | None = None,
    user_model: str | None = None,
    spend_guard: Callable[[], None] | None = None,
) -> None:
    """Patch ``tau2.utils.llm_utils.completion``. Call once, before running tau2.

    ``fake_completion`` replaces the network call (used by the offline dry run; ``vertex/``
    ids are then not routed). ``model_map`` rewrites the request kwargs first (used to
    reroute tau2's built-in default models, e.g. the NL-assertion judge, to the user model).
    ``meter`` counts usage by role (``call_role``), keyed by the model id before routing.
    ``spend_guard`` is called before every model call and raises when the run is over budget.
    """
    global _installed
    import tau2.utils.llm_utils as llm_utils

    with _lock:
        if _installed:
            return
        real_completion = fake_completion or llm_utils.completion

        # *args and **kwargs pass litellm's own call signature through untouched.
        def wrapped_completion(*args: Any, **kwargs: Any) -> Any:
            if spend_guard is not None:
                spend_guard()
            original_kwargs = kwargs
            if model_map is not None:
                kwargs = model_map(kwargs)
            model = str(kwargs.get("model", ""))
            role = call_role(original_kwargs, kwargs, agent_model, user_model)

            is_prompted_call = is_prompted_model(model) and bool(kwargs.get("tools"))
            conversation = kwargs.get("messages") or []  # tau2's history, before prompting
            prompt_context = None
            if is_prompted_call:
                kwargs, prompt_context = prepare_request(kwargs)

            attempt_counter = [0]

            def counted_completion(**call_kwargs: Any) -> Any:
                attempt_counter[0] += 1
                return real_completion(*args, **call_kwargs)

            def retry_count() -> int:
                return max(0, attempt_counter[0] - 1)

            bad_request_waits = vertex.BAD_REQUEST_WAITS if role in ("user", "judge") else ()

            def make_call_with(call_kwargs: dict) -> Any:
                if fake_completion is not None:
                    return counted_completion(**call_kwargs)
                if vertex.is_vertex_model(model):
                    return vertex.call_with_refresh(counted_completion, call_kwargs,
                                                    bad_request_waits=bad_request_waits)
                return call_with_transient_retries(counted_completion, router_gpt_oss_defaults(call_kwargs))

            def ask_model(call_kwargs: dict) -> Any:
                model_response = instrument.llm_call(role, model, call_kwargs, lambda: make_call_with(call_kwargs),
                                                     retries=retry_count)
                if meter is not None:
                    meter.add(role, model, getattr(model_response, "usage", None))
                return model_response

            is_nl_judge_call = is_nl_judge_request(kwargs)
            is_user_simulator_call = role == "user"
            attempts = 1
            if is_nl_judge_call:
                attempts += NL_JUDGE_JSON_RETRIES
            elif is_user_simulator_call:
                attempts += EMPTY_USER_REPLY_RETRIES
            for _ in range(attempts):
                response = ask_model(kwargs)
                if is_nl_judge_call:
                    strip_json_fences(response)
                    if is_json_reply(response):
                        break
                elif is_user_simulator_call:
                    cut_leaked_turn(response)
                    # Mitushi's speaks_as_agent and is_agent_voiced_reply catch different wordings of the same fault.
                    usable = not (is_empty_reply(response) or is_unexplained_stop(response)
                                  or speaks_as_agent(response) or is_agent_voiced_reply(response))
                    if usable:
                        break
                else:
                    break
            if is_prompted_call:
                convert_response(response, prompt_context)
                if guards.is_enabled() and role == "agent":

                    def ask_prompted_model(call_kwargs: dict) -> Any:
                        retried_response = ask_model(call_kwargs)
                        convert_response(retried_response, prompt_context)
                        return retried_response

                    response = ask_again_until_policy_ok(conversation, kwargs, response, ask_prompted_model)
            sanitize_tool_calls(response)
            return response

        llm_utils.completion = wrapped_completion
        # tau2 prices each reply with litellm's price map, which has none of our models: every call
        # logged an ERROR ("This model isn't mapped yet") and used 0. Cost is metered by
        # UsageMeter (summary.json cost_estimate_usd), so say 0 without the error.
        llm_utils.get_response_cost = lambda response: 0.0
        _installed = True
    instrument.install_tau2()


class ModelRerouter:
    """Send any model other than the allowed ones to a fallback model, and count it."""

    def __init__(self, allowed: set[str], fallback: str):
        self.allowed = allowed
        self.fallback = fallback
        self.counts: dict[str, int] = {}

    def __call__(self, kwargs: dict) -> dict:
        model = kwargs.get("model", "")
        if model in self.allowed:
            return kwargs
        with _lock:
            self.counts[model] = self.counts.get(model, 0) + 1
        rerouted_kwargs = dict(kwargs)
        rerouted_kwargs["model"] = self.fallback
        # Params added for OpenAI's Responses bridge do not apply to the router.
        rerouted_kwargs.pop("allowed_openai_params", None)
        return rerouted_kwargs

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
import threading
from typing import Any, Callable

from shipwright.observability import instrument
from shipwright.llm import vertex
from shipwright.llm.costs import UsageMeter
from shipwright.llm.prompted_tools import convert_response, is_prompted_model, prepare_request
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


def is_json_reply(response: Any) -> bool:
    """True when every choice's content parses as JSON (what tau2's NL judge expects)."""

    choices = getattr(response, "choices", None) or []
    for choice in choices:
        try:
            json.loads(choice.message.content or "")
        except ValueError:
            return False
    return bool(choices)


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
            if fake_completion is None and vertex.is_vertex_model(model):
                def make_call() -> Any:
                    return vertex.call_with_refresh(counted_completion, kwargs, bad_request_waits=bad_request_waits)
            else:
                def make_call() -> Any:
                    return counted_completion(**kwargs)

            is_nl_judge_call = is_nl_judge_request(kwargs)
            judge_attempts = 1
            if is_nl_judge_call:
                judge_attempts += NL_JUDGE_JSON_RETRIES
            for _ in range(judge_attempts):
                response = instrument.llm_call(role, model, kwargs, make_call, retries=retry_count)
                if meter is not None:
                    meter.add(role, model, getattr(response, "usage", None))
                if not is_nl_judge_call:
                    break
                strip_json_fences(response)
                if is_json_reply(response):
                    break
            if is_prompted_call:
                convert_response(response, prompt_context)
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

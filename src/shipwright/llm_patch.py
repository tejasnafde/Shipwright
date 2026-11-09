"""Wrap the litellm ``completion`` call that tau2 uses.

Why: tau2 runs ``json.loads`` on every tool-call argument string. One malformed
call from a small model raises and kills the whole simulation, and the raw text
is lost. The wrapper replaces bad arguments with
``{MALFORMED_KEY: "<original string>"}``. tau2 then executes the call, the
environment returns a tool error to the agent (as a real deployment would), and
the original text stays in the trace.

For Tiny Aya (no native tool calling) it also switches the call to prompted
JSON tool calls, see ``shipwright.prompted_tools``. ``vertex/`` model ids go to
Vertex AI with a fresh token (``shipwright.vertex``). A ``UsageMeter`` counts the
tokens of every call by role (agent, user, judge) for the cost estimate.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

from shipwright import instrument, vertex
from shipwright.costs import UsageMeter
from shipwright.prompted_tools import convert_response, is_prompted_model, prepare_request
from shipwright.traces import MALFORMED_KEY

_lock = threading.Lock()
_installed = False


def sanitize_tool_calls(response: Any) -> int:
    """Fix malformed tool-call arguments in place. Returns how many were fixed."""
    fixed = 0
    for choice in getattr(response, "choices", None) or []:
        message = getattr(choice, "message", None)
        for tc in getattr(message, "tool_calls", None) or []:
            fn = tc.function
            raw = fn.arguments
            try:
                ok = isinstance(json.loads(raw) if raw else {}, dict)
            except (TypeError, ValueError):
                ok = False
            if not raw:
                fn.arguments = "{}"
            elif not ok:
                fn.arguments = json.dumps({MALFORMED_KEY: raw})
                fixed += 1
            if fn.name is None:
                fn.name = ""
            if tc.id is None:
                tc.id = ""
    return fixed


def is_nl_judge_request(kwargs: dict) -> bool:
    """tau2's NL-assertion judge (it json.loads the reply with no fence handling)."""
    msgs = kwargs.get("messages") or []
    return bool(msgs) and "metExpectation" in str(msgs[0].get("content") or "")


def strip_json_fences(response: Any) -> None:
    """Remove a ```json fence around a reply, in place."""
    for choice in getattr(response, "choices", None) or []:
        text = (choice.message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            choice.message.content = text.strip()


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
) -> None:
    """Patch ``tau2.utils.llm_utils.completion``. Call once, before running tau2.

    ``fake_completion`` replaces the network call (used by the offline dry run; ``vertex/``
    ids are then not routed). ``model_map`` rewrites the request kwargs first (used to
    reroute tau2's built-in default models, e.g. the NL-assertion judge, to the user model).
    ``meter`` counts usage by role (``call_role``), keyed by the model id before routing.
    """
    global _installed
    import tau2.utils.llm_utils as llm_utils

    with _lock:
        if _installed:
            return
        real = fake_completion or llm_utils.completion

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            original = kwargs
            if model_map is not None:
                kwargs = model_map(kwargs)
            model = str(kwargs.get("model", ""))
            role = call_role(original, kwargs, agent_model, user_model)
            prompted = is_prompted_model(model) and bool(kwargs.get("tools"))
            context = None
            if prompted:
                kwargs, context = prepare_request(kwargs)
            attempts = [0]

            def counted(**kw: Any) -> Any:
                attempts[0] += 1
                return real(*args, **kw)

            if fake_completion is None and vertex.is_vertex_model(model):
                do = lambda: vertex.call_with_refresh(counted, kwargs)  # noqa: E731
            else:
                do = lambda: counted(**kwargs)  # noqa: E731
            response = instrument.llm_call(role, model, kwargs, do, retries=lambda: max(0, attempts[0] - 1))
            if meter is not None:
                meter.add(role, model, getattr(response, "usage", None))
            if is_nl_judge_request(kwargs):
                strip_json_fences(response)
            if prompted:
                convert_response(response, context)
            sanitize_tool_calls(response)
            return response

        llm_utils.completion = wrapped
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
        kwargs = dict(kwargs)
        kwargs["model"] = self.fallback
        # Params added for OpenAI's Responses bridge do not apply to the router.
        kwargs.pop("allowed_openai_params", None)
        return kwargs

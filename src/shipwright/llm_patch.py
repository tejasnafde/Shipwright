"""Wrap the litellm ``completion`` call that tau2 uses.

Why: tau2 runs ``json.loads`` on every tool-call argument string. One malformed
call from a small model raises and kills the whole simulation, and the raw text
is lost. The wrapper replaces bad arguments with
``{MALFORMED_KEY: "<original string>"}``. tau2 then executes the call, the
environment returns a tool error to the agent (as a real deployment would), and
the original text stays in the trace.

For Tiny Aya (no native tool calling) it also switches the call to prompted
JSON tool calls, see ``shipwright.prompted_tools``.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable

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


def install(fake_completion: Callable[..., Any] | None = None) -> None:
    """Patch ``tau2.utils.llm_utils.completion``. Call once, before running tau2.

    ``fake_completion`` replaces the network call (used by the offline dry run).
    """
    global _installed
    import tau2.utils.llm_utils as llm_utils

    with _lock:
        if _installed:
            return
        real = fake_completion or llm_utils.completion

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            prompted = is_prompted_model(str(kwargs.get("model", ""))) and bool(kwargs.get("tools"))
            if prompted:
                kwargs = prepare_request(kwargs)
            response = real(*args, **kwargs)
            if prompted:
                convert_response(response)
            sanitize_tool_calls(response)
            return response

        llm_utils.completion = wrapped
        _installed = True

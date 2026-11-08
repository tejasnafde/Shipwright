"""A scripted stand-in for litellm.completion, for the offline dry run.

The agent script covers each trace case: a valid call, malformed arguments,
an unknown tool name, then a plain text reply. The user stops after a few turns.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from litellm import ModelResponse

AGENT_SCRIPT: list[dict] = [
    {"tool": ("find_user_id_by_email", json.dumps({"email": "dry@run.test"}))},
    {"tool": ("get_user_details", '{"user_id": "abc"')},  # malformed JSON
    {"tool": ("refund_everything", json.dumps({}))},  # unknown tool
    {"text": "Sorry, I could not find your account."},
]


def _response(model: str, content: str | None, tool: tuple[str, str] | None) -> ModelResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": tool[0], "arguments": tool[1]},
            }
        ]
    return ModelResponse(
        model=model,
        choices=[{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    )


def make_fake_completion(agent_model: str):
    def fake_completion(model: str, messages: list[dict], **kwargs: Any) -> ModelResponse:
        if model == agent_model:
            step = sum(1 for m in messages if m.get("role") == "assistant")
            item = AGENT_SCRIPT[min(step, len(AGENT_SCRIPT) - 1)]
            return _response(model, item.get("text"), item.get("tool"))
        turns = sum(1 for m in messages if m.get("role") == "assistant")
        text = "###STOP###" if turns >= 4 else "Hi, my email is dry@run.test. I want a refund."
        return _response(model, text, None)

    return fake_completion

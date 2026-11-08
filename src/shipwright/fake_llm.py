"""A scripted stand-in for litellm.completion, for the offline dry run.

The agent script covers each trace case: a valid call, malformed arguments,
an unknown tool name, then a plain text reply. A Tiny Aya agent gets no native
tools (llm_patch removes them), so it gets the prompted-JSON script. The user stops after a few turns.
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


# Prompted format (Tiny Aya): the same script as plain text replies.
PROMPTED_SCRIPT: list[str] = [
    '```json\n{"name": "find_user_id_by_email", "arguments": {"email": "dry@run.test"}}\n```',
    '```json\n{"name": "get_user_details", "arguments": {"user_id": "abc"\n```',  # parse fails
    'Let me check. {"name": "refund_everything", "arguments": {}}',  # unknown tool, no fence
    "Sorry, I could not find your account.",
]


def make_fake_completion(agent_model: str):
    def fake_completion(model: str, messages: list[dict], **kwargs: Any) -> ModelResponse:
        if model == agent_model and not kwargs.get("tools"):
            # tau2's fixed greeting is the first assistant turn; do not count it.
            step = max(sum(1 for m in messages if m.get("role") == "assistant") - 1, 0)
            text = PROMPTED_SCRIPT[min(step, len(PROMPTED_SCRIPT) - 1)]
            return _response(model, text, None)
        if model == agent_model:
            step = sum(1 for m in messages if m.get("role") == "assistant")
            item = AGENT_SCRIPT[min(step, len(AGENT_SCRIPT) - 1)]
            return _response(model, item.get("text"), item.get("tool"))
        turns = sum(1 for m in messages if m.get("role") == "assistant")
        text = "###STOP###" if turns >= 4 else "Hi, my email is dry@run.test. I want a refund."
        return _response(model, text, None)

    return fake_completion

"""One tiny request: does the model support native tool calling (the `tools` param) via the HF router?

Usage: .venv/bin/python scripts/check_tool_calling.py [--model CohereLabs/tiny-aya-global:cohere]
Exit code 0: native tool call returned. 1: no native tool call. 2: no token or HTTP error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shipwright import cloud  # noqa: E402

TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_details",
        "description": "Get the status and details of an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order id, like #W0000000"}},
            "required": ["order_id"],
        },
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global:cohere")
    options = parser.parse_args()

    token = cloud.hf_token()
    if not token:
        print("No HF token found.", file=sys.stderr)
        return 2

    request_body = {
        "model": options.model,
        "messages": [
            {"role": "system", "content": "You are a retail support agent. Use tools to look up orders."},
            {"role": "user", "content": "What is the status of my order #W2378156?"},
        ],
        "tools": [TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 200,
    }
    response = httpx.post(
        f"{cloud.HF_ROUTER}/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json=request_body,
        timeout=60,
    )
    print("HTTP", response.status_code)
    if response.status_code != 200:
        print(response.text[:1000])
        return 2

    response_data = response.json()
    first_choice = response_data["choices"][0]
    message = first_choice["message"]
    report = {
        "message": message,
        "finish_reason": first_choice.get("finish_reason"),
        "usage": response_data.get("usage"),
    }
    print(json.dumps(report, indent=2))

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        function_call = tool_calls[0]["function"]
        print(f"NATIVE TOOL CALLING: yes ({function_call['name']} {function_call['arguments']})")
        return 0
    print("NATIVE TOOL CALLING: no tool_calls in the response")
    return 1


if __name__ == "__main__":
    sys.exit(main())

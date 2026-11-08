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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import HF_ROUTER, load_token  # noqa: E402

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
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="CohereLabs/tiny-aya-global:cohere")
    args = p.parse_args()
    token = load_token()
    if not token:
        print("No HF token found.", file=sys.stderr)
        return 2
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a retail support agent. Use tools to look up orders."},
            {"role": "user", "content": "What is the status of my order #W2378156?"},
        ],
        "tools": [TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 200,
    }
    r = httpx.post(
        f"{HF_ROUTER}/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=60,
    )
    print("HTTP", r.status_code)
    if r.status_code != 200:
        print(r.text[:1000])
        return 2
    data = r.json()
    msg = data["choices"][0]["message"]
    print(json.dumps({"message": msg, "finish_reason": data["choices"][0].get("finish_reason"),
                      "usage": data.get("usage")}, indent=2))
    calls = msg.get("tool_calls") or []
    if calls:
        print(f"NATIVE TOOL CALLING: yes ({calls[0]['function']['name']} {calls[0]['function']['arguments']})")
        return 0
    print("NATIVE TOOL CALLING: no tool_calls in the response")
    return 1


if __name__ == "__main__":
    sys.exit(main())

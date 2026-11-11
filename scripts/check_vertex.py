"""Smoke check: one tiny call to each Vertex AI model the team uses, and print the usage block.

  .venv/bin/python scripts/check_vertex.py                 # 2 calls, well under $0.001
  .venv/bin/python scripts/check_vertex.py --cache-probe   # + 2 DeepSeek calls of about 5k tokens
                                                           #   (about $0.003) to see cached_tokens

Billed to GCP project $SHIPWRIGHT_GCP_PROJECT (default lexsi-hackathon), not to HF credits.
Auth: gcloud --configuration=$SHIPWRIGHT_GCLOUD_CONFIG (default personal; teammates: lexsi).
The token is never printed.

--cache-probe sends the same long prefix twice. Vertex implicit caching (Preview) needs at least
4096 prompt tokens and a hit is not guaranteed; the second call should report cached tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.llm import vertex  # noqa: E402
from shipwright.llm.costs import cached_tokens  # noqa: E402

MODELS = {
    vertex.TEACHER_MODEL: {"max_tokens": 32},
    # gpt-oss reasons first: a small max_tokens can leave no answer.
    vertex.USER_MODEL: {"max_tokens": 1024, "reasoning_effort": "low"},
}


def call(http_client: httpx.Client, model: str, messages: list[dict], extra_options: dict) -> dict:
    """One chat call. Prints what came back; returns the response body, or {} on an HTTP error."""
    url, request_model, make_headers = vertex.endpoint(model, None, "")
    request_body = {"model": request_model, "messages": messages, "temperature": 0, **extra_options}

    started_at = time.perf_counter()
    response = http_client.post(url, headers=make_headers(), json=request_body)
    seconds_taken = time.perf_counter() - started_at
    if response.status_code != 200:
        print(f"{model}: HTTP {response.status_code}: {response.text[:400]}")
        return {}

    response_body = response.json()
    first_choice = response_body["choices"][0]
    message = first_choice["message"]
    print(f"{model}: {seconds_taken:.1f}s, finish {first_choice.get('finish_reason')}")
    print(f"  content:   {(message.get('content') or '')[:120]!r}")
    if message.get("reasoning_content"):
        print(f"  reasoning: {len(message['reasoning_content'])} chars")
    print(f"  usage:     {json.dumps(response_body.get('usage'))}")
    print(f"  cached:    {cached_tokens(response_body.get('usage'))}")
    return response_body


def run_cache_probe(http_client: httpx.Client) -> bool:
    """Send the same long prefix twice to DeepSeek. Returns True if both calls worked."""
    # A fixed, byte-identical prefix of about 5k tokens, then a short varying question.
    reference_lines = [
        f"item {item_number:04d}: order #W{1000000 + item_number} ships from warehouse {item_number % 17}"
        for item_number in range(280)
    ]
    prefix = "Reference list (ignore unless asked):\n" + "\n".join(reference_lines)

    all_calls_worked = True
    for question in ("Which warehouse ships item 0007?", "Which warehouse ships item 0011?"):
        messages = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": question + " Answer in 3 words."},
        ]
        response_body = call(http_client, vertex.TEACHER_MODEL, messages, {"max_tokens": 16})
        all_calls_worked = all_calls_worked and bool(response_body)
    return all_calls_worked


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-probe", action="store_true", help="also send a ~5k-token prefix twice to DeepSeek")
    options = parser.parse_args()

    print(f"project {vertex.project()}, location {vertex.location()}, base {vertex.api_base()}")
    try:
        vertex.provider().get()
    except RuntimeError as token_error:
        print(f"Vertex token: {token_error}", file=sys.stderr)
        return 2

    all_calls_worked = True
    with httpx.Client(timeout=120) as http_client:
        for model, extra_options in MODELS.items():
            messages = [{"role": "user", "content": "Reply with the single word: ready"}]
            response_body = call(http_client, model, messages, extra_options)
            all_calls_worked = all_calls_worked and bool(response_body)
        if options.cache_probe:
            cache_probe_worked = run_cache_probe(http_client)
            all_calls_worked = all_calls_worked and cache_probe_worked

    if all_calls_worked:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

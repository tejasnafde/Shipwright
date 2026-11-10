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


def call(client: httpx.Client, model: str, messages: list[dict], extra: dict) -> dict:
    url, body_model, headers = vertex.endpoint(model, None, "")
    t0 = time.perf_counter()
    r = client.post(url, headers=headers(), json={"model": body_model, "messages": messages, "temperature": 0, **extra})
    took = time.perf_counter() - t0
    if r.status_code != 200:
        print(f"{model}: HTTP {r.status_code}: {r.text[:400]}")
        return {}
    body = r.json()
    msg = body["choices"][0]["message"]
    print(f"{model}: {took:.1f}s, finish {body['choices'][0].get('finish_reason')}")
    print(f"  content:   {(msg.get('content') or '')[:120]!r}")
    if msg.get("reasoning_content"):
        print(f"  reasoning: {len(msg['reasoning_content'])} chars")
    print(f"  usage:     {json.dumps(body.get('usage'))}")
    print(f"  cached:    {cached_tokens(body.get('usage'))}")
    return body


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-probe", action="store_true", help="also send a ~5k-token prefix twice to DeepSeek")
    args = p.parse_args()
    print(f"project {vertex.project()}, location {vertex.location()}, base {vertex.api_base()}")
    try:
        vertex.provider().get()
    except RuntimeError as e:
        print(f"Vertex token: {e}", file=sys.stderr)
        return 2
    ok = True
    with httpx.Client(timeout=120) as client:
        for model, extra in MODELS.items():
            ok &= bool(call(client, model, [{"role": "user", "content": "Reply with the single word: ready"}], extra))
        if args.cache_probe:
            # A fixed, byte-identical prefix of about 5k tokens, then a short varying question.
            prefix = "Reference list (ignore unless asked):\n" + "\n".join(
                f"item {i:04d}: order #W{1000000 + i} ships from warehouse {i % 17}" for i in range(280))
            for q in ("Which warehouse ships item 0007?", "Which warehouse ships item 0011?"):
                msgs = [{"role": "system", "content": prefix}, {"role": "user", "content": q + " Answer in 3 words."}]
                ok &= bool(call(client, vertex.TEACHER_MODEL, msgs, {"max_tokens": 16}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

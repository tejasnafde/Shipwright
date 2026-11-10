"""Token usage per role and the cost estimate in summary.json.

Roles: ``agent`` (``teacher`` in teacher runs), ``user`` (the simulated customer) and
``judge`` (tau2's NL-assertion judge and any other tau2 default model, rerouted to the user
model). ``UsageMeter`` counts every LLM call the patched tau2 makes, by role, including
cached prompt tokens. Prices are estimates; check HF billing and GCP billing for real cost.
"""

from __future__ import annotations

import threading
from typing import Any

# USD per 1M tokens: (input, cached input, output).
VERTEX_PRICES = {  # Vertex AI MaaS list prices, checked 2026-11-09
    "deepseek-ai/deepseek-v4.1-flash-maas": (0.30, 0.03, 1.20),
    "openai/gpt-oss-120b-maas": (0.09, 0.09, 0.36),  # no implicit caching listed for it
}
HF_PRICES = {
    "openai/openai/gpt-oss-120b:cerebras": (0.25, 0.25, 0.69),
}
TINY_AYA_GUESS = (0.50, 0.50, 1.50)  # cohere provider: price not published, conservative guess
UNKNOWN_GUESS = (1.00, 1.00, 3.00)  # any other model: an assumed price, flagged in the output
ZERO = (0.0, 0.0, 0.0)

ROLES = ("agent", "user", "judge")


def provider_of(model: str, api_base: str | None = None, dry_run: bool = False) -> str:
    if dry_run:
        return "dry_run"
    if api_base:
        return "local"  # our own GPU (vLLM), billed per hour, not per token
    return "vertex" if model.startswith("vertex/") else "hf_router"


def price_for(model: str, provider: str) -> tuple[tuple[float, float, float], bool]:
    """((input, cached input, output) USD per 1M tokens, assumed?)."""
    if provider in ("dry_run", "local"):
        return ZERO, False
    if provider == "vertex":
        p = VERTEX_PRICES.get(model.removeprefix("vertex/"))
        return (p, False) if p else (UNKNOWN_GUESS, True)
    if model in HF_PRICES:
        return HF_PRICES[model], False
    if "tiny-aya" in model.lower():
        return TINY_AYA_GUESS, True
    return UNKNOWN_GUESS, True


def cached_tokens(usage: Any) -> int | None:
    """Cached prompt tokens from a usage block (dict or object), or None if not reported.

    OpenAI format: ``prompt_tokens_details.cached_tokens``. Vertex native format:
    ``cachedContentTokenCount``. Some providers put ``cached_tokens`` at the top level.
    The Vertex OpenAI-compatible endpoint sends ``prompt_tokens_details: null`` and an
    ``extra_properties.google`` block on a call with no cache hit (checked 2026-11-09);
    the field it uses on a hit is not verified yet, so the google block is searched too.
    """
    if usage is None:
        return None

    def get(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    details = get(usage, "prompt_tokens_details")
    google = get(get(usage, "extra_properties"), "google")
    found = [int(v) for v in (get(details, "cached_tokens"), get(usage, "cached_tokens"),
                  get(usage, "cachedContentTokenCount"), get(usage, "cache_read_input_tokens"),
                  get(google, "cached_content_token_count"), get(google, "cachedContentTokenCount"),
                  get(google, "cached_tokens"))
             if isinstance(v, (int, float)) and not isinstance(v, bool)]
    # litellm may fill a default 0 in one field while the provider reports the hit in another.
    return max(found) if found else None


def usage_numbers(usage: Any) -> dict:
    """prompt, completion and cached prompt tokens from a usage block (missing -> 0)."""
    def get(key: str) -> int:
        v = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}
    return {"prompt_tokens": get("prompt_tokens"), "completion_tokens": get("completion_tokens"),
            "cached_prompt_tokens": cached_tokens(usage) or 0}


class UsageMeter:
    """Counts calls and tokens per (role, model). Thread-safe (tau2 runs tasks in threads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rows: dict[tuple[str, str], dict] = {}

    def add(self, role: str, model: str, usage: Any) -> None:
        nums = usage_numbers(usage)
        with self._lock:
            row = self.rows.setdefault((role, model), {"calls": 0, "prompt_tokens": 0,
                                                        "completion_tokens": 0, "cached_prompt_tokens": 0})
            row["calls"] += 1
            for k, v in nums.items():
                row[k] += v

    def by_role(self) -> dict[str, dict]:
        """{role: {calls, prompt_tokens, completion_tokens, cached_prompt_tokens, models: [...]}}."""
        out: dict[str, dict] = {}
        with self._lock:
            for (role, model), row in self.rows.items():
                agg = out.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                            "cached_prompt_tokens": 0, "models": []})
                for k in ("calls", "prompt_tokens", "completion_tokens", "cached_prompt_tokens"):
                    agg[k] += row[k]
                agg["models"].append(model)
        return out


def cost_usd(prompt: int, cached: int, completion: int, price: tuple[float, float, float]) -> float:
    pin, pcached, pout = price
    cached = min(cached, prompt)
    return ((prompt - cached) * pin + cached * pcached + completion * pout) / 1e6


def hit_rate(prompt: int, cached: int) -> float | None:
    return round(cached / prompt, 4) if prompt else None


def estimate(meter_rows: dict[tuple[str, str], dict], providers: dict[str, str],
             role_names: dict[str, str] | None = None) -> dict:
    """Cost estimate per role, from ``UsageMeter.rows`` and the provider of each model.

    ``providers`` maps model -> provider (``provider_of``). ``role_names`` renames roles in the
    output (teacher runs: ``{"agent": "teacher"}``).
    """
    role_names = role_names or {}
    by_role: dict[str, dict] = {}
    total = 0.0
    for (role, model), row in sorted(meter_rows.items()):
        name = role_names.get(role, role)
        provider = providers.get(model) or provider_of(model)
        price, assumed = price_for(model, provider)
        usd = cost_usd(row["prompt_tokens"], row["cached_prompt_tokens"], row["completion_tokens"], price)
        total += usd
        entry = by_role.setdefault(name, {"usd": 0.0, "models": []})
        entry["usd"] = round(entry["usd"] + usd, 6)
        entry["models"].append({
            "model": model, "provider": provider, **row,
            "cache_hit_rate": hit_rate(row["prompt_tokens"], row["cached_prompt_tokens"]),
            "usd_per_1m": {"input": price[0], "cached_input": price[1], "output": price[2]},
            "price_assumed": assumed, "usd": round(usd, 6),
        })
    out: dict[str, Any] = {name: round(entry["usd"], 4) for name, entry in by_role.items()}
    out["total"] = round(total, 4)
    out["by_role"] = by_role
    out["note"] = ("estimate from metered token counts and list prices in src/shipwright/llm/costs.py;"
                   " check HF and GCP billing for the real number")
    return out

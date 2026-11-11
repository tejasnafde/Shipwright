"""Token usage per role and the cost estimate in summary.json.

Roles: ``agent`` (``teacher`` in teacher runs), ``user`` (the simulated customer) and
``judge`` (tau2's NL-assertion judge and any other tau2 default model, rerouted to the user
model). ``UsageMeter`` counts every LLM call the patched tau2 makes, by role, including
cached prompt tokens. Prices are estimates; check HF billing and GCP billing for real cost.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

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
    if model.startswith("vertex/"):
        return "vertex"
    return "hf_router"


def price_for(model: str, provider: str) -> tuple[tuple[float, float, float], bool]:
    """((input, cached input, output) USD per 1M tokens, assumed?)."""
    if provider in ("dry_run", "local"):
        return ZERO, False
    if provider == "vertex":
        vertex_price = VERTEX_PRICES.get(model.removeprefix("vertex/"))
        if vertex_price:
            return vertex_price, False
        return UNKNOWN_GUESS, True
    if model in HF_PRICES:
        return HF_PRICES[model], False
    if "tiny-aya" in model.lower():
        return TINY_AYA_GUESS, True
    return UNKNOWN_GUESS, True


def _read_field(container: Any, key: str) -> Any:
    """container[key] for a dict, getattr for an object, None for a missing container."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _is_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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

    prompt_details = _read_field(usage, "prompt_tokens_details")
    google_block = _read_field(_read_field(usage, "extra_properties"), "google")
    candidate_counts = [
        _read_field(prompt_details, "cached_tokens"),
        _read_field(usage, "cached_tokens"),
        _read_field(usage, "cachedContentTokenCount"),
        _read_field(usage, "cache_read_input_tokens"),
        _read_field(google_block, "cached_content_token_count"),
        _read_field(google_block, "cachedContentTokenCount"),
        _read_field(google_block, "cached_tokens"),
    ]
    reported_counts = [int(count) for count in candidate_counts if _is_count(count)]
    # litellm may fill a default 0 in one field while the provider reports the hit in another.
    if reported_counts:
        return max(reported_counts)
    return None


def usage_numbers(usage: Any) -> dict:
    """prompt, completion and cached prompt tokens from a usage block (missing -> 0)."""
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}

    def token_count(key: str) -> int:
        value = _read_field(usage, key)
        if _is_count(value):
            return int(value)
        return 0

    return {
        "prompt_tokens": token_count("prompt_tokens"),
        "completion_tokens": token_count("completion_tokens"),
        "cached_prompt_tokens": cached_tokens(usage) or 0,
    }


class UsageMeter:
    """Counts calls and tokens per (role, model). Thread-safe (tau2 runs tasks in threads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rows: dict[tuple[str, str], dict] = {}

    def add(self, role: str, model: str, usage: Any) -> None:
        token_counts = usage_numbers(usage)
        with self._lock:
            row = self.rows.setdefault(
                (role, model),
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0},
            )
            row["calls"] += 1
            for counter_name, count in token_counts.items():
                row[counter_name] += count

    def by_role(self) -> dict[str, dict]:
        """{role: {calls, prompt_tokens, completion_tokens, cached_prompt_tokens, models: [...]}}."""
        totals_by_role: dict[str, dict] = {}
        with self._lock:
            for (role, model), row in self.rows.items():
                role_totals = totals_by_role.setdefault(
                    role,
                    {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0, "models": []},
                )
                for counter_name in ("calls", "prompt_tokens", "completion_tokens", "cached_prompt_tokens"):
                    role_totals[counter_name] += row[counter_name]
                role_totals["models"].append(model)
        return totals_by_role


class BudgetExceeded(RuntimeError):
    """Raised by SpendGuard before a model call once the run has spent its --max-cost-usd."""


class SpendGuard:
    """Stops a run from spending more than ``limit_usd``: call it before every model call.

    ``spent_usd`` returns the cost metered so far. Calls already in flight when the limit is
    crossed still finish, so the final cost can pass the limit by a few cents.
    """

    def __init__(self, limit_usd: float, spent_usd: Callable[[], float]) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self.hit = False

    def __call__(self) -> None:
        if self.hit or self.spent_usd() >= self.limit_usd:
            self.hit = True
            raise BudgetExceeded(f"run stopped: spent the --max-cost-usd limit of ${self.limit_usd:.2f}")


def cost_usd(prompt: int, cached: int, completion: int, price: tuple[float, float, float]) -> float:
    input_price, cached_input_price, output_price = price
    cached = min(cached, prompt)
    uncached = prompt - cached
    return (uncached * input_price + cached * cached_input_price + completion * output_price) / 1e6


def hit_rate(prompt: int, cached: int) -> float | None:
    if prompt:
        return round(cached / prompt, 4)
    return None


def estimate(
    meter_rows: dict[tuple[str, str], dict],
    providers: dict[str, str],
    role_names: dict[str, str] | None = None,
) -> dict:
    """Cost estimate per role, from ``UsageMeter.rows`` and the provider of each model.

    ``providers`` maps model -> provider (``provider_of``). ``role_names`` renames roles in the
    output (teacher runs: ``{"agent": "teacher"}``).
    """
    role_names = role_names or {}
    entry_by_role_name: dict[str, dict] = {}
    total_usd = 0.0
    for (role, model), row in sorted(meter_rows.items()):
        role_name = role_names.get(role, role)
        provider = providers.get(model) or provider_of(model)
        price, price_is_assumed = price_for(model, provider)
        model_usd = cost_usd(row["prompt_tokens"], row["cached_prompt_tokens"], row["completion_tokens"], price)
        total_usd += model_usd

        role_entry = entry_by_role_name.setdefault(role_name, {"usd": 0.0, "models": []})
        role_entry["usd"] = round(role_entry["usd"] + model_usd, 6)
        role_entry["models"].append({
            "model": model,
            "provider": provider,
            **row,
            "cache_hit_rate": hit_rate(row["prompt_tokens"], row["cached_prompt_tokens"]),
            "usd_per_1m": {"input": price[0], "cached_input": price[1], "output": price[2]},
            "price_assumed": price_is_assumed,
            "usd": round(model_usd, 6),
        })

    estimate_by_role: dict[str, Any] = {name: round(entry["usd"], 4) for name, entry in entry_by_role_name.items()}
    estimate_by_role["total"] = round(total_usd, 4)
    estimate_by_role["by_role"] = entry_by_role_name
    estimate_by_role["note"] = (
        "estimate from metered token counts and list prices in src/shipwright/llm/costs.py;"
        " check HF and GCP billing for the real number"
    )
    return estimate_by_role

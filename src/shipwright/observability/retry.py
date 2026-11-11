"""One retry behaviour for every external call (model APIs, token fetch, GCS, HF Hub, HF Jobs).

    from shipwright.observability import retry

    retry.call_with_retry(fn, *args, what="gcs_upload gs://...", **kwargs)

    @retry.retry(what="hf_hub_upload")
    def push(...): ...

Rules (AGENTS.md: every external call goes through this module, never a new ad-hoc loop):

- Only transient errors are retried: HTTP 408/409/425/429/500/502/503/504, connection reset or
  timeout errors, a JSON decode error of a model reply, and litellm's RateLimit / APIConnection /
  Timeout / ServiceUnavailable / InternalServer errors. A 400 is retried only when the policy
  says so (``retry_bad_request``: the Vertex gpt-oss user simulator and judge answer some valid
  requests with a 400 that passes on a retry). 401/403/404 and other client errors raise at once.
- Exponential backoff with jitter, at most ``attempts`` tries and ``max_total_wait`` seconds of
  waiting. A ``Retry-After`` header (seconds) is respected, capped by ``max_wait``.
- Each retry logs one structured line (attempt, wait, error class) and adds a span event.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import random
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
PERMANENT_STATUS = frozenset({400, 401, 403, 404, 405, 410, 413, 422})
# Exception class names (any class in the MRO) that mean "try again".
TRANSIENT_NAMES = frozenset({
    # litellm / openai
    "RateLimitError", "APIConnectionError", "APITimeoutError", "Timeout", "ServiceUnavailableError",
    "InternalServerError",
    # httpx / requests / stdlib
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ReadError", "WriteError",
    "RemoteProtocolError", "ConnectionError", "ConnectionResetError", "ConnectionAbortedError",
    "BrokenPipeError", "TimeoutError", "TimeoutExpired", "ChunkedEncodingError",
    # a model reply that is not the JSON it must be
    "JSONDecodeError",
})
# Connection errors where the request surely never reached the server (safe for non-idempotent calls).
NOT_SENT_NAMES = frozenset({"ConnectError", "ConnectTimeout", "APIConnectionError", "ConnectionRefusedError"})
NOT_SENT_STATUS = frozenset({425, 429, 502, 503})


@dataclasses.dataclass(frozen=True)
class Policy:
    attempts: int = 5             # tries in total, the first one included
    base: float = 2.0             # first wait (seconds); doubles each retry
    max_wait: float = 60.0        # cap on one wait
    max_total_wait: float = 180.0  # cap on all waits of one call
    retry_bad_request: bool = False  # also retry HTTP 400 (gpt-oss user simulator and judge)
    idempotent: bool = True       # False: retry only errors where the request was not processed


DEFAULT = Policy()
# Model calls: Vertex gpt-oss returns 429 at a few concurrent tau2 runs (2026-11-11).
LLM = Policy(attempts=6, base=2.0, max_wait=60.0, max_total_wait=240.0)
LLM_USER = dataclasses.replace(LLM, retry_bad_request=True)
# Launching a paid HF Job: never risk a second job when the first request may have gone through.
LAUNCH = Policy(attempts=4, base=5.0, max_total_wait=120.0, idempotent=False)


def status_of(exc: BaseException) -> int | None:
    """HTTP status of an exception from litellm, openai, httpx, requests or huggingface_hub."""
    for obj in (exc, getattr(exc, "response", None)):
        if obj is None:
            continue
        for attr in ("status_code", "status"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and 100 <= v < 600:
                return v
    return None


def _names(exc: BaseException) -> set[str]:
    return {c.__name__ for c in type(exc).__mro__}


def is_transient(exc: BaseException, policy: Policy = DEFAULT) -> bool:
    status = status_of(exc)
    names = _names(exc)
    if not policy.idempotent:
        return status in NOT_SENT_STATUS or bool(names & NOT_SENT_NAMES)
    if status is not None:
        if status == 400 and policy.retry_bad_request:
            return True
        if status in TRANSIENT_STATUS:
            return True
        if status in PERMANENT_STATUS or 400 <= status < 500:
            return False
    if isinstance(exc, json.JSONDecodeError):
        return True
    return bool(names & TRANSIENT_NAMES)


def retry_after(exc: BaseException) -> float | None:
    """Seconds from a Retry-After header (numbers only), if the error carries one."""
    for obj in (exc, getattr(exc, "response", None)):
        headers = getattr(obj, "headers", None)
        if not headers:
            continue
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:  # noqa: BLE001
            continue
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    return None


def backoff(attempt: int, policy: Policy, rng: Callable[[], float] = random.random) -> float:
    """Wait before retry number ``attempt`` (1-based): base * 2^(attempt-1), jitter 0.5x to 1.5x."""
    return min(policy.max_wait, policy.base * (2 ** (attempt - 1)) * (0.5 + rng()))


def call_with_retry(fn: Callable[..., T], *args: Any, policy: Policy = DEFAULT, what: str = "",
                    sleep: Callable[[float], None] = time.sleep, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)``, retried on transient errors under ``policy``."""
    from shipwright.observability import log, otel_live

    logger = log.get_logger("shipwright.retry")
    waited = 0.0
    attempt = 1
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt >= policy.attempts or not is_transient(e, policy):
                raise
            wait = backoff(attempt, policy)
            hinted = retry_after(e)
            if hinted is not None:
                wait = min(policy.max_wait, max(wait, hinted))
            if waited + wait > policy.max_total_wait:
                raise
            fields = {"what": what or getattr(fn, "__name__", "call"), "attempt": attempt,
                      "wait_s": round(wait, 2), "error_class": type(e).__name__, "status": status_of(e)}
            logger.warning(f"retry {fields['what']}", **fields, error=str(e)[:300])
            try:
                from opentelemetry import trace as trace_api

                trace_api.get_current_span().add_event("retry", otel_live._clean(fields))
            except Exception:  # noqa: BLE001
                pass
            sleep(wait)
            waited += wait
            attempt += 1


def retry(policy: Policy = DEFAULT, what: str = "") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of ``call_with_retry``."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(fn, *args, policy=policy, what=what or fn.__name__, **kwargs)

        return wrapper

    return deco

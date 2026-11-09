"""Google Vertex AI models (OpenAI-compatible MaaS endpoint) for the teacher, user simulator and judge.

Model ids: ``vertex/<publisher>/<model>``, for example
``vertex/deepseek-ai/deepseek-v4.1-flash-maas`` or ``vertex/openai/gpt-oss-120b-maas``.
Every other id (HF router ids such as ``openai/openai/gpt-oss-120b:cerebras``) is unchanged.

A ``vertex/`` request goes to litellm as an OpenAI-compatible model, with the Vertex
``api_base`` and a fresh bearer token per call (``route_kwargs``). Nothing token-bearing goes
into tau2 config or env vars.

Auth: ``gcloud --configuration=$SHIPWRIGHT_GCLOUD_CONFIG auth print-access-token``. Never
Application Default Credentials: on Tejas's machine they are a work identity. A token lives
about 1 hour, so ``TokenProvider`` caches it and gets a new one after 45 minutes (or after a
401). ``$SHIPWRIGHT_VERTEX_TOKEN`` (a token minted elsewhere, e.g. passed to an HF Job) is used
as is, with no refresh. The token is never logged or written to disk.

Env vars:
  SHIPWRIGHT_GCP_PROJECT     default lexsi-hackathon
  SHIPWRIGHT_GCLOUD_CONFIG   default personal (teammates: lexsi)
  SHIPWRIGHT_GCLOUD          gcloud binary (default: the Homebrew path if it exists, else PATH)
  SHIPWRIGHT_VERTEX_LOCATION default global (DeepSeek V4.1 Flash is served only there)
  SHIPWRIGHT_VERTEX_TOKEN    a static bearer token (no gcloud call, no refresh)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

PREFIX = "vertex/"
DEFAULT_PROJECT = "lexsi-hackathon"
DEFAULT_GCLOUD_CONFIG = "personal"
DEFAULT_LOCATION = "global"
HOMEBREW_GCLOUD = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
# Tokens from gcloud live about 1 hour; refresh well before that.
TOKEN_MAX_AGE_S = 45 * 60
# gpt-oss spends tokens on reasoning before it answers: a small max_tokens leaves no answer.
MIN_REPLY_TOKENS = {"openai/gpt-oss-120b-maas": 1024}

# The team's current choice (2026-11-09). Scripts still take these as explicit arguments.
TEACHER_MODEL = "vertex/deepseek-ai/deepseek-v4.1-flash-maas"
USER_MODEL = "vertex/openai/gpt-oss-120b-maas"
JUDGE_MODEL = USER_MODEL


def is_vertex_model(model: str | None) -> bool:
    return bool(model) and str(model).startswith(PREFIX)


def model_id(model: str) -> str:
    """``vertex/deepseek-ai/x`` -> ``deepseek-ai/x`` (the id Vertex expects)."""
    if not is_vertex_model(model):
        raise ValueError(f"not a vertex model id: {model!r}")
    rest = model[len(PREFIX):]
    if "/" not in rest:
        raise ValueError(f"vertex model id must be vertex/<publisher>/<model>: {model!r}")
    return rest


def project() -> str:
    return os.environ.get("SHIPWRIGHT_GCP_PROJECT") or DEFAULT_PROJECT


def location() -> str:
    return os.environ.get("SHIPWRIGHT_VERTEX_LOCATION") or DEFAULT_LOCATION


def api_base(project_id: str | None = None, loc: str | None = None) -> str:
    """OpenAI-compatible base URL; litellm and our clients append ``/chat/completions``."""
    project_id = project_id or project()
    loc = loc or location()
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project_id}/locations/{loc}/endpoints/openapi"


def gcloud_binary() -> str:
    explicit = os.environ.get("SHIPWRIGHT_GCLOUD")
    if explicit:
        return explicit
    if Path(HOMEBREW_GCLOUD).exists():
        return HOMEBREW_GCLOUD
    return shutil.which("gcloud") or "gcloud"


def gcloud_token() -> str:
    """A fresh access token from gcloud, with the named config (never ADC)."""
    config = os.environ.get("SHIPWRIGHT_GCLOUD_CONFIG") or DEFAULT_GCLOUD_CONFIG
    cmd = [gcloud_binary(), f"--configuration={config}", "auth", "print-access-token"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"gcloud failed ({' '.join(cmd[:2])}): {e}") from None
    token = out.stdout.strip()
    if out.returncode != 0 or not token:
        raise RuntimeError(
            f"gcloud --configuration={config} auth print-access-token failed (exit {out.returncode}):"
            f" {out.stderr.strip()[:300]}. Set SHIPWRIGHT_GCLOUD_CONFIG to a config with access"
            f" to {project()} (Tejas: personal, others: lexsi)."
        )
    return token


def default_fetch() -> str:
    static = os.environ.get("SHIPWRIGHT_VERTEX_TOKEN", "").strip()
    return static or gcloud_token()


class TokenProvider:
    """Cache a bearer token and fetch a new one after ``max_age`` seconds. Thread-safe."""

    def __init__(self, fetch: Callable[[], str] | None = None, max_age: float = TOKEN_MAX_AGE_S,
                 clock: Callable[[], float] = time.monotonic):
        self._fetch = fetch or default_fetch
        self._max_age = max_age
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._at = 0.0
        self.fetches = 0

    def get(self) -> str:
        with self._lock:
            if self._token is None or self._clock() - self._at >= self._max_age:
                self._token = self._fetch()
                self._at = self._clock()
                self.fetches += 1
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token = None

    def __repr__(self) -> str:  # never show the token
        return f"TokenProvider(fetches={self.fetches})"


_provider: TokenProvider | None = None
_provider_lock = threading.Lock()


def provider() -> TokenProvider:
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = TokenProvider()
        return _provider


def route_kwargs(kwargs: dict, tokens: TokenProvider | None = None) -> dict:
    """litellm kwargs for a ``vertex/`` model: OpenAI-compatible model, Vertex base, fresh token."""
    model = str(kwargs.get("model", ""))
    if not is_vertex_model(model):
        return kwargs
    mid = model_id(model)
    out = dict(kwargs)
    out["model"] = "openai/" + mid
    out["api_base"] = api_base()
    out["api_key"] = (tokens or provider()).get()
    floor = MIN_REPLY_TOKENS.get(mid)
    if floor and isinstance(out.get("max_tokens"), int) and out["max_tokens"] < floor:
        out["max_tokens"] = floor
    # Params added for OpenAI's Responses bridge do not apply here.
    out.pop("allowed_openai_params", None)
    return out


def is_auth_error(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 401 or "401" in type(exc).__name__


def call_with_refresh(call: Callable[..., Any], kwargs: dict, tokens: TokenProvider | None = None) -> Any:
    """Run ``call(**route_kwargs(kwargs))``; on a 401, get a new token and retry once."""
    tokens = tokens or provider()
    try:
        return call(**route_kwargs(kwargs, tokens))
    except Exception as e:  # noqa: BLE001
        if not is_auth_error(e):
            raise
        tokens.invalidate()
        return call(**route_kwargs(kwargs, tokens))


def endpoint(model: str, hf_token: str | None, hf_base: str) -> tuple[str, str, Callable[[], dict]]:
    """(chat completions URL, model name for the body, headers factory) for a direct HTTP client.

    ``model`` is a raw router id (``openai/gpt-oss-120b:cerebras``) or a ``vertex/`` id.
    """
    if is_vertex_model(model):
        tokens = provider()
        tokens.get()  # fail now (RuntimeError) if gcloud cannot mint a token
        return f"{api_base()}/chat/completions", model_id(model), lambda: {"Authorization": f"Bearer {tokens.get()}"}
    if not hf_token:
        raise RuntimeError("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.")
    return f"{hf_base}/chat/completions", model, lambda: {"Authorization": f"Bearer {hf_token}"}

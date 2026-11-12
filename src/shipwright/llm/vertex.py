"""Google Vertex AI models (OpenAI-compatible MaaS endpoint) for the teacher, user simulator and judge.

Model ids: ``vertex/<publisher>/<model>``, for example
``vertex/deepseek-ai/deepseek-v4.1-flash-maas`` or ``vertex/openai/gpt-oss-120b-maas``.
Every other id (HF router ids such as ``openai/openai/gpt-oss-120b:cerebras``) is unchanged.

A ``vertex/`` request goes to litellm as an OpenAI-compatible model, with the Vertex
``api_base`` and a fresh bearer token per call (``route_kwargs``). Nothing token-bearing goes
into tau2 config or env vars.

Auth, in this order:
  1. ``$SHIPWRIGHT_VERTEX_TOKEN`` (a token minted elsewhere, e.g. passed to an HF Job): used as
     is, with no refresh.
  2. On GCP (Cloud Run: ``$CLOUD_RUN_JOB`` or ``$K_SERVICE`` set, or
     ``SHIPWRIGHT_VERTEX_AUTH=metadata``): the service account token from the metadata server.
     It comes with ``expires_in``; ``TokenProvider`` refreshes 5 minutes before that.
  3. Laptops: ``gcloud --configuration=$SHIPWRIGHT_GCLOUD_CONFIG auth print-access-token``. Never
     Application Default Credentials: on Tejas's machine they are a work identity.
A token lives about 1 hour, so ``TokenProvider`` caches it and gets a new one after 45 minutes
(or after a 401). The token is never logged or written to disk.

Env vars:
  SHIPWRIGHT_GCP_PROJECT     default lexsi-hackathon
  SHIPWRIGHT_GCLOUD_CONFIG   default personal (teammates: lexsi)
  SHIPWRIGHT_GCLOUD          gcloud binary (default: the Homebrew path if it exists, else PATH)
  SHIPWRIGHT_VERTEX_LOCATION default global (DeepSeek V4.1 Flash is served only there)
  SHIPWRIGHT_VERTEX_TOKEN    a static bearer token (no gcloud call, no refresh)
  SHIPWRIGHT_VERTEX_AUTH     metadata: use the metadata server token (set in the harness image)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from shipwright.observability import retry

PREFIX = "vertex/"
DEFAULT_PROJECT = "lexsi-hackathon"
DEFAULT_GCLOUD_CONFIG = "personal"
DEFAULT_LOCATION = "global"
HOMEBREW_GCLOUD = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
# Tokens from gcloud live about 1 hour; refresh well before that.
TOKEN_MAX_AGE_S = 45 * 60
METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
# Token fetches (metadata server, gcloud) are short calls: retry them quickly.
TOKEN_POLICY = retry.Policy(attempts=4, base=1.0, max_wait=10.0, max_total_wait=30.0)


# Refresh a metadata token this long before it expires.
TOKEN_EXPIRY_MARGIN_S = 5 * 60
# gpt-oss spends tokens on reasoning before it answers: a small max_tokens leaves no answer.
MIN_REPLY_TOKENS = {"openai/gpt-oss-120b-maas": 1024}
# gpt-oss reasons before it answers; "low" cuts that (and the reply time) at no cost to the simulator's job.
REASONING_EFFORT = {"openai/gpt-oss-120b-maas": "low"}
# Seconds before a hung request is abandoned and retried. Round 2 saw gpt-oss calls hang for up to
# 20 minutes (the litellm default) while its median reply takes a few seconds.
REQUEST_TIMEOUT_S = {"openai/gpt-oss-120b-maas": 60.0}
DEFAULT_REQUEST_TIMEOUT_S = 180.0

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
    publisher_and_model = model[len(PREFIX):]
    if "/" not in publisher_and_model:
        raise ValueError(f"vertex model id must be vertex/<publisher>/<model>: {model!r}")
    return publisher_and_model


def project() -> str:
    return os.environ.get("SHIPWRIGHT_GCP_PROJECT") or DEFAULT_PROJECT


def location() -> str:
    return os.environ.get("SHIPWRIGHT_VERTEX_LOCATION") or DEFAULT_LOCATION


def api_base(project_id: str | None = None, vertex_location: str | None = None) -> str:
    """OpenAI-compatible base URL; litellm and our clients append ``/chat/completions``."""
    project_id = project_id or project()
    vertex_location = vertex_location or location()
    if vertex_location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{vertex_location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project_id}/locations/{vertex_location}/endpoints/openapi"


def gcloud_binary() -> str:
    explicit = os.environ.get("SHIPWRIGHT_GCLOUD")
    if explicit:
        return explicit
    if Path(HOMEBREW_GCLOUD).exists():
        return HOMEBREW_GCLOUD
    return shutil.which("gcloud") or "gcloud"


def gcloud_token() -> str:
    """A fresh access token from gcloud, with the named config (never ADC)."""
    gcloud_config = os.environ.get("SHIPWRIGHT_GCLOUD_CONFIG") or DEFAULT_GCLOUD_CONFIG
    command = [gcloud_binary(), f"--configuration={gcloud_config}", "auth", "print-access-token"]
    try:
        completed_process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as run_error:
        raise RuntimeError(f"gcloud failed ({' '.join(command[:2])}): {run_error}") from None
    token = completed_process.stdout.strip()
    if completed_process.returncode != 0 or not token:
        raise RuntimeError(
            f"gcloud --configuration={gcloud_config} auth print-access-token failed"
            f" (exit {completed_process.returncode}): {completed_process.stderr.strip()[:300]}. Set SHIPWRIGHT_GCLOUD_CONFIG to a config with access"
            f" to {project()} (Tejas: personal, others: lexsi)."
        )
    return token


def on_gcp() -> bool:
    """True inside Cloud Run (job or service), or when ``SHIPWRIGHT_VERTEX_AUTH=metadata``."""
    if os.environ.get("SHIPWRIGHT_VERTEX_AUTH", "").strip().lower() == "metadata":
        return True
    return bool(os.environ.get("CLOUD_RUN_JOB") or os.environ.get("K_SERVICE"))


def metadata_token(get: Callable[..., Any] | None = None) -> tuple[str, float]:
    """(access token, seconds until it expires) for the attached service account.

    HTTP and connection errors propagate as they are, so the retry helper can tell a transient one."""
    import httpx

    http_get = get or httpx.get
    response = http_get(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"}, timeout=10)
    response.raise_for_status()
    response_body = response.json()
    token = str(response_body.get("access_token") or "")
    if not token:
        raise RuntimeError("metadata server returned no access_token")
    return token, float(response_body.get("expires_in") or TOKEN_MAX_AGE_S)


def default_fetch() -> str | tuple[str, float]:
    static_token = os.environ.get("SHIPWRIGHT_VERTEX_TOKEN", "").strip()
    if static_token:
        return static_token
    if on_gcp():
        return metadata_token()
    return gcloud_token()


class TokenProvider:
    """Cache a bearer token and fetch a new one after ``max_age`` seconds. Thread-safe.

    ``fetch`` returns a token, or ``(token, expires_in)``: then the token is replaced
    ``TOKEN_EXPIRY_MARGIN_S`` before it expires, if that is sooner than ``max_age``.
    """

    def __init__(
        self,
        fetch: Callable[[], str | tuple[str, float]] | None = None,
        max_age: float = TOKEN_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._fetch = fetch or default_fetch
        self._max_age = max_age
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._fetched_at = 0.0
        self._time_to_live = max_age
        self.fetches = 0

    def get(self) -> str:
        with self._lock:
            token_is_stale = self._clock() - self._fetched_at >= self._time_to_live
            if self._token is None or token_is_stale:
                from shipwright.observability import instrument

                fetch_name = getattr(self._fetch, "__name__", "fetch")
                with instrument.external("token_fetch", fetch_name, refresh=self.fetches > 0):
                    try:
                        fetch_result = retry.call_with_retry(self._fetch, policy=TOKEN_POLICY,
                                                             what=f"token_fetch {fetch_name}")
                    except RuntimeError:
                        raise
                    except Exception as fetch_error:  # callers expect RuntimeError for "no token"
                        raise RuntimeError(
                            f"{fetch_name} token failed: {type(fetch_error).__name__}: {str(fetch_error)[:200]}"
                        ) from None
                if isinstance(fetch_result, tuple):
                    self._token, expires_in = fetch_result
                    self._time_to_live = max(0.0, min(self._max_age, expires_in - TOKEN_EXPIRY_MARGIN_S))
                else:
                    self._token = fetch_result
                    self._time_to_live = self._max_age
                self._fetched_at = self._clock()
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
    vertex_model_id = model_id(model)
    routed_kwargs = dict(kwargs)
    routed_kwargs["model"] = "openai/" + vertex_model_id
    routed_kwargs["api_base"] = api_base()
    routed_kwargs["api_key"] = (tokens or provider()).get()
    min_reply_tokens = MIN_REPLY_TOKENS.get(vertex_model_id)
    max_tokens = routed_kwargs.get("max_tokens")
    if min_reply_tokens and isinstance(max_tokens, int) and max_tokens < min_reply_tokens:
        routed_kwargs["max_tokens"] = min_reply_tokens
    effort = REASONING_EFFORT.get(vertex_model_id)
    if effort and "reasoning_effort" not in routed_kwargs:
        routed_kwargs["reasoning_effort"] = effort
    if "timeout" not in routed_kwargs:
        routed_kwargs["timeout"] = REQUEST_TIMEOUT_S.get(vertex_model_id, DEFAULT_REQUEST_TIMEOUT_S)
    # Params added for OpenAI's Responses bridge do not apply here.
    routed_kwargs.pop("allowed_openai_params", None)
    return routed_kwargs


def is_auth_error(error: BaseException) -> bool:
    return getattr(error, "status_code", None) == 401 or "401" in type(error).__name__


def call_with_refresh(call: Callable[..., Any], kwargs: dict, tokens: TokenProvider | None = None,
                      sleep: Callable[[float], None] = time.sleep, policy: Any = None) -> Any:
    """Run ``call(**route_kwargs(kwargs))``; on a 401, get a new token and try again at once.
    Transient errors (429, 5xx, timeouts, connection errors) retry through ``shipwright.observability.retry``
    with ``policy`` (default ``retry.LLM``; the user simulator and judge pass ``retry.LLM_USER``,
    which also retries the gpt-oss 400s)."""
    tokens = tokens or provider()

    def once() -> Any:
        try:
            return call(**route_kwargs(kwargs, tokens))
        except Exception as e:  # noqa: BLE001
            if not is_auth_error(e):
                raise
            tokens.invalidate()
            return call(**route_kwargs(kwargs, tokens))

    return retry.call_with_retry(once, policy=policy or retry.LLM, what=f"llm {kwargs.get('model')}", sleep=sleep)


def endpoint(model: str, hf_token: str | None, hf_base: str) -> tuple[str, str, Callable[[], dict]]:
    """(chat completions URL, model name for the body, headers factory) for a direct HTTP client.

    ``model`` is a raw router id (``openai/gpt-oss-120b:cerebras``) or a ``vertex/`` id.
    """
    if is_vertex_model(model):
        tokens = provider()
        tokens.get()  # fail now (RuntimeError) if no token can be had

        def vertex_headers() -> dict:
            return {"Authorization": f"Bearer {tokens.get()}"}

        return f"{api_base()}/chat/completions", model_id(model), vertex_headers
    if not hf_token:
        raise RuntimeError("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.")

    def hf_headers() -> dict:
        return {"Authorization": f"Bearer {hf_token}"}

    return f"{hf_base}/chat/completions", model, hf_headers

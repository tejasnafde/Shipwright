"""Vertex AI routing, token cache, cached-token cost math. Offline: gcloud and HTTP are mocked."""

from __future__ import annotations

import subprocess

import pytest

from shipwright import costs, vertex
from shipwright.llm_patch import call_role
from shipwright.traces import build_trace, summarize

TEACHER = "vertex/deepseek-ai/deepseek-v4.1-flash-maas"
USER = "vertex/openai/gpt-oss-120b-maas"
HF_USER = "openai/openai/gpt-oss-120b:cerebras"


# ------------------------------------------------------------------ ids and routing

def test_model_id_parsing():
    assert vertex.is_vertex_model(TEACHER)
    assert not vertex.is_vertex_model(HF_USER)
    assert not vertex.is_vertex_model(None)
    assert vertex.model_id(TEACHER) == "deepseek-ai/deepseek-v4.1-flash-maas"
    with pytest.raises(ValueError):
        vertex.model_id("vertex/no-publisher")
    with pytest.raises(ValueError):
        vertex.model_id(HF_USER)


def test_api_base_global_and_regional(monkeypatch):
    monkeypatch.delenv("SHIPWRIGHT_GCP_PROJECT", raising=False)
    monkeypatch.delenv("SHIPWRIGHT_VERTEX_LOCATION", raising=False)
    assert vertex.api_base() == (
        "https://aiplatform.googleapis.com/v1/projects/lexsi-hackathon/locations/global/endpoints/openapi")
    monkeypatch.setenv("SHIPWRIGHT_GCP_PROJECT", "p2")
    monkeypatch.setenv("SHIPWRIGHT_VERTEX_LOCATION", "us-central1")
    assert vertex.api_base() == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p2/locations/us-central1/endpoints/openapi")


def test_route_kwargs_vertex_and_passthrough():
    tokens = vertex.TokenProvider(fetch=lambda: "tok-1")
    out = vertex.route_kwargs({"model": TEACHER, "messages": [], "max_tokens": 4096}, tokens)
    assert out["model"] == "openai/deepseek-ai/deepseek-v4.1-flash-maas"
    assert out["api_base"].endswith("/locations/global/endpoints/openapi")
    assert out["api_key"] == "tok-1"
    assert out["max_tokens"] == 4096
    # gpt-oss reasons first: a small max_tokens is raised.
    out = vertex.route_kwargs({"model": USER, "messages": [], "max_tokens": 700}, tokens)
    assert out["max_tokens"] == 1024 and out["model"] == "openai/openai/gpt-oss-120b-maas"
    hf = {"model": HF_USER, "messages": []}
    assert vertex.route_kwargs(hf, tokens) is hf  # HF router ids are unchanged


def test_endpoint_hf_and_vertex(monkeypatch):
    url, body_model, headers = vertex.endpoint(HF_USER.removeprefix("openai/"), "hf-tok", "https://router.x/v1")
    assert url == "https://router.x/v1/chat/completions" and body_model == "openai/gpt-oss-120b:cerebras"
    assert headers() == {"Authorization": "Bearer hf-tok"}
    with pytest.raises(RuntimeError):
        vertex.endpoint("openai/gpt-oss-120b:cerebras", None, "https://router.x/v1")
    monkeypatch.setattr(vertex, "_provider", vertex.TokenProvider(fetch=lambda: "gtok"))
    url, body_model, headers = vertex.endpoint(USER, None, "unused")
    assert url.endswith("/endpoints/openapi/chat/completions") and body_model == "openai/gpt-oss-120b-maas"
    assert headers() == {"Authorization": "Bearer gtok"}


# ------------------------------------------------------------------ token cache and refresh

def test_token_cached_then_refreshed_after_max_age():
    now = [0.0]
    issued = iter(["t1", "t2", "t3"])
    tokens = vertex.TokenProvider(fetch=lambda: next(issued), max_age=45 * 60, clock=lambda: now[0])
    assert tokens.get() == "t1"
    now[0] = 44 * 60
    assert tokens.get() == "t1" and tokens.fetches == 1
    now[0] = 45 * 60
    assert tokens.get() == "t2" and tokens.fetches == 2
    tokens.invalidate()
    assert tokens.get() == "t3"
    assert "t3" not in repr(tokens)


def test_call_with_refresh_retries_once_on_401():
    issued = iter(["old", "new"])
    tokens = vertex.TokenProvider(fetch=lambda: next(issued))
    seen = []

    class Unauthorized(Exception):
        status_code = 401

    def call(**kw):
        seen.append(kw["api_key"])
        if kw["api_key"] == "old":
            raise Unauthorized()
        return "ok"

    assert vertex.call_with_refresh(call, {"model": TEACHER, "messages": []}, tokens) == "ok"
    assert seen == ["old", "new"]

    def boom(**kw):
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        vertex.call_with_refresh(boom, {"model": TEACHER, "messages": []}, tokens)


def test_gcloud_token_uses_named_config_not_adc(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ya29.secret\n", stderr="")

    monkeypatch.setattr(vertex.subprocess, "run", fake_run)
    monkeypatch.setenv("SHIPWRIGHT_GCLOUD", "/bin/gcloud-x")
    monkeypatch.setenv("SHIPWRIGHT_GCLOUD_CONFIG", "lexsi")
    assert vertex.gcloud_token() == "ya29.secret"
    assert calls == [["/bin/gcloud-x", "--configuration=lexsi", "auth", "print-access-token"]]
    monkeypatch.delenv("SHIPWRIGHT_GCLOUD_CONFIG")
    vertex.gcloud_token()
    assert calls[-1][1] == "--configuration=personal"


def test_gcloud_failure_message_has_no_token(monkeypatch):
    monkeypatch.setattr(vertex.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="ERROR: no credentialed accounts"))
    with pytest.raises(RuntimeError, match="no credentialed accounts"):
        vertex.gcloud_token()


def test_static_token_env_skips_gcloud(monkeypatch):
    monkeypatch.setenv("SHIPWRIGHT_VERTEX_TOKEN", "static")
    monkeypatch.setattr(vertex.subprocess, "run", lambda *a, **k: pytest.fail("gcloud must not run"))
    assert vertex.default_fetch() == "static"


# ------------------------------------------------------------------ cost math and cached tokens

def test_cached_tokens_fields():
    assert costs.cached_tokens(None) is None
    assert costs.cached_tokens({"prompt_tokens": 5, "prompt_tokens_details": None}) is None
    assert costs.cached_tokens({"prompt_tokens_details": {"cached_tokens": 4096}}) == 4096
    assert costs.cached_tokens({"cachedContentTokenCount": 10}) == 10
    # litellm default 0 in one field, the hit reported elsewhere: take the hit.
    assert costs.cached_tokens({"prompt_tokens_details": {"cached_tokens": 0},
                                "extra_properties": {"google": {"cached_content_token_count": 300}}}) == 300


def test_cost_with_cached_tokens():
    price = costs.VERTEX_PRICES["deepseek-ai/deepseek-v4.1-flash-maas"]
    # 1M prompt tokens, 800k cached, 100k output: 0.2*0.30 + 0.8*0.03 + 0.1*1.20
    assert costs.cost_usd(1_000_000, 800_000, 100_000, price) == pytest.approx(0.06 + 0.024 + 0.12)
    assert costs.cost_usd(100, 500, 0, (1.0, 0.0, 0.0)) == pytest.approx(0.0)  # cached capped at prompt


def test_estimate_per_role_and_provider():
    meter = costs.UsageMeter()
    meter.add("agent", TEACHER, {"prompt_tokens": 1_000_000, "completion_tokens": 0,
                                 "prompt_tokens_details": {"cached_tokens": 500_000}})
    meter.add("user", USER, {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    meter.add("judge", USER, {"prompt_tokens": 1_000_000, "completion_tokens": 0})
    meter.add("judge", HF_USER, {"prompt_tokens": 1_000_000, "completion_tokens": 0})
    out = costs.estimate(meter.rows, {}, {"agent": "teacher"})
    assert out["teacher"] == pytest.approx(0.5 * 0.30 + 0.5 * 0.03)
    assert out["user"] == pytest.approx(0.09 + 0.36)
    assert out["judge"] == pytest.approx(0.09 + 0.25)
    assert out["total"] == pytest.approx(0.165 + 0.45 + 0.34)
    teacher = out["by_role"]["teacher"]["models"][0]
    assert teacher["provider"] == "vertex" and teacher["cache_hit_rate"] == 0.5
    providers = {m["model"]: m["provider"] for m in out["by_role"]["judge"]["models"]}
    assert providers == {USER: "vertex", HF_USER: "hf_router"}


def test_provider_and_assumed_prices():
    assert costs.provider_of(TEACHER) == "vertex"
    assert costs.provider_of(HF_USER) == "hf_router"
    assert costs.provider_of("openai/tiny-aya-x", api_base="http://localhost:8000/v1") == "local"
    assert costs.provider_of(TEACHER, dry_run=True) == "dry_run"
    assert costs.price_for("vertex/x/unknown-maas", "vertex") == (costs.UNKNOWN_GUESS, True)
    assert costs.price_for("openai/CohereLabs/tiny-aya-global:cohere", "hf_router")[1] is True


def test_call_role():
    agent = {"model": TEACHER, "tools": [{}], "messages": [{"role": "system", "content": "policy"}]}
    assert call_role(agent, agent, TEACHER, USER) == "agent"
    user = {"model": USER, "messages": [{"role": "system", "content": "you are a customer"}]}
    assert call_role(user, user, TEACHER, USER) == "user"
    judge = {"model": USER, "messages": [{"role": "system", "content": "... metExpectation ..."}]}
    assert call_role(judge, judge, TEACHER, USER) == "judge"
    rerouted_from = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "x"}]}
    assert call_role(rerouted_from, {**rerouted_from, "model": USER}, TEACHER, USER) == "judge"
    # Teacher and user simulator are the same model: the agent call has tools.
    same = {**agent, "model": USER}
    assert call_role(same, same, USER, USER) == "agent"
    assert call_role(user, user, USER, USER) == "user"


def test_trace_records_cached_tokens_from_raw_usage():
    raw = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 5000, "completion_tokens": 10,
                     "prompt_tokens_details": {"cached_tokens": 4096}}}
    sim = {"id": "s", "task_id": "1", "reward_info": {"reward": 1.0}, "termination_reason": "user_stop",
           "messages": [
               {"role": "assistant", "content": "hi", "raw_data": raw,
                "usage": {"prompt_tokens": 5000, "completion_tokens": 10}},
               {"role": "user", "content": "ok", "raw_data": {**raw, "usage": {"prompt_tokens": 100,
                                                                                "completion_tokens": 5}},
                "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
           ]}
    trace = build_trace(sim, set())
    assert trace["agent_turns"][0]["usage"]["cached_prompt_tokens"] == 4096
    assert "cached_prompt_tokens" not in trace["messages"][1]["usage"]  # not reported: not invented
    s = summarize([trace])
    assert s["cached_prompt_tokens"] == {"agent": 4096, "user": 0}
    assert s["cache_hit_rate"]["agent"] == pytest.approx(0.8192)


def test_teacher_prefix_is_byte_identical_across_requests():
    """Implicit prefix caching needs the same system message bytes on every teacher request."""
    from shipwright import prompted_tools

    prompted_tools.force_prompted(TEACHER, 4096)
    tools = [{"type": "function", "function": {"name": "get_order_details", "description": "Get an order.",
                                               "parameters": {"type": "object", "properties": {
                                                   "order_id": {"type": "string", "description": "id"}},
                                                   "required": ["order_id"]}}}]
    base = [{"role": "system", "content": "policy text"}, {"role": "user", "content": "hi"}]
    a, _ = prompted_tools.prepare_request({"model": TEACHER, "messages": base, "tools": tools})
    b, _ = prompted_tools.prepare_request({"model": TEACHER, "tools": tools, "messages": base + [
        {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "order #W1"}]})
    assert a["messages"][0]["role"] == "system"
    assert a["messages"][0]["content"] == b["messages"][0]["content"]
    assert a["messages"][:2] == b["messages"][:2]

import json

import pytest

from shipwright.observability import retry


class HTTPError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status, "headers": headers or {}})()


class RateLimitError(Exception):  # litellm's class name, no status
    pass


class APIConnectionError(Exception):
    pass


def flaky(errors, result="ok"):
    errs = list(errors)
    calls = []

    def fn(*a, **k):
        calls.append((a, k))
        if errs:
            raise errs.pop(0)
        return result

    return fn, calls


@pytest.mark.parametrize("exc", [HTTPError(429), HTTPError(503), HTTPError(408), RateLimitError(),
                                 APIConnectionError(), ConnectionResetError(), TimeoutError(),
                                 json.JSONDecodeError("x", "", 0)])
def test_transient_errors_are_retried(exc):
    waits = []
    fn, calls = flaky([exc])
    assert retry.call_with_retry(fn, 1, k=2, sleep=waits.append) == "ok"
    assert len(calls) == 2 and calls[0] == ((1,), {"k": 2})
    assert len(waits) == 1


@pytest.mark.parametrize("exc", [HTTPError(401), HTTPError(403), HTTPError(404), HTTPError(400), ValueError("bug")])
def test_client_errors_raise_at_once(exc):
    waits = []
    fn, calls = flaky([exc])
    with pytest.raises(type(exc)):
        retry.call_with_retry(fn, sleep=waits.append)
    assert len(calls) == 1 and waits == []


def test_400_is_retried_only_when_the_policy_says_so():
    fn, calls = flaky([HTTPError(400)])
    assert retry.call_with_retry(fn, policy=retry.LLM_USER, sleep=lambda s: None) == "ok"
    assert len(calls) == 2


def test_max_attempts_and_max_total_wait():
    waits = []
    fn, calls = flaky([HTTPError(503)] * 10)
    with pytest.raises(HTTPError):
        retry.call_with_retry(fn, policy=retry.Policy(attempts=3, base=0.1), sleep=waits.append)
    assert len(calls) == 3 and len(waits) == 2

    waits.clear()
    fn, calls = flaky([HTTPError(503)] * 10)
    with pytest.raises(HTTPError):
        retry.call_with_retry(fn, policy=retry.Policy(attempts=10, base=4.0, max_total_wait=10.0), sleep=waits.append)
    assert sum(waits) <= 10.0 and len(calls) < 10


def test_backoff_grows_with_jitter_and_cap():
    p = retry.Policy(base=2.0, max_wait=20.0)
    assert retry.backoff(1, p, rng=lambda: 0.5) == 2.0
    assert retry.backoff(3, p, rng=lambda: 0.5) == 8.0
    assert retry.backoff(1, p, rng=lambda: 0.0) == 1.0
    assert retry.backoff(10, p, rng=lambda: 0.9) == 20.0


def test_retry_after_is_respected():
    waits = []
    fn, _ = flaky([HTTPError(429, {"retry-after": "7"})])
    retry.call_with_retry(fn, policy=retry.Policy(base=0.1), sleep=waits.append)
    assert waits == [7.0]


def test_non_idempotent_policy_retries_only_unsent_requests():
    fn, calls = flaky([HTTPError(503)])
    assert retry.call_with_retry(fn, policy=retry.LAUNCH, sleep=lambda s: None) == "ok"
    fn, calls = flaky([HTTPError(500)])
    with pytest.raises(HTTPError):
        retry.call_with_retry(fn, policy=retry.LAUNCH, sleep=lambda s: None)
    fn, calls = flaky([type("ReadTimeout", (Exception,), {})()])
    with pytest.raises(Exception):
        retry.call_with_retry(fn, policy=retry.LAUNCH, sleep=lambda s: None)


def test_decorator_and_retry_log_line(caplog):
    @retry.retry(retry.Policy(base=0.01), what="unit")
    def fn(state=[]):  # noqa: B006
        state.append(1)
        if len(state) < 2:
            raise HTTPError(502)
        return len(state)

    import logging

    with caplog.at_level(logging.WARNING):
        assert fn() == 2
    assert any("retry unit" in r.getMessage() for r in caplog.records)

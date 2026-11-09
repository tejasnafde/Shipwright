"""Shared test setup: every test starts with fresh logging and tracing state (shipwright.log)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _fresh_log_state():
    saved = {k: os.environ.get(k) for k in ("SHIPWRIGHT_RUN_ID", "TRACEPARENT")}
    yield
    from shipwright import log

    log.reset_for_tests()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

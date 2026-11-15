"""Trajectory efficiency: calls against tau2's gold path, repeats, failures and guard-blocked attempts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import trajectory_efficiency as te  # noqa: E402


def call(call_id, name, **arguments):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def result(call_id, error=False):
    return {"role": "tool", "id": call_id, "content": "x", "error": error}


GOLD = [te.canonical("get_order_details", {"order_id": "#W1"}),
        te.canonical("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})]


def test_expert_path_is_fully_efficient():
    messages = [call("a", "get_order_details", order_id="#W1"), result("a"),
                call("b", "cancel_pending_order", order_id="#W1", reason="no longer needed"), result("b")]
    row = te.measure(True, messages, GOLD)
    assert (row["precision"], row["recall"], row["redundant"], row["failed"]) == (1.0, 1.0, 0, 0)
    assert row["efficient"]


def test_repeat_failure_and_guard_block_are_waste():
    messages = [call("a", "get_order_details", order_id="#W9"), result("a", error=True),
                call("b", "get_order_details", order_id="#W1"), result("b"),
                call("c", "get_order_details", order_id="#W1"), result("c")]
    row = te.measure(False, messages, GOLD, blocked=2)
    assert (row["failed"], row["redundant"], row["blocked"], row["attempts"]) == (1, 1, 2, 5)
    assert row["on_path"] == 1 and row["recall"] == 0.5
    assert not row["efficient"]
    summary = te.summarise("x", [row])
    assert summary["wasted_share"] == 4 / 5


def test_a_successful_change_reopens_lookups():
    messages = [call("a", "get_order_details", order_id="#W1"), result("a"),
                call("b", "cancel_pending_order", order_id="#W1", reason="no longer needed"), result("b"),
                call("c", "get_order_details", order_id="#W1"), result("c")]
    assert te.measure(True, messages, GOLD)["redundant"] == 0

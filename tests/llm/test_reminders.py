"""Per-turn status reminders (src/shipwright/llm/reminders.py) and where they go in a request. Offline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shipwright.llm import prompted_tools, reminders

from test_guards import AUTHENTICATED, GREETING, agent_call, agent_text, customer, tool_result

ROOT = Path(__file__).resolve().parents[2]


def test_unidentified_customer_is_reminded_to_authenticate_first():
    note = reminders.status_note(GREETING)
    assert "not identified yet" in note and "email" in note
    failed = GREETING + [agent_call("find_user_id_by_email", {"email": "x@y.z"}, "c1"),
                         tool_result("c1", "Error: User not found")]
    assert "found no user" in reminders.status_note(failed)


def test_identified_customer_and_last_error_are_shown():
    assert "user id yusuf_rossi_9620" in reminders.status_note(AUTHENTICATED)
    failed_lookup = AUTHENTICATED + [agent_call("get_order_details", {"order_id": "#W1"}, "c2"),
                                     tool_result("c2", "Error: Order not found")]
    note = reminders.status_note(failed_lookup)
    assert "failed" in note and "Order not found" in note


def test_a_yes_to_a_proposed_change_says_act_now_once():
    proposed = AUTHENTICATED + [agent_text("I will cancel #W2378156 (no longer needed). Shall I proceed?"),
                                customer("Yes, go ahead.")]
    assert "Make that tool call now" in reminders.status_note(proposed)
    done = proposed + [agent_call("cancel_pending_order", {"order_id": "#W2378156"}, "c3"),
                       tool_result("c3", '{"status": "cancelled"}')]
    assert "Make that tool call now" not in reminders.status_note(done)
    # "Sure" answering a request for an email is not a confirmation
    asked_email = GREETING + [agent_text("Could you give me your email?"), customer("Sure, it is a@b.c")]
    assert "Make that tool call now" not in reminders.status_note(asked_email)


def test_note_goes_on_the_last_message_sent_to_the_model_only():
    request = {"model": "openai/CohereLabs/tiny-aya-global",
               "messages": [{"role": "system", "content": "Policy."}, *GREETING]}
    try:
        reminders.set_enabled(True)
        prepared, info = prompted_tools.prepare_request(request)
    finally:
        reminders.set_enabled(False)
    assert info["reminder"] and prepared["messages"][-1]["content"].endswith(info["reminder"])
    assert "[Status" not in json.dumps(request["messages"])  # tau2's conversation is untouched
    prepared_off, info_off = prompted_tools.prepare_request(request)
    assert info_off["reminder"] is None and "[Status" not in json.dumps(prepared_off["messages"])


def test_dry_run_with_reminders(tmp_path):
    pytest.importorskip("tau2")
    output_directory = tmp_path / "reminded"
    result = subprocess.run([sys.executable, "scripts/baseline.py", "--dry-run", "--num-tasks", "1",
                             "--state-reminders", "--output-dir", str(output_directory)],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads((output_directory / "summary.json").read_text())["state_reminders"] is True

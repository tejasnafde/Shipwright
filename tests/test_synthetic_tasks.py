"""Synthetic training tasks: verified by replay, nothing from the held-out or hidden tasks, loadable as a tau2 split."""

from __future__ import annotations

import json

import pytest

from shipwright.heal import synthetic_tasks as st


@pytest.fixture(scope="module")
def generated():
    return st.generate(40, seed=7)


def test_no_generated_task_mentions_a_held_out_or_hidden_customer_order_or_item(generated):
    tasks, _ = generated
    forbidden = st.protected_forbidden(st.load_db())
    assert forbidden.user_ids and forbidden.order_ids
    assert all(st.leaks(task, forbidden) == [] for task in tasks)
    text = json.dumps(tasks)
    assert not any(order_id in text for order_id in forbidden.order_ids)
    assert not any(user_id in text for user_id in forbidden.user_ids)


def test_every_generated_task_replays_cleanly_and_ids_are_unique(generated):
    tasks, report = generated
    assert len(tasks) == report["kept"] == 40
    assert len({task["id"] for task in tasks}) == 40
    assert all(st.verify(task) is None for task in tasks)


def test_the_leak_guard_finds_an_order_named_only_in_the_customer_text():
    db = st.load_db()
    order_id = next(iter(db["orders"]))
    task = {"user_scenario": {"instructions": {"reason_for_call": f"About order {order_id}."}},
            "evaluation_criteria": {"actions": []}}
    forbidden = st.forbidden_from([task], db)
    assert order_id in forbidden.order_ids
    assert db["orders"][order_id]["user_id"] in forbidden.user_ids
    assert st.leaks(task, forbidden) == [f"order {order_id}"]


def test_a_task_whose_write_fails_is_rejected():
    db = st.load_db()
    order = next(o for o in db["orders"].values() if o["status"] == "delivered")
    bad = {"description": {"purpose": "synthetic cancel task with 1 write(s)"},
           "user_scenario": {"instructions": {"reason_for_call": order["order_id"]}},
           "evaluation_criteria": {"actions": [st.action("cancel_pending_order", order_id=order["order_id"],
                                                         reason="no longer needed")]}}
    assert "failed" in st.verify(bad)


def test_a_refusal_task_needs_a_write_the_tools_reject():
    db = st.load_db()
    pending = next(o for o in db["orders"].values() if o["status"] == "pending")
    task = {"description": {"purpose": "synthetic refuse_cancel task with 0 write(s)"},
            "user_scenario": {"instructions": {"reason_for_call": f"Cancel {pending['order_id']}."}},
            "evaluation_criteria": {"actions": [st.action("get_order_details", order_id=pending["order_id"])]}}
    assert st.verify(task) == "the refused write would have succeeded"  # a pending order can be cancelled


def test_the_synthetic_split_loads_through_tau2(generated, tmp_path):
    tasks, _ = generated
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks))
    from tau2.registry import registry

    saved_tasks, saved_splits = registry._tasks["retail"], registry._task_splits.get("retail")
    try:
        st.register_split(path)
        loaded = registry.get_tasks_loader("retail")(st.SPLIT_NAME)
        assert [task.id for task in loaded] == [task["id"] for task in tasks]
        assert len(registry.get_tasks_loader("retail")("train")) == 74  # the official splits are unchanged
    finally:
        registry._tasks["retail"], registry._task_splits["retail"] = saved_tasks, saved_splits

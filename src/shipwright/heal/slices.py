"""Slices of the held-out tasks, so the gate can show where the weights helped.

A slice is defined from the task itself (how many write actions its answer needs), never from an outcome,
so it can be fixed before any gate result exists. The headline stays the overall pass^1 and pass^k.
Teacher data (2026-11-12): tasks needing 0 to 1 writes pass 40 to 59% for the teacher, tasks needing 2 or
more 22 to 25%.
"""

from __future__ import annotations

WRITE_ACTIONS = frozenset({
    "cancel_pending_order", "return_delivered_order_items", "exchange_delivered_order_items",
    "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
    "modify_user_address", "transfer_to_human_agents",
})
FEW_WRITES = "0-1 writes"
MANY_WRITES = "2+ writes"
SLICE_NAMES = (FEW_WRITES, MANY_WRITES)


def slice_for_write_count(write_count: int) -> str:
    if write_count <= 1:
        return FEW_WRITES
    return MANY_WRITES


def write_slice_by_task(domain: str, task_split: str) -> dict[str, str]:
    """Task id -> slice name, from the expected actions of every task of the tau2 split."""
    from tau2.registry import registry

    slice_by_task: dict[str, str] = {}
    for task in registry.get_tasks_loader(domain)(task_split_name=task_split):
        criteria = task.evaluation_criteria
        expected_actions = (criteria.actions if criteria else None) or []
        write_count = sum(1 for action in expected_actions if action.name in WRITE_ACTIONS)
        slice_by_task[str(task.id)] = slice_for_write_count(write_count)
    return slice_by_task


def slice_pass_metrics(per_task: dict, slice_by_task: dict[str, str]) -> dict[str, dict]:
    """pass^1 and pass^k per slice, from ``compute_metrics``' ``per_task`` (tasks outside the map are left out)."""
    pass_1_by_slice: dict[str, list[float]] = {name: [] for name in SLICE_NAMES}
    pass_k_by_slice: dict[str, list[float]] = {name: [] for name in SLICE_NAMES}
    for task_id, task_result in per_task.items():
        slice_name = slice_by_task.get(str(task_id))
        if slice_name is None:
            continue
        if task_result["n_decided"]:
            pass_1_by_slice[slice_name].append(task_result["n_success"] / task_result["n_decided"])
        if task_result["pass_k"] is not None:
            pass_k_by_slice[slice_name].append(task_result["pass_k"])

    def average(values: list[float]) -> float | None:
        if values:
            return sum(values) / len(values)
        return None

    return {
        name: {
            "num_tasks": len(pass_1_by_slice[name]),
            "pass_1": average(pass_1_by_slice[name]),
            "pass_k": average(pass_k_by_slice[name]),
        }
        for name in SLICE_NAMES
    }

"""Staged example data for control-room's "Follow a Task" and gate-rejection views.

No network, no GPU, no paid call, no model. This writes run dirs shaped exactly like the
real ones (scripts/baseline.py, scripts/teacher_runs.py, scripts/gate_eval.py produce), but
every value in them is hand-written, not measured - because round 2 has not actually been
trained yet. Every dir gets a demo.json marker; control-room's render_demo_banner() reads it
and shows an orange "STAGED EXAMPLE, not a measured result" banner wherever that dir's data
is displayed, so a staged illustration is never mistaken for a real result.

Two scenarios, matching mentor feedback on the demo:

1. The "aha moment": task 42 hallucinates an order id and fails in round 1, the teacher-replay
   fix for that same task, and task 42 passing in a staged "round 2".
   Writes: runs/baseline-demo-round1/, runs/sft-demo-fix/, runs/baseline-demo-round2/

2. A genuine gate rejection, not just insufficient sample size: the gate's own decide() runs
   for real against hand-built metrics where the candidate has plenty of valid runs but a
   real pass^1 regression, so the rejection reason is the actual regression check, not
   "too few valid runs".
   Writes: runs/gate-demo-regression/

Run: .venv/bin/python scripts/demo_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.heal import gate
from shipwright.heal.traces import write_jsonl

RUNS_DIRECTORY = ROOT / "runs"
TASK_ID = "42"
OTHER_TASK_ID = "7"
HALLUCINATED_ID = "#W4471182"
REAL_ID = "#W0000042"


def _demo_json(note: str) -> dict:
    return {"staged": True, "note": note}


def _user(content: str) -> dict:
    return {"role": "user", "content": content, "tool_calls": None}


def _tool_call_message(call_id: str, name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "name": name, "arguments": arguments, "requestor": "assistant"}]}


def _tool_result(call_id: str, content: str, error: bool) -> dict:
    return {"id": call_id, "role": "tool", "content": content, "error": error}


def _assistant_text(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None}


def _refund_conversation(order_id: str, found: bool) -> list[dict]:
    """The shared shape of this scenario's conversation: greeting, the customer naming
    their order, one lookup, then the agent's final answer - correct when ``found`` is
    True (the real id resolves), wrong when it's False (an invented id does not)."""
    messages = [
        _assistant_text("Hi! How can I help you today?"),
        _user(f"Hi, I'd like a refund for order {REAL_ID} please."),
        _tool_call_message("call_1", "get_order_details", {"order_id": order_id}),
    ]
    if found:
        messages.append(_tool_result("call_1", f"order {order_id}: 1 item, delivered", error=False))
        messages.append(_assistant_text("Found it - your return has been processed and the refund is on its way."))
    else:
        messages.append(_tool_result("call_1", f"Error: no order found with id {order_id}", error=True))
        messages.append(_assistant_text("Your return has been processed and a refund has been issued."))
    return messages


def _other_task_trial(passed: bool = True) -> dict:
    """A second, unremarkable task in the same run dir, so the round doesn't look cherry-picked
    down to a single trial."""
    return {
        "simulation_id": f"demo-{OTHER_TASK_ID}-0",
        "task_id": OTHER_TASK_ID,
        "trial": 0,
        "seed": 300,
        "reward": 1.0 if passed else 0.0,
        "passed": passed,
        "termination_reason": "user_stop",
        "error": None,
        "error_type": None,
        "messages": [
            _assistant_text("Hi! How can I help you today?"),
            _user("Can you check the status of order #W0000007?"),
            _tool_call_message("call_1", "get_order_details", {"order_id": "#W0000007"}),
            _tool_result("call_1", "order #W0000007: delivered", error=False),
            _assistant_text("Your order #W0000007 was delivered. Anything else?"),
        ],
    }


def round1_baseline() -> Path:
    output_directory = RUNS_DIRECTORY / "baseline-demo-round1"
    output_directory.mkdir(parents=True, exist_ok=True)

    failing_trial = {
        "simulation_id": f"demo-{TASK_ID}-0",
        "task_id": TASK_ID,
        "trial": 0,
        "seed": 300,
        "reward": 0.0,
        "passed": False,
        "termination_reason": "user_stop",
        "error": None,
        "error_type": None,
        "messages": _refund_conversation(HALLUCINATED_ID, found=False),
    }
    write_jsonl(output_directory / "traces.jsonl", [failing_trial, _other_task_trial()])

    failure = {
        "task_id": TASK_ID,
        "trial": 0,
        "simulation_id": f"demo-{TASK_ID}-0",
        "reward": 0.0,
        "labels": ["invented_id"],
        "rule_hits": [{"label": "invented_id", "turn": 2,
                        "detail": f"order_id {HALLUCINATED_ID} does not match any id in context"}],
        "first_wrong_turn": 2,
        "fault_layer": "model",
        "route": "weight_fix",
        "source": "rules",
        "explanation": (
            f"invented_id at turn 2: the customer's real order id ({REAL_ID}) is right there in "
            f"their message, but the model called get_order_details with {HALLUCINATED_ID} - an id "
            f"it made up - then told the customer the refund was done after the lookup failed, "
            f"instead of rereading the message or asking for the order number."
        ),
        "outcome_failed": True,
        "detected_by": ["outcome", "agenttune_detector"],
        "agenttune_root_cause": "hallucinated_output",
    }
    write_jsonl(output_directory / "failures.jsonl", [failure])

    (output_directory / "round_status.json").write_text(json.dumps({
        "should_train": True,
        "reason": "dominant_failure_type_ready",
        "num_weight_fix_failures": 1,
        "num_with_correction": 1,
        "run_dir": str(output_directory),
    }, indent=2))

    (output_directory / "demo.json").write_text(json.dumps(_demo_json(
        f"Round 1 baseline: task {TASK_ID} fails because the model invents an order id "
        f"({HALLUCINATED_ID}) instead of using the one the customer gave it ({REAL_ID}). "
        "Hand-written to illustrate the 'aha moment' story end to end (see runs/sft-demo-fix, "
        "runs/baseline-demo-round2) - not a real tau2 run."
    ), indent=2))
    return output_directory


def fix_sft() -> Path:
    """The teacher's own run on task 42, kept because it passed - same shape as a row in
    scripts/teacher_runs.py's kept.jsonl (a full trace dict, reward 1.0), not a hand-rolled
    {student, teacher} pair: control-room pairs this with the student's round-1 trace itself
    (same task id, from the baseline dir) rather than needing a separate field for it."""
    output_directory = RUNS_DIRECTORY / "sft-demo-fix"
    output_directory.mkdir(parents=True, exist_ok=True)

    kept_trace = {
        "simulation_id": f"demo-{TASK_ID}-teacher",
        "task_id": TASK_ID,
        "trial": 0,
        "seed": 300,
        "reward": 1.0,
        "passed": True,
        "termination_reason": "user_stop",
        "error": None,
        "error_type": None,
        "source": "teacher",
        "messages": _refund_conversation(REAL_ID, found=True),
    }
    write_jsonl(output_directory / "kept.jsonl", [kept_trace])

    (output_directory / "teacher_summary.json").write_text(json.dumps({
        "kept": 1, "per_task": {TASK_ID: 1}, "tasks_without_success": [],
    }, indent=2))

    (output_directory / "demo.json").write_text(json.dumps(_demo_json(
        f"Teacher replay for task {TASK_ID}, hand-written to show the contrast: the real "
        "order id was sitting in the customer's own message, the student ignored it and invented "
        "one, the teacher replay uses it directly. Shaped exactly like a real teacher_runs.py "
        "kept.jsonl row (a full passing trace, not a synthetic pair)."
    ), indent=2))
    return output_directory


def round2_baseline() -> Path:
    output_directory = RUNS_DIRECTORY / "baseline-demo-round2"
    output_directory.mkdir(parents=True, exist_ok=True)

    passing_trial = {
        "simulation_id": f"demo-{TASK_ID}-0-r2",
        "task_id": TASK_ID,
        "trial": 0,
        "seed": 300,
        "reward": 1.0,
        "passed": True,
        "termination_reason": "user_stop",
        "error": None,
        "error_type": None,
        "messages": _refund_conversation(REAL_ID, found=True),
    }
    write_jsonl(output_directory / "traces.jsonl", [passing_trial, _other_task_trial()])

    (output_directory / "demo.json").write_text(json.dumps(_demo_json(
        f"Staged 'after' round: task {TASK_ID} passing once the LoRA adapter has learned to use "
        "the order id already in the customer's message instead of inventing one. No round-2 "
        "adapter has actually been trained and gated yet - this illustrates what success on this "
        "task should look like once one is, for the demo's before/after story."
    ), indent=2))
    return output_directory


def gate_regression() -> Path:
    """A real gate.decide() call on hand-built metrics: plenty of valid runs on both sides
    (so 'too few valid runs' can't be the reason), candidate pass^1 genuinely lower than
    current by more than the configured drop tolerance. Shows the gate rejecting a real
    regression, not the easier/less convincing insufficient-sample-size case."""
    output_directory = RUNS_DIRECTORY / "gate-demo-regression"
    output_directory.mkdir(parents=True, exist_ok=True)

    task_ids = [str(i) for i in range(20)]
    gate_config = {
        "status": "frozen",
        "heldout": {"domain": "retail", "task_split": "test", "num_tasks": 20},
        "k": 3,
        "seed": 300,
        "thresholds": {
            "min_valid_runs": 54, "min_valid_run_fraction": 0.9, "min_complete_tasks": 18,
            "pass_1_max_drop": 0.0, "pass_k_max_drop": 0.0, "target_label_min_drop": 0.025,
            "non_target_max_increase": 0.05, "bfcl_max_drop_points": 2.0,
        },
        "bfcl": {"enabled": True},
    }

    def per_task(pass_1: float, valid_runs: int) -> dict:
        """Spread valid_runs (up to k=3 per task, in task order) and pass_1 * valid_runs successes over the tasks."""
        decided_per_task = [min(3, max(0, valid_runs - 3 * task_number)) for task_number in range(len(task_ids))]
        successes_left = round(pass_1 * valid_runs)
        rows = {}
        for task_id, decided_count in zip(task_ids, decided_per_task):
            success_count = min(decided_count, successes_left)
            successes_left -= success_count
            rows[task_id] = {"n_success": success_count, "n_decided": decided_count, "pass_k": None}
        return rows

    def failed_run_labels(target_count: int, other_count: int) -> list[dict]:
        rows = [{"simulation_id": f"t{number}", "labels": ["invented_id"]} for number in range(target_count)]
        rows += [{"simulation_id": f"o{number}", "labels": ["loop"]} for number in range(other_count)]
        return rows

    current = {
        "per_task": per_task(0.55, 60), "system": "current", "task_split": "test", "k": 3,
        "task_ids": task_ids, "num_runs": 60, "num_valid_runs": 60, "num_tasks_complete": 20,
        "pass_1": 0.55, "pass_k": 0.30, "failed_run_labels": failed_run_labels(12, 8),
        "bfcl": {"accuracy_points": 61.0},
    }
    candidate = {
        "per_task": per_task(0.40, 60), "system": "candidate", "task_split": "test", "k": 3,
        "task_ids": task_ids, "num_runs": 60, "num_valid_runs": 60, "num_tasks_complete": 20,
        "pass_1": 0.40, "pass_k": 0.18, "failed_run_labels": failed_run_labels(6, 20),
        "bfcl": {"accuracy_points": 59.5},
    }

    decision = gate.decide(current, candidate, gate_config, ["invented_id"])
    decision["round"] = 2
    decision["systems"] = {"current": "round1-sft", "candidate": "round2-sft-overfit-demo"}
    (output_directory / "metrics_current.json").write_text(json.dumps(current, indent=1, default=str))
    (output_directory / "metrics_candidate.json").write_text(json.dumps(candidate, indent=1, default=str))
    (output_directory / "gate_decision.json").write_text(json.dumps(decision, indent=1, default=str))
    metrics_json_row = gate.metrics_row(2, decision, decision["systems"])
    write_jsonl(output_directory / "metrics.jsonl", [metrics_json_row])

    (output_directory / "demo.json").write_text(json.dumps(_demo_json(
        "Staged gate run: hand-built current/candidate metrics fed into the real gate.decide() "
        "(src/shipwright/heal/gate.py) - the decision itself is computed, not written by hand. "
        f"Both sides have {current['num_valid_runs']} valid runs, well above the "
        f"{gate_config['thresholds']['min_valid_runs']} floor, so 'too few valid runs' is not why this "
        f"rejects: pass^1 drops from {current['pass_1']:.0%} to {candidate['pass_1']:.0%} - a real "
        "regression, which is the case worth showing, not the easier insufficient-sample-size one. "
        "No such overfit round has actually happened."
    ), indent=2))
    assert decision["decision"] == "reject", "demo gate scenario must actually reject - check thresholds"
    return output_directory


def main() -> int:
    paths = [round1_baseline(), fix_sft(), round2_baseline(), gate_regression()]
    for written_path in paths:
        print(f"wrote {written_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

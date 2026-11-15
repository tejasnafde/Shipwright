"""Deploy gate (docs/evaluation.md, docs/pipeline/gate.md): promote or reject a candidate.

Pure logic: two metric dicts (``shipwright.heal.eval_metrics.compute_metrics``) and the gate
config (``configs/gate.yaml``) in, a decision dict out. No I/O except ``load_config``.

The candidate is promoted only when every check passes:

1. comparable   both systems ran the configured held-out split, the same task ids, the same k
2. valid_runs   each system has enough valid (non infra-error) runs, and enough tasks with
                all k trials valid; otherwise reject (AgentTune's DeploymentGate approves in
                that case: we never do)
3. pass_1       candidate pass^1 >= current pass^1 - pass_1_max_drop (a tie passes)
4. pass_k       candidate pass^k >= current pass^k - pass_k_max_drop (a tie passes)
5. target       the target-label failure rate drops by at least target_label_min_drop and
                strictly (a tie rejects: no measured improvement)
6. non_target   other failure labels rise by at most non_target_max_increase (null: off)
7. bfcl         when BFCL is enabled, candidate >= current - bfcl_max_drop_points
                (missing BFCL numbers reject)
8. agenttune_gate  AgentTune ``DeploymentGate.evaluate_decision`` on the paired held-out runs
                approves (no task regression beyond agenttune_gate.task_regression_tol), and
                there are at least min_test_samples paired runs (the valid_runs floor).
                AgentTune alone approves an empty test set; this check does not.

Promote = AgentTune's DeploymentGate approves AND every AuditKIT threshold passes.

A missing metric (None) fails its check. The decision is ``reject`` unless all pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EPS = 1e-9
THRESHOLD_KEYS = (
    "min_valid_runs", "min_valid_run_fraction", "min_complete_tasks",
    "pass_1_max_drop", "pass_k_max_drop", "target_label_min_drop",
    "non_target_max_increase", "bfcl_max_drop_points",
)


# ---------------------------------------------------------------- config

def validate_config(gate_config: dict) -> dict:
    """Check the gate config. Raises ValueError with the first problem."""
    if not isinstance(gate_config, dict):
        raise ValueError("gate config must be a mapping")
    if gate_config.get("status") not in ("proposed", "frozen"):
        raise ValueError("gate config status must be 'proposed' or 'frozen'")

    held_out = gate_config.get("heldout") or {}
    for key in ("domain", "task_split", "num_tasks"):
        if key not in held_out:
            raise ValueError(f"heldout.{key} missing")
    if held_out["task_split"] == "train":
        raise ValueError("the gate must not score on the train split")

    k = gate_config.get("k")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")

    thresholds = gate_config.get("thresholds") or {}
    missing_keys = [key for key in THRESHOLD_KEYS if key not in thresholds]
    if missing_keys:
        raise ValueError(f"thresholds missing: {missing_keys}")
    for key in THRESHOLD_KEYS:
        value = thresholds[key]
        if value is None and key == "non_target_max_increase":
            continue
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"thresholds.{key} must be a number >= 0")
    if thresholds["min_valid_runs"] < 1:
        raise ValueError("thresholds.min_valid_runs must be at least 1 (never approve on no data)")
    if not 0 < thresholds["min_valid_run_fraction"] <= 1:
        raise ValueError("thresholds.min_valid_run_fraction must be in (0, 1]")

    task_tolerance = agenttune_task_tol(gate_config)
    if not isinstance(task_tolerance, (int, float)) or task_tolerance < 0:
        raise ValueError("agenttune_gate.task_regression_tol must be a number >= 0")
    return gate_config


def agenttune_task_tol(gate_config: dict) -> float:
    """Task regression tolerance for AgentTune's DeploymentGate (default 0: no drop)."""
    return (gate_config.get("agenttune_gate") or {}).get("task_regression_tol", 0.0)


def load_config(path: Path) -> dict:
    import yaml

    return validate_config(yaml.safe_load(Path(path).read_text()))


def config_sha(gate_config: dict) -> str:
    return hashlib.sha256(json.dumps(gate_config, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- label rates

def _labelled_runs(metrics: dict, targets: set[str], exclude_targets: bool) -> int:
    """Failed runs with a target label (or, with exclude_targets, with a label outside the targets)."""
    run_count = 0
    for failed_run in metrics.get("failed_run_labels") or []:
        labels = set(failed_run.get("labels") or []) - {"environment"}
        if exclude_targets:
            has_matching_label = bool(labels - targets)
        else:
            has_matching_label = bool(labels & targets)
        run_count += int(has_matching_label)
    return run_count


def target_rate(metrics: dict, targets: list[str]) -> float | None:
    """Share of valid runs that fail with any target label."""
    valid_run_count = metrics.get("num_valid_runs") or 0
    if not valid_run_count:
        return None
    return _labelled_runs(metrics, set(targets), exclude_targets=False) / valid_run_count


def non_target_rate(metrics: dict, targets: list[str]) -> float | None:
    """Share of valid runs that fail with any label outside the targets (environment excluded)."""
    valid_run_count = metrics.get("num_valid_runs") or 0
    if not valid_run_count:
        return None
    return _labelled_runs(metrics, set(targets), exclude_targets=True) / valid_run_count


# ---------------------------------------------------------------- checks

def _check(name: str, passed: bool, detail: str, **values: Any) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail, **values}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def required_valid_runs(gate_config: dict) -> int:
    """Valid runs each system needs: the larger of the absolute floor and the fraction."""
    thresholds = gate_config["thresholds"]
    planned_runs = gate_config["heldout"]["num_tasks"] * gate_config["k"]
    fraction_floor = math.ceil(thresholds["min_valid_run_fraction"] * planned_runs)
    return max(int(thresholds["min_valid_runs"]), fraction_floor)


def check_comparable(current: dict, candidate: dict, gate_config: dict) -> dict:
    expected_split = gate_config["heldout"]["task_split"]
    problems = []
    for role, metrics in (("current", current), ("candidate", candidate)):
        if metrics.get("task_split") != expected_split:
            problems.append(f"{role} ran split {metrics.get('task_split')!r}, gate needs {expected_split!r}")
        if metrics.get("k") != gate_config["k"]:
            problems.append(f"{role} k={metrics.get('k')}, gate k={gate_config['k']}")
    if sorted(current.get("task_ids") or []) != sorted(candidate.get("task_ids") or []):
        problems.append("current and candidate ran different task ids")

    if problems:
        detail = "; ".join(problems)
    else:
        detail = f"same {expected_split} tasks, k={gate_config['k']}"
    return _check("comparable", not problems, detail)


def check_valid_runs(current: dict, candidate: dict, gate_config: dict) -> dict:
    required_runs = required_valid_runs(gate_config)
    required_tasks = int(gate_config["thresholds"]["min_complete_tasks"])
    problems = []
    for role, metrics in (("current", current), ("candidate", candidate)):
        valid_runs = metrics.get("num_valid_runs")
        complete_tasks = metrics.get("num_tasks_complete")
        if not _is_number(valid_runs) or valid_runs < required_runs:
            problems.append(f"{role} has {valid_runs} valid runs, needs {required_runs}")
        if not _is_number(complete_tasks) or complete_tasks < required_tasks:
            problems.append(
                f"{role} has {complete_tasks} tasks with {gate_config['k']} valid trials, needs {required_tasks}"
            )

    if problems:
        detail = "too few valid runs: " + "; ".join(problems)
    else:
        detail = f"both have >= {required_runs} valid runs and >= {required_tasks} complete tasks"
    return _check(
        "valid_runs", not problems, detail,
        required_runs=required_runs, required_complete_tasks=required_tasks,
    )


def check_not_lower(name: str, current_value: Any, candidate_value: Any, max_drop: float) -> dict:
    if not (_is_number(current_value) and _is_number(candidate_value)):
        return _check(
            name, False, f"{name} missing (current={current_value}, candidate={candidate_value})",
            current=current_value, candidate=candidate_value,
        )
    passed = candidate_value >= current_value - max_drop - EPS
    verb = "keeps" if passed else "lowers"
    return _check(
        name, passed,
        f"candidate {verb} {name}: {candidate_value:.4f} vs {current_value:.4f} (max drop {max_drop})",
        current=current_value, candidate=candidate_value, delta=candidate_value - current_value,
    )


def check_target(current: dict, candidate: dict, targets: list[str], min_drop: float) -> dict:
    if not targets:
        return _check("target", False, "no target failure label: the candidate must name what it fixes")
    current_rate = target_rate(current, targets)
    candidate_rate = target_rate(candidate, targets)
    if not (_is_number(current_rate) and _is_number(candidate_rate)):
        return _check(
            "target", False,
            f"target rate missing (current={current_rate}, candidate={candidate_rate})",
            targets=targets,
        )
    drop = current_rate - candidate_rate
    passed = drop > EPS and drop >= min_drop - EPS
    detail = (
        f"target {'+'.join(targets)} rate {candidate_rate:.4f} vs {current_rate:.4f}: drop {drop:.4f}"
        f" (needs > 0 and >= {min_drop})"
    )
    return _check(
        "target", passed, detail,
        targets=targets, current=current_rate, candidate=candidate_rate, delta=-drop,
    )


def check_non_target(current: dict, candidate: dict, targets: list[str], max_increase: float | None) -> dict:
    if max_increase is None:
        return _check("non_target", True, "non-target check off", skipped=True)
    current_rate = non_target_rate(current, targets)
    candidate_rate = non_target_rate(candidate, targets)
    if not (_is_number(current_rate) and _is_number(candidate_rate)):
        return _check(
            "non_target", False,
            f"non-target rate missing (current={current_rate}, candidate={candidate_rate})",
        )
    passed = candidate_rate - current_rate <= max_increase + EPS
    return _check(
        "non_target", passed,
        f"other labels rate {candidate_rate:.4f} vs {current_rate:.4f} (max increase {max_increase})",
        current=current_rate, candidate=candidate_rate, delta=candidate_rate - current_rate,
    )


def check_bfcl(current: dict, candidate: dict, gate_config: dict) -> dict:
    bfcl_config = gate_config.get("bfcl") or {}
    if not bfcl_config.get("enabled"):
        return _check("bfcl", True, "BFCL off in the gate config", skipped=True)
    max_drop_points = float(gate_config["thresholds"]["bfcl_max_drop_points"])
    current_points = (current.get("bfcl") or {}).get("accuracy_points")
    candidate_points = (candidate.get("bfcl") or {}).get("accuracy_points")
    if not (_is_number(current_points) and _is_number(candidate_points)):
        return _check(
            "bfcl", False,
            f"BFCL enabled but missing (current={current_points}, candidate={candidate_points})",
        )
    passed = candidate_points >= current_points - max_drop_points - EPS
    return _check(
        "bfcl", passed,
        f"BFCL {candidate_points:.2f} vs {current_points:.2f} points (max drop {max_drop_points})",
        current=current_points, candidate=candidate_points, delta=candidate_points - current_points,
    )


def check_agenttune(current: dict, candidate: dict, gate_config: dict) -> dict:
    from shipwright.heal.agenttune_bridge import agenttune_gate_check

    return agenttune_gate_check(
        current, candidate,
        min_test_samples=required_valid_runs(gate_config),
        task_regression_tol=float(agenttune_task_tol(gate_config)),
    )


# ---------------------------------------------------------------- decision

def public_metrics(metrics: dict) -> dict:
    """Metrics without the bulky per-run label list (kept in metrics_*.json files)."""
    return {key: value for key, value in metrics.items() if key not in ("failed_run_labels", "per_task")}


def decide(
    current: dict,
    candidate: dict,
    gate_config: dict,
    targets: list[str] | None = None,
    now: str | None = None,
) -> dict:
    """The gate decision for ``candidate`` against ``current``."""
    validate_config(gate_config)
    thresholds = gate_config["thresholds"]
    targets = list(targets or [])
    checks = [
        check_comparable(current, candidate, gate_config),
        check_valid_runs(current, candidate, gate_config),
        check_not_lower("pass_1", current.get("pass_1"), candidate.get("pass_1"), thresholds["pass_1_max_drop"]),
        check_not_lower("pass_k", current.get("pass_k"), candidate.get("pass_k"), thresholds["pass_k_max_drop"]),
        check_target(current, candidate, targets, thresholds["target_label_min_drop"]),
        check_non_target(current, candidate, targets, thresholds["non_target_max_increase"]),
        check_bfcl(current, candidate, gate_config),
        check_agenttune(current, candidate, gate_config),
    ]
    failed_checks = [check for check in checks if not check["passed"]]

    warnings = []
    if gate_config.get("status") != "frozen":
        warnings.append("gate config status is 'proposed': the team has not frozen it yet")

    agenttune_check = checks[-1]
    failed_threshold_checks = [check for check in checks[:-1] if not check["passed"]]
    if failed_checks:
        decision = "reject"
        reasons = [check["detail"] for check in failed_checks]
    else:
        decision = "promote"
        reasons = [check["detail"] for check in checks if not check.get("skipped")]
    return {
        "decision": decision,
        "verdicts": {
            "agenttune": {
                "approved": agenttune_check["agenttune_approved"],
                "reason": agenttune_check["agenttune_reason"],
                "check_passed": agenttune_check["passed"],
                "num_cases": agenttune_check["num_cases"],
                "min_test_samples": agenttune_check["min_test_samples"],
            },
            "auditkit_thresholds": {
                "passed": not failed_threshold_checks,
                "failed_checks": [check["name"] for check in failed_threshold_checks],
            },
        },
        "reasons": reasons,
        "checks": checks,
        "warnings": warnings,
        "targets": targets,
        "metrics_current": public_metrics(current),
        "metrics_candidate": public_metrics(candidate),
        "thresholds": copy.deepcopy(thresholds),
        "k": gate_config["k"],
        "heldout": copy.deepcopy(gate_config["heldout"]),
        "config_status": gate_config.get("status"),
        "config_sha": config_sha(gate_config),
        "decided_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


METRICS_ROW_KEYS = (
    "system", "num_runs", "num_valid_runs", "num_tasks_complete", "pass_1", "pass_1_ci", "pass_k",
    "failure_rate", "label_rates", "tool_call_error_rate", "avg_steps", "auditkit", "bfcl",
    "active_tool_patches", "agent_model",
)


def metrics_row(round_number: int, decision: dict, systems: dict) -> dict:
    """One metrics.jsonl row: round, systems compared, headline metrics, decision."""

    def headline_metrics(metrics: dict) -> dict:
        return {key: metrics.get(key) for key in METRICS_ROW_KEYS}

    return {
        "kind": "gate",
        "round": round_number,
        "at": decision["decided_at"],
        "systems": systems,
        "targets": decision["targets"],
        "metrics": {
            "current": headline_metrics(decision["metrics_current"]),
            "candidate": headline_metrics(decision["metrics_candidate"]),
        },
        "decision": decision["decision"],
        "reasons": decision["reasons"],
        "verdicts": decision.get("verdicts"),
        "config_status": decision["config_status"],
        "config_sha": decision["config_sha"],
    }

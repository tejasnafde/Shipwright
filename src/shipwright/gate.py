"""Deploy gate (step 05 GATE in docs/system-design.html): promote or reject a candidate.

Pure logic: two metric dicts (``shipwright.eval_metrics.compute_metrics``) and the gate
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

def validate_config(cfg: dict) -> dict:
    """Check the gate config. Raises ValueError with the first problem."""
    if not isinstance(cfg, dict):
        raise ValueError("gate config must be a mapping")
    if cfg.get("status") not in ("proposed", "frozen"):
        raise ValueError("gate config status must be 'proposed' or 'frozen'")
    held = cfg.get("heldout") or {}
    for key in ("domain", "task_split", "num_tasks"):
        if key not in held:
            raise ValueError(f"heldout.{key} missing")
    if held["task_split"] == "train":
        raise ValueError("the gate must not score on the train split")
    k = cfg.get("k")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")
    th = cfg.get("thresholds") or {}
    missing = [key for key in THRESHOLD_KEYS if key not in th]
    if missing:
        raise ValueError(f"thresholds missing: {missing}")
    for key in THRESHOLD_KEYS:
        val = th[key]
        if val is None and key == "non_target_max_increase":
            continue
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(f"thresholds.{key} must be a number >= 0")
    if th["min_valid_runs"] < 1:
        raise ValueError("thresholds.min_valid_runs must be at least 1 (never approve on no data)")
    if not 0 < th["min_valid_run_fraction"] <= 1:
        raise ValueError("thresholds.min_valid_run_fraction must be in (0, 1]")
    return cfg


def load_config(path: Path) -> dict:
    import yaml

    return validate_config(yaml.safe_load(Path(path).read_text()))


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- label rates

def _labelled_runs(metrics: dict, targets: set[str], exclude: bool) -> int:
    n = 0
    for row in metrics.get("failed_run_labels") or []:
        have = set(row.get("labels") or []) - {"environment"}
        n += int(bool(have - targets if exclude else have & targets))
    return n


def target_rate(metrics: dict, targets: list[str]) -> float | None:
    """Share of valid runs that fail with any target label."""
    n = metrics.get("num_valid_runs") or 0
    return _labelled_runs(metrics, set(targets), exclude=False) / n if n else None


def non_target_rate(metrics: dict, targets: list[str]) -> float | None:
    """Share of valid runs that fail with any label outside the targets (environment excluded)."""
    n = metrics.get("num_valid_runs") or 0
    return _labelled_runs(metrics, set(targets), exclude=True) / n if n else None


# ---------------------------------------------------------------- checks

def _check(name: str, passed: bool, detail: str, **values: Any) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail, **values}


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def required_valid_runs(cfg: dict) -> int:
    """Valid runs each system needs: the larger of the absolute floor and the fraction."""
    th = cfg["thresholds"]
    asked = cfg["heldout"]["num_tasks"] * cfg["k"]
    return max(int(th["min_valid_runs"]), math.ceil(th["min_valid_run_fraction"] * asked))


def check_comparable(cur: dict, cand: dict, cfg: dict) -> dict:
    split = cfg["heldout"]["task_split"]
    problems = []
    for who, m in (("current", cur), ("candidate", cand)):
        if m.get("task_split") != split:
            problems.append(f"{who} ran split {m.get('task_split')!r}, gate needs {split!r}")
        if m.get("k") != cfg["k"]:
            problems.append(f"{who} k={m.get('k')}, gate k={cfg['k']}")
    if sorted(cur.get("task_ids") or []) != sorted(cand.get("task_ids") or []):
        problems.append("current and candidate ran different task ids")
    return _check("comparable", not problems, "; ".join(problems) or f"same {split} tasks, k={cfg['k']}")


def check_valid_runs(cur: dict, cand: dict, cfg: dict) -> dict:
    need_runs = required_valid_runs(cfg)
    need_tasks = int(cfg["thresholds"]["min_complete_tasks"])
    problems = []
    for who, m in (("current", cur), ("candidate", cand)):
        runs, tasks = m.get("num_valid_runs"), m.get("num_tasks_complete")
        if not _num(runs) or runs < need_runs:
            problems.append(f"{who} has {runs} valid runs, needs {need_runs}")
        if not _num(tasks) or tasks < need_tasks:
            problems.append(f"{who} has {tasks} tasks with {cfg['k']} valid trials, needs {need_tasks}")
    return _check("valid_runs", not problems,
                  "too few valid runs: " + "; ".join(problems) if problems
                  else f"both have >= {need_runs} valid runs and >= {need_tasks} complete tasks",
                  required_runs=need_runs, required_complete_tasks=need_tasks)


def check_not_lower(name: str, cur: Any, cand: Any, max_drop: float) -> dict:
    if not (_num(cur) and _num(cand)):
        return _check(name, False, f"{name} missing (current={cur}, candidate={cand})", current=cur, candidate=cand)
    ok = cand >= cur - max_drop - EPS
    word = "keeps" if ok else "lowers"
    return _check(name, ok, f"candidate {word} {name}: {cand:.4f} vs {cur:.4f} (max drop {max_drop})",
                  current=cur, candidate=cand, delta=cand - cur)


def check_target(cur: dict, cand: dict, targets: list[str], min_drop: float) -> dict:
    if not targets:
        return _check("target", False, "no target failure label: the candidate must name what it fixes")
    rc, rn = target_rate(cur, targets), target_rate(cand, targets)
    if not (_num(rc) and _num(rn)):
        return _check("target", False, f"target rate missing (current={rc}, candidate={rn})", targets=targets)
    drop = rc - rn
    ok = drop > EPS and drop >= min_drop - EPS
    detail = (f"target {'+'.join(targets)} rate {rn:.4f} vs {rc:.4f}: drop {drop:.4f}"
              f" (needs > 0 and >= {min_drop})")
    return _check("target", ok, detail, targets=targets, current=rc, candidate=rn, delta=-drop)


def check_non_target(cur: dict, cand: dict, targets: list[str], max_increase: float | None) -> dict:
    if max_increase is None:
        return _check("non_target", True, "non-target check off", skipped=True)
    rc, rn = non_target_rate(cur, targets), non_target_rate(cand, targets)
    if not (_num(rc) and _num(rn)):
        return _check("non_target", False, f"non-target rate missing (current={rc}, candidate={rn})")
    ok = rn - rc <= max_increase + EPS
    return _check("non_target", ok, f"other labels rate {rn:.4f} vs {rc:.4f} (max increase {max_increase})",
                  current=rc, candidate=rn, delta=rn - rc)


def check_bfcl(cur: dict, cand: dict, cfg: dict) -> dict:
    bfcl = cfg.get("bfcl") or {}
    if not bfcl.get("enabled"):
        return _check("bfcl", True, "BFCL off in the gate config", skipped=True)
    margin = float(cfg["thresholds"]["bfcl_max_drop_points"])
    c, n = (cur.get("bfcl") or {}).get("accuracy_points"), (cand.get("bfcl") or {}).get("accuracy_points")
    if not (_num(c) and _num(n)):
        return _check("bfcl", False, f"BFCL enabled but missing (current={c}, candidate={n})")
    ok = n >= c - margin - EPS
    return _check("bfcl", ok, f"BFCL {n:.2f} vs {c:.2f} points (max drop {margin})",
                  current=c, candidate=n, delta=n - c)


# ---------------------------------------------------------------- decision

def public_metrics(m: dict) -> dict:
    """Metrics without the bulky per-run label list (kept in metrics_*.json files)."""
    return {k: v for k, v in m.items() if k not in ("failed_run_labels", "per_task")}


def decide(current: dict, candidate: dict, cfg: dict, targets: list[str] | None = None,
           now: str | None = None) -> dict:
    """The gate decision for ``candidate`` against ``current``."""
    validate_config(cfg)
    th = cfg["thresholds"]
    targets = list(targets or [])
    checks = [
        check_comparable(current, candidate, cfg),
        check_valid_runs(current, candidate, cfg),
        check_not_lower("pass_1", current.get("pass_1"), candidate.get("pass_1"), th["pass_1_max_drop"]),
        check_not_lower("pass_k", current.get("pass_k"), candidate.get("pass_k"), th["pass_k_max_drop"]),
        check_target(current, candidate, targets, th["target_label_min_drop"]),
        check_non_target(current, candidate, targets, th["non_target_max_increase"]),
        check_bfcl(current, candidate, cfg),
    ]
    failed = [c for c in checks if not c["passed"]]
    warnings = []
    if cfg.get("status") != "frozen":
        warnings.append("gate config status is 'proposed': the team has not frozen it yet")
    return {
        "decision": "reject" if failed else "promote",
        "reasons": [c["detail"] for c in failed] or [c["detail"] for c in checks if not c.get("skipped")],
        "checks": checks,
        "warnings": warnings,
        "targets": targets,
        "metrics_current": public_metrics(current),
        "metrics_candidate": public_metrics(candidate),
        "thresholds": copy.deepcopy(th),
        "k": cfg["k"],
        "heldout": copy.deepcopy(cfg["heldout"]),
        "config_status": cfg.get("status"),
        "config_sha": config_sha(cfg),
        "decided_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def metrics_row(round_: int, decision: dict, systems: dict) -> dict:
    """One metrics.jsonl row: round, systems compared, headline metrics, decision."""
    keys = ("system", "num_runs", "num_valid_runs", "num_tasks_complete", "pass_1", "pass_k",
            "failure_rate", "label_rates", "tool_call_error_rate", "avg_steps", "auditkit", "bfcl",
            "active_tool_patches", "agent_model")
    pick = lambda m: {k: m.get(k) for k in keys}  # noqa: E731
    return {
        "kind": "gate",
        "round": round_,
        "at": decision["decided_at"],
        "systems": systems,
        "targets": decision["targets"],
        "metrics": {"current": pick(decision["metrics_current"]),
                    "candidate": pick(decision["metrics_candidate"])},
        "decision": decision["decision"],
        "reasons": decision["reasons"],
        "config_status": decision["config_status"],
        "config_sha": decision["config_sha"],
    }

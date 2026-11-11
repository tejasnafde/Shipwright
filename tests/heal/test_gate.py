import copy
from pathlib import Path

import pytest

from shipwright.heal import gate as g

ROOT = Path(__file__).resolve().parents[2]
TASKS = [str(i) for i in range(40)]


def cfg(**th):
    c = {
        "status": "proposed",
        "heldout": {"domain": "retail", "task_split": "test", "num_tasks": 40},
        "k": 3,
        "seed": 300,
        "thresholds": {
            "min_valid_runs": 108, "min_valid_run_fraction": 0.9, "min_complete_tasks": 36,
            "pass_1_max_drop": 0.0, "pass_k_max_drop": 0.0, "target_label_min_drop": 0.025,
            "non_target_max_increase": 0.05, "bfcl_max_drop_points": 2.0,
        },
        "bfcl": {"enabled": True},
    }
    c["thresholds"].update(th)
    return c


def failures(n_target, n_other, n_both=0, target="tool_example_copied"):
    rows = [{"simulation_id": f"t{i}", "labels": [target]} for i in range(n_target)]
    rows += [{"simulation_id": f"o{i}", "labels": ["loop"]} for i in range(n_other)]
    rows += [{"simulation_id": f"b{i}", "labels": [target, "loop"]} for i in range(n_both)]
    return rows


def per_task(pass_1, valid, k=3):
    """AuditKIT per-task counts: ``valid`` decided runs over the 40 tasks, pass rate ``pass_1``."""
    decided = [min(k, max(0, valid - k * i)) for i in range(len(TASKS))]
    left = round((pass_1 or 0) * valid)
    out = {}
    for tid, n in zip(TASKS, decided):
        s = min(n, left)
        left -= s
        out[tid] = {"n_success": s, "n_decided": n, "pass_k": None}
    return out


def metrics(pass_1=0.3, pass_k=0.1, valid=120, complete=40, n_target=30, n_other=40, bfcl=50.0, **kw):
    m = {
        "per_task": per_task(pass_1, valid),
        "system": "x", "task_split": "test", "k": 3, "task_ids": list(TASKS),
        "num_runs": 120, "num_valid_runs": valid, "num_tasks_complete": complete,
        "pass_1": pass_1, "pass_k": pass_k,
        "failed_run_labels": failures(n_target, n_other),
        "bfcl": {"accuracy_points": bfcl} if bfcl is not None else None,
    }
    m.update(kw)
    return m


def check(decision, name):
    return next(c for c in decision["checks"] if c["name"] == name)


def test_clear_improvement_promotes():
    d = g.decide(metrics(), metrics(pass_1=0.35, pass_k=0.15, n_target=10), cfg(), ["tool_example_copied"])
    assert d["decision"] == "promote", d["reasons"]
    assert all(c["passed"] for c in d["checks"])
    assert d["warnings"]  # proposed config is flagged, but does not block


def test_too_few_valid_runs_rejects_even_when_everything_else_improves():
    cur = metrics(valid=60, complete=20)
    cand = metrics(pass_1=0.9, pass_k=0.9, n_target=0, valid=60, complete=20)
    d = g.decide(cur, cand, cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject"
    assert not check(d, "valid_runs")["passed"]
    assert "too few valid runs" in " ".join(d["reasons"])


def test_too_few_runs_on_one_side_only_rejects():
    d = g.decide(metrics(), metrics(n_target=0, valid=107), cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject" and not check(d, "valid_runs")["passed"]
    d = g.decide(metrics(complete=35), metrics(n_target=0), cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject" and not check(d, "valid_runs")["passed"]


def test_zero_runs_rejects():
    empty = metrics(valid=0, complete=0, pass_1=None, pass_k=None, failed_run_labels=[])
    d = g.decide(empty, empty, cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject"
    assert not check(d, "pass_1")["passed"] and not check(d, "target")["passed"]


def test_fraction_threshold_can_exceed_absolute_floor():
    c = cfg(min_valid_runs=10, min_valid_run_fraction=1.0)
    assert g.required_valid_runs(c) == 120
    d = g.decide(metrics(), metrics(n_target=10, valid=119), c, ["tool_example_copied"])
    assert not check(d, "valid_runs")["passed"]


def test_pass_rate_ties_pass():
    d = g.decide(metrics(), metrics(n_target=10), cfg(), ["tool_example_copied"])
    assert check(d, "pass_1")["passed"] and check(d, "pass_k")["passed"]
    assert d["decision"] == "promote"


def test_float_noise_tie_passes():
    d = g.decide(metrics(pass_1=0.1 + 0.2), metrics(pass_1=0.3, n_target=10), cfg(), ["tool_example_copied"])
    assert check(d, "pass_1")["passed"]


def test_lower_pass_1_rejects():
    d = g.decide(metrics(), metrics(pass_1=0.29, n_target=10), cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject" and not check(d, "pass_1")["passed"]


def test_lower_pass_k_rejects():
    d = g.decide(metrics(), metrics(pass_k=0.05, n_target=10), cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject" and not check(d, "pass_k")["passed"]


def test_pass_drop_within_allowed_margin_passes():
    d = g.decide(metrics(), metrics(pass_1=0.28, n_target=10), cfg(pass_1_max_drop=0.025), ["tool_example_copied"])
    assert check(d, "pass_1")["passed"]


def test_target_tie_rejects():
    d = g.decide(metrics(), metrics(), cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject"
    assert not check(d, "target")["passed"]


def test_target_drop_below_minimum_rejects():
    # 30 -> 28 of 120 runs: drop 1.7 points < 2.5
    d = g.decide(metrics(), metrics(n_target=28), cfg(), ["tool_example_copied"])
    assert not check(d, "target")["passed"]
    # 30 -> 27: drop exactly 2.5 points passes
    d = g.decide(metrics(), metrics(n_target=27), cfg(), ["tool_example_copied"])
    assert check(d, "target")["passed"]


def test_target_zero_min_drop_still_needs_strict_drop():
    d = g.decide(metrics(), metrics(), cfg(target_label_min_drop=0.0), ["tool_example_copied"])
    assert not check(d, "target")["passed"]
    d = g.decide(metrics(), metrics(n_target=29), cfg(target_label_min_drop=0.0), ["tool_example_copied"])
    assert check(d, "target")["passed"]


def test_target_already_zero_rejects():
    d = g.decide(metrics(n_target=0), metrics(n_target=0), cfg(), ["tool_example_copied"])
    assert not check(d, "target")["passed"]


def test_no_targets_rejects():
    d = g.decide(metrics(), metrics(n_target=0), cfg(), [])
    assert d["decision"] == "reject" and "no target" in check(d, "target")["detail"]


def test_multiple_targets_count_a_run_once():
    m = metrics(n_target=0, n_other=0)
    m["failed_run_labels"] = [{"simulation_id": "a", "labels": ["loop", "invented_id"]},
                              {"simulation_id": "b", "labels": ["environment"]}]
    assert g.target_rate(m, ["loop", "invented_id"]) == pytest.approx(1 / 120)
    assert g.non_target_rate(m, ["loop"]) == pytest.approx(1 / 120)  # invented_id; environment ignored


def test_non_target_regression_rejects():
    d = g.decide(metrics(), metrics(n_target=10, n_other=47), cfg(), ["tool_example_copied"])
    assert not check(d, "non_target")["passed"]
    d = g.decide(metrics(), metrics(n_target=10, n_other=46), cfg(), ["tool_example_copied"])
    assert check(d, "non_target")["passed"]


def test_non_target_check_can_be_off():
    d = g.decide(metrics(), metrics(n_target=10, n_other=100), cfg(non_target_max_increase=None),
                 ["tool_example_copied"])
    assert check(d, "non_target")["passed"] and check(d, "non_target")["skipped"]


def test_bfcl_within_margin_passes_and_beyond_rejects():
    d = g.decide(metrics(), metrics(n_target=10, bfcl=48.0), cfg(), ["tool_example_copied"])
    assert check(d, "bfcl")["passed"]
    d = g.decide(metrics(), metrics(n_target=10, bfcl=47.9), cfg(), ["tool_example_copied"])
    assert d["decision"] == "reject" and not check(d, "bfcl")["passed"]


def test_bfcl_missing_rejects_when_enabled():
    d = g.decide(metrics(), metrics(n_target=10, bfcl=None), cfg(), ["tool_example_copied"])
    assert not check(d, "bfcl")["passed"]


def test_bfcl_off_is_skipped():
    c = cfg()
    c["bfcl"]["enabled"] = False
    d = g.decide(metrics(bfcl=None), metrics(n_target=10, bfcl=None), c, ["tool_example_copied"])
    assert check(d, "bfcl")["passed"] and d["decision"] == "promote"


def test_different_tasks_or_split_or_k_rejects():
    d = g.decide(metrics(), metrics(n_target=10, task_ids=TASKS[:-1] + ["99"]), cfg(), ["tool_example_copied"])
    assert not check(d, "comparable")["passed"]
    d = g.decide(metrics(), metrics(n_target=10, task_split="train"), cfg(), ["tool_example_copied"])
    assert not check(d, "comparable")["passed"]
    d = g.decide(metrics(k=1), metrics(n_target=10, k=1), cfg(), ["tool_example_copied"])
    assert not check(d, "comparable")["passed"]


def test_decision_shape():
    d = g.decide(metrics(), metrics(n_target=10), cfg(), ["tool_example_copied"], now="2026-11-08T00:00:00+00:00")
    for key in ("decision", "reasons", "metrics_current", "metrics_candidate", "thresholds", "checks"):
        assert key in d
    assert "failed_run_labels" not in d["metrics_current"]
    row = g.metrics_row(1, d, {"current": {"name": "base"}, "candidate": {"name": "r1-tool"}})
    assert row["round"] == 1 and row["decision"] == "promote" and row["at"] == "2026-11-08T00:00:00+00:00"
    assert row["metrics"]["candidate"]["pass_1"] == 0.3


def test_decide_does_not_mutate_inputs():
    cur, cand, c = metrics(), metrics(n_target=10), cfg()
    before = copy.deepcopy((cur, cand, c))
    g.decide(cur, cand, c, ["tool_example_copied"])
    assert (cur, cand, c) == before


@pytest.mark.parametrize("bad", [
    {"status": "maybe"},
    {"k": 0},
    {"heldout": {"domain": "retail", "task_split": "train", "num_tasks": 40}},
])
def test_bad_config_raises(bad):
    c = cfg()
    c.update(bad)
    with pytest.raises(ValueError):
        g.validate_config(c)


def test_min_valid_runs_zero_is_refused():
    with pytest.raises(ValueError):
        g.validate_config(cfg(min_valid_runs=0))


def test_repo_gate_config_is_valid_and_proposed():
    c = g.load_config(ROOT / "configs" / "gate.yaml")
    assert c["status"] == "proposed"
    assert c["heldout"] == {"domain": "retail", "task_split": "test", "num_tasks": 20}
    assert c["k"] == 3 and g.required_valid_runs(c) == 54  # 90% of 20 held-out tasks x 3


def test_run_with_target_and_other_label_counts_for_both():
    m = metrics(n_target=0, n_other=0)
    m["failed_run_labels"] = failures(0, 0, n_both=6)
    assert g.target_rate(m, ["tool_example_copied"]) == pytest.approx(6 / 120)
    assert g.non_target_rate(m, ["tool_example_copied"]) == pytest.approx(6 / 120)


# ---------------------------------------------------------------- AgentTune DeploymentGate (AND check)

def test_agenttune_approval_alone_is_not_enough():
    # Same pass rate: AgentTune sees no regression and approves, but the target label did not drop.
    d = g.decide(metrics(), metrics(), cfg(), ["tool_example_copied"])
    at = check(d, "agenttune_gate")
    assert at["passed"] and at["agenttune_approved"]
    assert d["decision"] == "reject"
    assert d["verdicts"]["agenttune"]["approved"] and not d["verdicts"]["auditkit_thresholds"]["passed"]


def test_agenttune_too_few_samples_rejects_even_though_agenttune_approves():
    small = {"1": {"n_success": 1, "n_decided": 3}}
    d = g.decide(metrics(per_task=small), metrics(n_target=10, per_task=small), cfg(), ["tool_example_copied"])
    at = check(d, "agenttune_gate")
    assert at["agenttune_approved"] and not at["passed"]  # AgentTune alone would approve
    assert at["num_cases"] == 3 and at["min_test_samples"] == 108
    assert d["decision"] == "reject"
    # No paired runs at all: AgentTune approves an empty set, our floor rejects.
    d = g.decide(metrics(per_task={}), metrics(n_target=10, per_task={}), cfg(), ["tool_example_copied"])
    assert check(d, "agenttune_gate")["agenttune_approved"] and d["decision"] == "reject"


def test_our_thresholds_alone_are_not_enough():
    # pass^1 drop of 2 points is inside our margin, but AgentTune's tolerance is 0: it blocks.
    cur, cand = metrics(), metrics(pass_1=0.28, n_target=10)
    d = g.decide(cur, cand, cfg(pass_1_max_drop=0.025), ["tool_example_copied"])
    assert all(c["passed"] for c in d["checks"] if c["name"] != "agenttune_gate")
    at = check(d, "agenttune_gate")
    assert not at["agenttune_approved"] and "regressed" in at["agenttune_reason"]
    assert d["decision"] == "reject" and d["verdicts"]["auditkit_thresholds"]["passed"]
    # With an AgentTune tolerance that allows the drop, the same candidate promotes.
    c = cfg(pass_1_max_drop=0.025)
    c["agenttune_gate"] = {"task_regression_tol": 0.025}
    assert g.decide(cur, cand, c, ["tool_example_copied"])["decision"] == "promote"


def test_paired_test_set_counts_only_shared_decided_trials():
    from shipwright.heal.agenttune_bridge import paired_test_set

    cases, old, new = paired_test_set({"1": {"n_success": 2, "n_decided": 3}, "2": {"n_success": 0, "n_decided": 3}},
                                      {"1": {"n_success": 1, "n_decided": 2}, "3": {"n_success": 3, "n_decided": 3}})
    assert [c["pipeline_id"] for c in cases] == ["1#0", "1#1"]
    assert old == {"1#0": "PASS", "1#1": "PASS"} and new == {"1#0": "PASS", "1#1": "FAIL"}

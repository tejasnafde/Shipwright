import copy
import json
from pathlib import Path

import pytest

from shipwright import eval_metrics as em
from shipwright.classify import classify_run

FIX = Path(__file__).parent / "fixtures"
NAMES = ["passed_run", "wrong_tool_delivered_order", "invented_user_id_and_loop", "premature_transfer"]


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def with_trials(trace, k, fail_trials=(), infra_trials=()):
    out = []
    for tr in range(k):
        t = copy.deepcopy(trace)
        t["simulation_id"] = f"{trace['simulation_id']}-{tr}"
        t["trial"] = tr
        if tr in fail_trials:
            t["passed"], t["reward"] = False, 0.0
        if tr in infra_trials:
            t["termination_reason"], t["reward"], t["passed"] = "infrastructure_error", None, False
        out.append(t)
    return out


def run_of(traces):
    fails = [r for r in (classify_run(t) for t in traces) if r]
    return {"traces": traces, "failures": fails, "run_config": {"task_split": "test"}, "tools": []}


def test_openai_adapter_shapes():
    t = load("wrong_tool_delivered_order")
    msgs = em.trace_to_openai_messages(t["messages"])
    calls = [m for m in msgs if m.get("tool_calls")]
    assert calls and calls[0]["tool_calls"][0]["type"] == "function"
    assert isinstance(calls[0]["tool_calls"][0]["function"]["arguments"], str)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    ids = {tc["id"] for m in calls for tc in m["tool_calls"]}
    assert tool_msgs and all(m["tool_call_id"] in ids for m in tool_msgs)


def test_episode_coverage_and_outcome_state():
    ep = em.episode_from_trace(load("passed_run"))
    assert ep.coverage_label() == "full"
    assert ep.final_state == {"tau2_pass": True, "tau2_reward": 1.0}
    infra = with_trials(load("passed_run"), 1, infra_trials=(0,))[0]
    ep = em.episode_from_trace(infra)
    assert ep.final_state is None or "tau2_pass" not in (ep.final_state or {})


def test_reference_turns_from_action_checks():
    t = load("wrong_tool_delivered_order")
    ref = em.reference_turns(t)
    assert ref and all(len(turn) == 1 and "name" in turn[0] for turn in ref)


def test_single_trial_metrics_on_fixtures():
    m = em.compute_metrics(run_of([load(n) for n in NAMES]), k=1, system="fx")
    assert m["num_runs"] == 4 and m["num_valid_runs"] == 4 and m["num_tasks"] == 4
    assert m["pass_1"] == pytest.approx(0.25) and m["pass_k"] == pytest.approx(0.25)
    assert m["failure_rate"] == pytest.approx(0.75)
    assert m["label_rates"]["invented_id"] == pytest.approx(0.75)
    assert m["label_rates"]["loop"] == pytest.approx(0.5)
    assert m["tool_calls"] == 24 and m["tool_call_errors"] == 8
    assert m["tool_call_error_rate"] == pytest.approx(1 / 3)
    assert m["avg_steps"] == pytest.approx(12.25)
    ak = m["auditkit"]
    assert 0 < ak["tool_call_f1"] < 1
    assert ak["tool_call_validity"] is None  # no tool schemas given
    assert "tool_call_validity: case.allowed_tools" in ak["ineligible"]
    assert len(m["failed_run_labels"]) == 3


def test_pass_k_with_trials_and_infra_errors():
    traces = []
    traces += with_trials(load("passed_run"), 3, fail_trials=(2,))  # 2/3 pass
    traces += with_trials(load("wrong_tool_delivered_order"), 3, infra_trials=(1,))  # 0/2 decided
    traces += with_trials(load("premature_transfer"), 3)  # 0/3
    m = em.compute_metrics(run_of(traces), k=3)
    assert m["num_runs"] == 9 and m["num_valid_runs"] == 8 and m["num_infra_errors"] == 1
    # pass^1: mean of per-task success rates over decided trials
    assert m["pass_1"] == pytest.approx((2 / 3 + 0 + 0) / 3)
    # pass^k: only tasks with k decided trials; 2/3 pass gives C(2,3)/C(3,3) = 0
    assert m["num_tasks_complete"] == 2 and m["pass_k"] == 0.0
    assert m["per_task"]["1"]["pass_k"] is None


def test_pass_k_all_pass():
    traces = with_trials(load("passed_run"), 3) + with_trials(load("premature_transfer"), 3)
    m = em.compute_metrics(run_of(traces), k=3)
    assert m["pass_k"] == pytest.approx(0.5) and m["pass_1"] == pytest.approx(0.5)


def test_pass_k_lower_k_than_trials():
    traces = with_trials(load("passed_run"), 3, fail_trials=(0,))  # 2 of 3
    m = em.compute_metrics(run_of(traces), k=2)
    assert m["pass_k"] == pytest.approx(1 / 3)  # C(2,2)/C(3,2)


def test_label_rates_ignore_infra_runs():
    traces = with_trials(load("wrong_tool_delivered_order"), 2, infra_trials=(1,))
    m = em.compute_metrics(run_of(traces), k=2)
    assert "environment" not in m["label_rates"]
    assert m["num_valid_runs"] == 1


def test_load_run_needs_failures(tmp_path):
    (tmp_path / "traces.jsonl").write_text(json.dumps(load("passed_run")) + "\n")
    with pytest.raises(FileNotFoundError):
        em.load_run(tmp_path)
    (tmp_path / "failures.jsonl").write_text("")
    (tmp_path / "prompts.json").write_text(json.dumps({"abc": {"system": "s", "tools": [
        {"type": "function", "function": {"name": "get_user_details", "parameters": {"type": "object"}}}]}}))
    run = em.load_run(tmp_path)
    assert run["tools"][0]["function"]["name"] == "get_user_details"
    m = em.compute_metrics(run, k=1)
    assert m["pass_1"] == 1.0 and m["auditkit"]["tool_call_validity"] is not None

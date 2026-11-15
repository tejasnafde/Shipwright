"""DETECT, CLASSIFY and the retrain trigger through AgentTune's classes. No network: the
classifier's litellm call is replaced by a fake."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shipwright.heal import agenttune_bridge as ab
from shipwright.heal.classify import summarize_failures

FIX = Path(__file__).parent.parent / "fixtures"
EXAMPLES = {"#W0000000", "sara_doe_496", "1008292230", "6086499569", "gift_card_0000000"}


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def synthetic(calls, reward=0.0, sim="s1", task="x"):
    """A trace from (name, args, error) tuples."""
    messages, turns = [{"role": "user", "content": "my order is #W1111111"}], []
    for i, (name, args, err) in enumerate(calls):
        idx, cid = len(messages), f"c{i}"
        messages.append({"role": "assistant", "tool_calls": [{"id": cid, "name": name, "arguments": args}]})
        messages.append({"role": "tool", "id": cid, "content": "Error: not found" if err else "{}", "error": err})
        turns.append({"message_index": idx, "raw_output": f"raw text {i}", "parsed_tool_calls": [
            {"id": cid, "name": name, "arguments": args}],
            "tool_results": [{"id": cid, "name": name, "content": "Error: not found" if err else "{}", "error": err}]})
    return {"task_id": task, "simulation_id": sim, "reward": reward, "passed": reward >= 1,
            "termination_reason": "user_stop", "messages": messages, "agent_turns": turns, "reward_info": {}}


ORDER = {"order_id": "#W1111111"}


# ---------------------------------------------------------------- DETECT

def test_audit_records_have_the_event_log_schema():
    recs = ab.trace_audit_records(synthetic([("get_order_details", {"order_id": "#W1111111", "a": 1}, True)]), "s1")
    [r] = recs
    assert {"trajectory_id", "stage_name", "tool_name", "stage_type", "state_snapshot", "status",
            "error_details"} <= set(r)
    assert r["stage_name"] == 'get_order_details {"a": 1, "order_id": "#W1111111"}'  # sorted args
    assert r["stage_type"] == "tool_call" and r["status"] == "error"
    assert r["state_snapshot"]["raw_output"] == "raw text 0"


def test_detector_finds_loop_and_tool_crash_and_rescans(tmp_path):
    traces = [synthetic([("get_order_details", ORDER, False)] * 3, sim="loop"),
              synthetic([("cancel_pending_order", ORDER, True)], sim="crash"),
              synthetic([("get_order_details", ORDER, False)] * 2, sim="ok")]
    audit = ab.write_audit_log(traces, tmp_path / "audit.jsonl")
    hits = ab.detect(audit)
    assert [f.failure_type for f in hits["loop"]] == ["loop_collapse"]
    assert [f.failure_type for f in hits["crash"]] == ["tool_crash"]
    assert "ok" not in hits
    assert ab.detect(audit).keys() == hits.keys()  # the detector's offset file does not hide a rescan
    assert not (tmp_path / "audit.jsonl.offset").exists()


def test_no_llm_union_of_outcome_and_detector(tmp_path):
    traces = [load("invented_user_id_and_loop"), load("passed_run"),
              synthetic([("cancel_pending_order", ORDER, True), ("get_order_details", ORDER, False)],
                        reward=1.0, sim="passed-with-crash")]
    rows = ab.classify_traces(traces, tmp_path)
    assert (tmp_path / "audit.jsonl").exists()
    failed = rows[0]
    assert failed["outcome_failed"] and failed["detected_by"][0] == "outcome"
    assert failed["route"] == "weight_fix" and failed["agenttune_root_cause"] is None
    extra = next(r for r in rows if r["simulation_id"] == "passed-with-crash")
    assert extra["detected_by"] == ["agenttune_detector"] and extra["route"] == "no_fix"
    assert extra["outcome_failed"] is False and extra["agenttune_failures"][0]["failure_type"] == "tool_crash"
    # The fixture's passed run also had two tool errors, so the detector flags it too.
    assert sum(r["outcome_failed"] is False for r in rows) == 2
    s = summarize_failures(rows, len(traces))
    assert s["num_failed"] == 1 and s["num_detector_only"] == 2
    assert s["per_label"] == {lab: 1 for lab in failed["labels"]}  # detector-only rows add no labels


# ---------------------------------------------------------------- CLASSIFY

def fake_litellm(monkeypatch, reply):
    import agenttune.decide.closed_loop.failure_classifier as fc

    seen = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])

    monkeypatch.setattr(fc.litellm, "acompletion", acompletion)
    return seen


def test_agenttune_classifier_labels_and_routes(monkeypatch, tmp_path):
    seen = fake_litellm(monkeypatch, {"root_cause": "hallucinated_output", "confidence": 0.8, "analysis": "made it up"})
    t = synthetic([("search_products", {"page": i}, False) for i in range(40)])
    rows = ab.classify_traces([t], tmp_path, instructions={"x": "cancel my order"},
                              classifier=ab.make_classifier("fake/model"))
    [row] = rows
    assert row["agenttune_root_cause"] == "hallucinated_output" and row["agenttune_confidence"] == 0.8
    assert row["labels"] == ["invented_answer"] and row["source"] == "agenttune"
    assert row["route"] == "weight_fix" and row["explanation"] == "made it up"
    [call] = seen
    assert call["model"] == "fake/model"
    assert "wrong_tool, wrong_routing" in call["messages"][0]["content"]  # AgentTune's own system prompt
    user = call["messages"][1]["content"]
    assert "cancel my order" in user and "[79] agent calls search_products" in user and len(user) > 2000


def test_tool_layer_rules_win_over_agenttune(monkeypatch, tmp_path):
    seen = fake_litellm(monkeypatch, {"root_cause": "wrong_tool", "confidence": 0.9, "analysis": "a"})
    [row] = ab.classify_traces([load("wrong_tool_delivered_order")], tmp_path, example_values=EXAMPLES,
                               classifier=ab.make_classifier("fake/model"))
    assert row["route"] == "tool_fix" and row["source"] == "rules"
    assert "tool_example_copied" in row["labels"] and "wrong_tool" in row["labels"]
    assert row["agenttune_root_cause"] == "wrong_tool"
    assert "tool_example_copied" in seen[0]["messages"][1]["content"]  # rule hits go in as hints


def test_classifier_error_keeps_rules(monkeypatch, tmp_path):
    fake_litellm(monkeypatch, RuntimeError("no network"))
    [row] = ab.classify_traces([load("invented_user_id_and_loop")], tmp_path,
                               classifier=ab.make_classifier("fake/model"))
    assert row["agenttune_root_cause"] is None and "no network" in row["agenttune_error"]
    assert row["source"] == "rules" and row["route"] == "weight_fix"


def test_vertex_classifier_routes_with_a_fresh_token(monkeypatch, tmp_path):
    from shipwright.llm import vertex

    class Expired(Exception):
        status_code = 401

    tokens = iter(["tok-1", "tok-2"])
    monkeypatch.setattr(vertex, "_provider", vertex.TokenProvider(fetch=lambda: next(tokens)))
    seen = fake_litellm(monkeypatch, {"root_cause": "wrong_tool", "confidence": 0.7, "analysis": "w"})
    import agenttune.decide.closed_loop.failure_classifier as fc

    ok = fc.litellm.acompletion
    calls = []

    async def first_401(**kwargs):
        calls.append(kwargs["api_key"])
        if len(calls) == 1:
            raise Expired("401")
        return await ok(**kwargs)

    monkeypatch.setattr(fc.litellm, "acompletion", first_401)
    clf = ab.make_classifier(vertex.JUDGE_MODEL)
    [row] = ab.classify_traces([synthetic([("search_products", {"page": 1}, False)])], tmp_path, classifier=clf)
    assert row["agenttune_root_cause"] == "wrong_tool" and row["source"] == "agenttune"
    assert calls == ["tok-1", "tok-2"]  # 401, new token, retry
    [call] = seen
    assert call["model"] == "openai/openai/gpt-oss-120b-maas"
    assert call["api_base"].startswith("https://aiplatform.googleapis.com/v1/projects/")
    assert "tok" not in repr(clf.__dict__)  # the token is not stored on the classifier


def test_vertex_classifier_waits_and_retries_a_429_and_caps_concurrency(monkeypatch, tmp_path):
    import asyncio

    from shipwright.llm import vertex

    class RateLimited(Exception):
        status_code = 429

    monkeypatch.setattr(vertex, "_provider", vertex.TokenProvider(fetch=lambda: "tok"))
    import agenttune.decide.closed_loop.failure_classifier as fc

    real_sleep = asyncio.sleep
    waits = []
    running = 0
    most_running = 0
    attempts_by_call = {}

    async def no_wait(seconds):
        waits.append(seconds)

    async def acompletion(**kwargs):
        nonlocal running, most_running
        running += 1
        most_running = max(most_running, running)
        await real_sleep(0)  # let the other classifications start
        running -= 1
        key = kwargs["messages"][-1]["content"]
        attempts_by_call[key] = attempts_by_call.get(key, 0) + 1
        if attempts_by_call[key] == 1:
            raise RateLimited("429")  # every classification is rate limited once
        reply = {"root_cause": "wrong_tool", "confidence": 0.7, "analysis": "w"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])

    monkeypatch.setattr(fc.litellm, "acompletion", acompletion)
    monkeypatch.setattr(ab.asyncio, "sleep", no_wait)
    traces = [synthetic([("search_products", {"page": number}, False)]) for number in range(10)]
    rows = ab.classify_traces(traces, tmp_path, classifier=ab.make_classifier(vertex.JUDGE_MODEL))
    assert all(row["agenttune_root_cause"] == "wrong_tool" for row in rows)  # nothing fell back to rules
    assert len(waits) == 10 and most_running <= ab.CLASSIFIER_CONCURRENCY


def test_vertex_classifier_error_is_recorded_not_raised(monkeypatch, tmp_path):
    from shipwright.llm import vertex

    def no_gcloud():
        raise RuntimeError("gcloud failed")

    monkeypatch.setattr(vertex, "_provider", vertex.TokenProvider(fetch=no_gcloud))
    [row] = ab.classify_traces([synthetic([("search_products", {"page": 1}, False)])], tmp_path,
                               classifier=ab.make_classifier(vertex.JUDGE_MODEL))
    assert row["agenttune_root_cause"] is None and "gcloud failed" in row["agenttune_error"]


def test_every_agenttune_root_cause_is_mapped():
    from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
    from shipwright.heal.classify import JUDGE_LABELS

    assert set(ab.AGENTTUNE_ROOT_CAUSE_MAP) == set(FailureClassifier.ROOT_CAUSES)
    assert all(label in JUDGE_LABELS for label, _ in ab.AGENTTUNE_ROOT_CAUSE_MAP.values())


# ---------------------------------------------------------------- RETRAIN TRIGGER

def weight_rows(n, label="invented_id"):
    return [{"task_id": str(i), "simulation_id": f"s{i}", "route": "weight_fix", "labels": [label],
             "agenttune_root_cause": "hallucinated_output", "outcome_failed": True} for i in range(n)]


SFT = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]


def test_trigger_fires_on_dominant_corrected_failures():
    st = ab.retrain_status(weight_rows(12), {str(i): SFT for i in range(12)})
    assert st["should_train"] and st["reason"] == "dominance:hallucinated_output"
    assert st["num_with_correction"] == 12 and st["buffer_health"]["drop_rate"] == 0.0


def test_trigger_drop_rate_gate_blocks_uncorrected_buffer():
    st = ab.retrain_status(weight_rows(12), {"0": SFT})
    assert not st["should_train"] and st["reason"] == "drop_rate_too_high"


def test_trigger_below_threshold_and_tool_rows_ignored():
    rows = weight_rows(2) + [{"task_id": "9", "route": "tool_fix", "labels": ["tool_example_copied"]}]
    st = ab.retrain_status(rows, {}, {"dominance_ratio": 1.1, "novel_type_min_count": 5})
    assert not st["should_train"] and st["reason"] == "below_threshold"
    assert st["num_weight_fix_failures"] == 2
    with pytest.raises(ValueError):
        ab.trigger_config({"bogus": 1})


def test_trigger_reward_drift_needs_a_baseline():
    st = ab.retrain_status(weight_rows(1), {}, {"dominance_ratio": 1.1, "novel_type_min_count": 5},
                           baseline_rewards=[1.0, 1.0, 0.0, 1.0], rewards=[0.0, 0.0, 0.0, 0.0])
    assert st["should_train"] and st["reason"] == "reward_drift"
def test_round_status_script(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import round_status

    run = tmp_path / "run"
    run.mkdir()
    (run / "failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in weight_rows(12)))
    sft = tmp_path / "sft.jsonl"
    sft.write_text("".join(json.dumps({"task_id": str(i), "messages": SFT}) + "\n" for i in range(12)))
    assert round_status.main([str(run), "--sft", str(sft)]) == 0
    st = json.loads((run / "round_status.json").read_text())
    assert st["should_train"]
    assert st["trigger_config"]["max_drop_rate"] == 0.6

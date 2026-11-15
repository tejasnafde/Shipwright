"""rounds.py: which dirs a round uses, what the gate said in words, how tasks changed."""

import json

from shipwright.control_room import rounds


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_a_round_finds_its_own_run_dirs_and_the_next_rounds_after_run(tmp_path):
    write(tmp_path / "round-3.json", {"stages": {"baseline": {"num_runs": 74, "num_passed": 5},
                                                  "gate": {"decision": "reject", "pass_1": {"current": 0.10, "candidate": 0.15}}}})
    for name in ("r3-baseline", "r3-teacher", "r4-cand-train"):
        (tmp_path / name).mkdir()
    write(tmp_path / "gate-r3" / "gate_decision.json", {"decision": "reject"})
    [entry] = rounds.discover_rounds(tmp_path)
    assert (entry.before, entry.teacher, entry.gate, entry.after) == ("r3-baseline", "r3-teacher", "gate-r3", "r4-cand-train")
    assert entry.outcome == "rejected" and "10% to 15%" in entry.label


def test_a_loop_round_starts_from_the_previous_adapter_on_train_tasks(tmp_path):
    write(tmp_path / "round-4.json", {"stages": {"candidate_train": {"exit_code": 0}}})
    for name in ("r4-cand-train", "r4-sft"):
        (tmp_path / name).mkdir()
    [entry] = rounds.discover_rounds(tmp_path)
    assert entry.before == "r4-cand-train" and entry.teacher == "r4-sft" and entry.outcome == "unfinished"


def test_round_0_and_rounds_without_a_before_run_are_left_out(tmp_path):
    write(tmp_path / "round-0.json", {"stages": {}})
    (tmp_path / "r0-baseline").mkdir()
    write(tmp_path / "round-2.json", {"stages": {}})  # no r2-baseline dir
    assert rounds.discover_rounds(tmp_path) == []


def test_the_round_3_gate_in_words():
    decision = {
        "metrics_current": {"pass_1": 0.10, "pass_k": 0.10, "tool_call_error_rate": 0.46, "avg_steps": 10.0,
                            "label_rates": {"invented_id": 0.667, "loop": 0.05, "premature_transfer": 0.283},
                            "bfcl": {"accuracy_points": 43.33}},
        "metrics_candidate": {"pass_1": 0.15, "pass_k": 0.05, "tool_call_error_rate": 0.22, "avg_steps": 18.65,
                              "label_rates": {"invented_id": 0.233, "loop": 0.45, "premature_transfer": 0.0},
                              "bfcl": {"accuracy_points": 35.0}},
    }
    findings = {finding.text: finding.kind for finding in rounds.what_changed(decision)}
    assert findings["Runs that failed by making up ids: 67% to 23%."] == "good"
    assert findings["Runs that failed by repeating the same call: 5% to 45%."] == "bad"
    assert findings["Runs that failed by transferring to a human too early: 28% to 0%."] == "good"
    assert findings["Tool calls that returned an error: 46% to 22%."] == "good"
    assert findings["General tool-calling quiz (BFCL): 43.3 to 35.0 points."] == "bad"
    assert findings["Steps per conversation: 10 to 19."] == "bad"


def trace(task, reward, termination="user_stop"):
    return {"task_id": task, "reward": reward, "termination_reason": termination}


def test_tasks_are_compared_by_how_often_they_pass_and_infra_errors_do_not_count():
    before = [trace(1, 0), trace(1, 0), trace(2, 1), trace(2, 1), trace(3, 0), trace(3, 1), trace(4, 0), trace(4, 0, "infrastructure_error")]
    after = [trace(1, 1), trace(1, 0), trace(2, 0), trace(2, 0), trace(3, 1), trace(3, 1), trace(4, 0)]
    changes = {change.task_id: change for change in rounds.compare_tasks(before, after, {"1": "0-1 writes"})}
    assert changes["1"].change == "fixed" and changes["1"].group == "0-1 writes"
    assert changes["2"].change == "broke"
    assert changes["3"].change == "better"
    assert changes["4"].change == "same" and changes["4"].before_trials == 1
    assert [change.change for change in rounds.compare_tasks(before, after)][:2] == ["fixed", "broke"]  # changes first


def test_a_round_that_reuses_an_earlier_train_run_takes_it_as_its_before_run(tmp_path):
    write(tmp_path / "round-5.json", {"failures_run": "r4-cand-train", "stages": {"train": {"exit_code": 0}}})
    (tmp_path / "r4-cand-train").mkdir()
    [entry] = rounds.discover_rounds(tmp_path)
    assert entry.before == "r4-cand-train" and entry.outcome == "unfinished"


def test_a_round_written_by_the_loop_keeps_the_whole_gate_decision_in_its_manifest(tmp_path):
    decision = {"round": 16, "decision": "promote",
                "checks": [{"name": "pass_1", "passed": True, "current": 0.1, "candidate": 0.15}]}
    write(tmp_path / "round-16.json", {"failures_run": "r16-cand-train",
                                       "stages": {"candidate_train": {}, "gate": {"decision": decision}}})
    (tmp_path / "r16-cand-train").mkdir()
    entry = rounds.discover_rounds(tmp_path)[0]
    assert entry.outcome == "promoted" and entry.note == "pass^1 10% to 15%"


def test_the_gate_history_lists_every_decision_in_round_order_with_why_it_refused(tmp_path):
    def gate(number, decision, failed):
        names = ["pass_1", "non_target", "agenttune_gate"]
        checks = [{"name": name, "passed": name not in failed} for name in names]
        checks[0].update(current=0.1, candidate=0.15)
        write(tmp_path / f"gate-r{number}" / "gate_decision.json", {"decision": decision, "checks": checks})
    gate(16, "promote", [])
    gate(8, "reject", ["non_target"])
    history = rounds.gate_history(tmp_path)
    assert [record.round_number for record in history] == [8, 16]
    assert history[0].why_refused == "a new kind of failure rose" and history[0].agenttune_approved
    assert history[1].checks_passed == 3 and history[1].candidate_pass_1 == 0.15

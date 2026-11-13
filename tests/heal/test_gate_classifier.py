"""The gate classifies the failures of both systems with AgentTune's FailureClassifier (2026-11-13).
No network: the classifier's litellm call is a fake."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from shipwright.heal import agenttune_bridge, gate
from shipwright.heal.classify import JUDGE_LABELS
from shipwright.heal.eval_metrics import compute_metrics, load_run

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import gate_eval  # noqa: E402

GATE_CONFIG = gate.load_config(ROOT / "configs" / "gate.yaml")


def read_only_failure(simulation_id: str) -> dict:
    """A failed run that looked an order up and never made a write call: no rule labels it."""
    call = {"id": "c0", "name": "get_order_details", "arguments": {"order_id": "#W1111111"}}
    result = {"id": "c0", "name": "get_order_details", "content": "{}", "error": False}
    return {
        "task_id": simulation_id, "simulation_id": simulation_id, "trial": 0, "reward": 0.0, "passed": False,
        "termination_reason": "user_stop", "reward_info": {},
        "messages": [
            {"role": "user", "content": "please return order #W1111111"},
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", **result},
            {"role": "assistant", "content": "Anything else?"},
        ],
        "agent_turns": [{"message_index": 1, "raw_output": "lookup", "parsed_tool_calls": [call], "tool_results": [result]}],
    }


def fake_classifier_reply(monkeypatch, root_cause: str) -> None:
    import agenttune.decide.closed_loop.failure_classifier as failure_classifier

    async def acompletion(**_):
        reply = {"root_cause": root_cause, "confidence": 0.9, "analysis": "never made the write call"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])

    monkeypatch.setattr(failure_classifier.litellm, "acompletion", acompletion)


def run_directory(path: Path, failures: list[dict]) -> Path:
    path.mkdir(parents=True)
    traces = [read_only_failure(f"s{index}") for index in range(len(failures))]
    (path / "traces.jsonl").write_text("".join(json.dumps(trace) + "\n" for trace in traces))
    (path / "failures.jsonl").write_text("".join(json.dumps(row) + "\n" for row in failures))
    (path / "run_config.json").write_text(json.dumps({"task_split": "test"}))
    return path


def test_unlabelled_no_write_failure_gets_a_frozen_label(monkeypatch, tmp_path):
    traces = [read_only_failure("s0")]
    rules_only = agenttune_bridge.classify_traces(traces, tmp_path)
    assert rules_only[0]["labels"] == ["unclassified"]

    fake_classifier_reply(monkeypatch, "incomplete_reasoning")
    classifier = agenttune_bridge.make_classifier("fake/model")
    [row] = agenttune_bridge.classify_traces(traces, tmp_path, classifier=classifier)
    assert row["labels"] == ["skipped_policy_step"] and row["labels"][0] in JUDGE_LABELS

    rules_metrics = compute_metrics(load_run(run_directory(tmp_path / "rules", rules_only)), 3)
    classified_metrics = compute_metrics(load_run(run_directory(tmp_path / "agenttune", [row])), 3)
    assert rules_metrics["label_coverage"] == {"rules": 0, "agenttune": 0, "unlabelled": 1, "agenttune_errors": 0}
    assert classified_metrics["label_coverage"] == {"rules": 0, "agenttune": 1, "unlabelled": 0, "agenttune_errors": 0}
    targets = ["skipped_policy_step"]
    assert gate.non_target_rate(rules_metrics, targets) == 1.0
    assert gate.non_target_rate(classified_metrics, targets) == 0.0


def test_both_systems_are_classified_the_same_way(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(gate_eval, "run", lambda command, log_path=None: commands.append(command) or 0)
    current = run_directory(tmp_path / "current", [])
    candidate = run_directory(tmp_path / "candidate", [])
    gate_eval.main(["--round", "1", "--current-run", str(current), "--candidate-run", str(candidate),
                    "--output-dir", str(tmp_path / "out"), "--reclassify"])
    assert [command[2] for command in commands] == [str(current), str(candidate)]
    assert commands[0][3:] == commands[1][3:] == ["--classifier-model", "vertex/openai/gpt-oss-120b-maas"]


def test_no_llm_keeps_rules_only(tmp_path):
    assert gate_eval.classify_cmd(GATE_CONFIG, tmp_path, no_llm=True)[-1] == "--no-llm"
    assert gate_eval.classify_cmd({"classify": {"classifier_model": None}}, tmp_path)[-1] == "--no-llm"


def test_frozen_thresholds_are_unchanged():
    assert GATE_CONFIG["status"] == "frozen"
    assert GATE_CONFIG["thresholds"] == {
        "min_valid_runs": 54, "min_valid_run_fraction": 0.9, "min_complete_tasks": 18,
        "pass_1_max_drop": 0.0, "pass_k_max_drop": 0.0, "target_label_min_drop": 0.05,
        "non_target_max_increase": 0.05, "bfcl_max_drop_points": 2.0,
    }

"""scripts/reproduce.py: recomputes the write-up's numbers from run files, offline here (no Hub)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce  # noqa: E402
from shipwright.heal.eval_metrics import exact_mcnemar_p  # noqa: E402


def write(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, list):
        path.write_text("\n".join(json.dumps(row) for row in content) + "\n")
    else:
        path.write_text(json.dumps(content))


def gate_metrics(pass_1, tool_error_rate, bfcl, labels):
    return {"num_runs": 60, "num_valid_runs": 60, "pass_1": pass_1, "pass_k": 0.1,
            "tool_call_error_rate": tool_error_rate, "bfcl": {"accuracy_points": bfcl},
            "label_rates": {label: count / 60 for label, count in labels.items()}}


def traces(passing_pairs):
    return [{"task_id": str(task), "trial": trial, "passed": (task, trial) in passing_pairs}
            for task in range(20) for trial in range(3)]


def published_runs(cache: Path, base_pass_1=0.1) -> None:
    gate = cache / reproduce.GATE_16
    write(gate / "gate_decision.json", {"decision": "promote", "checks": [{"passed": True}] * 8})
    write(gate / "metrics_current.json", gate_metrics(base_pass_1, 0.443, 45.0, {
        "invented_id": 40, "premature_transfer": 11, "invented_answer": 8, "malformed_args": 4, "unknown_tool": 2,
        "wrong_tool": 11, "skipped_policy_step": 25, "loop": 8}))
    write(gate / "metrics_candidate.json", gate_metrics(0.15, 0.057, 73.33, {
        "wrong_tool": 4, "skipped_policy_step": 31, "loop": 12}))
    hidden = cache / reproduce.HIDDEN
    write(hidden / "base" / "traces.jsonl", traces(set()))
    write(hidden / reproduce.HIDDEN_STUDENT / "traces.jsonl", traces({(3, 0), (4, 0), (4, 1), (5, 0), (5, 1),
                                                                       (5, 2)}))
    write(hidden / "base" / "summary.json", {"tool_calls": 192, "tool_errors": 107})
    write(hidden / reproduce.HIDDEN_STUDENT / "summary.json", {"tool_calls": 485, "tool_errors": 23})


def test_every_claim_matches_files_that_say_what_the_write_up_says(tmp_path, capsys):
    published_runs(tmp_path)
    assert reproduce.main(["--offline", "--cache-dir", str(tmp_path)]) == 0
    assert "19 of 19 claims match" in capsys.readouterr().out


def test_a_number_the_files_do_not_support_fails_the_check(tmp_path, capsys):
    published_runs(tmp_path, base_pass_1=0.05)
    assert reproduce.main(["--offline", "--cache-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "MISMATCH" in output and "18 of 19 claims match" in output


def test_exact_mcnemar():
    assert round(exact_mcnemar_p(6, 0), 5) == 0.03125
    assert exact_mcnemar_p(0, 0) == 1.0
    assert round(exact_mcnemar_p(8, 0), 4) == 0.0078

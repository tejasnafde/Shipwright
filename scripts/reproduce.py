"""Recompute the write-up's headline numbers from the published run files, and check each against WRITEUP.md.

    python scripts/reproduce.py                 # downloads about 2 MB from the public runs dataset, no token, no GPU
    python scripts/reproduce.py --offline       # only what is already in --cache-dir

For a judge, or for us before a submission: every number below comes from a file the gate or the hidden-split run
wrote (`mitushigoyal5/shipwright-runs`, a public dataset), not from the write-up. A claim prints MATCH when the value
recomputed from those files, rounded the way the write-up shows it, equals the write-up's number; any MISMATCH ends
with exit code 1. Nothing is run or changed: this only reads.

Sources:
  gate 16 (round 16 promoted)   gate/round-16/r16-weight/  gate_decision.json, metrics_current.json (base),
                                metrics_candidate.json (round 16): 20 held-out tasks x 3 trials per system
  hidden split (opened once)    gate/round-15/r11-r7-fixes-hidden/{base,r11-r7-fixes}/  traces.jsonl, summary.json:
                                base against the round 7 adapter + guards, 20 hidden tasks x 3 trials
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright import cloud  # noqa: E402
from shipwright.heal.eval_metrics import exact_mcnemar_p  # noqa: E402
from shipwright.observability import log  # noqa: E402

GATE_16 = "gate/round-16/r16-weight"
HIDDEN = "gate/round-15/r11-r7-fixes-hidden"
HIDDEN_STUDENT = "r11-r7-fixes"
RUNS_PER_SYSTEM = 60


@dataclass
class Claim:
    what: str            # the claim, as the write-up states it
    written: str         # the number in WRITEUP.md
    recomputed: str      # the same number from the run files, rounded the same way

    @property
    def matches(self) -> bool:
        return self.written == self.recomputed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=cloud.RUNS_REPO, help="the runs dataset on the Hub")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "_reproduce")
    parser.add_argument("--offline", action="store_true", help="read only files already in --cache-dir")
    return parser.parse_args(argv)


def fetch(options: argparse.Namespace, path_in_repo: str) -> Path:
    local_path = options.cache_dir / path_in_repo
    if local_path.exists() or options.offline:
        return local_path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(options.repo, path_in_repo, repo_type="dataset", local_dir=options.cache_dir))


def read_json(options: argparse.Namespace, path_in_repo: str) -> dict:
    return json.loads(fetch(options, path_in_repo).read_text())


def read_traces(options: argparse.Namespace, path_in_repo: str) -> list[dict]:
    lines = fetch(options, path_in_repo).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def percent(value: float, decimals: int = 0) -> str:
    return f"{100 * value:.{decimals}f}%"


def runs_with_label(metrics: dict, label: str) -> int:
    """Runs (of 60) that carry a failure label: the gate stores it as a rate over all runs."""
    return round(metrics["label_rates"].get(label, 0.0) * metrics["num_runs"])


def gate_16_claims(options: argparse.Namespace) -> list[Claim]:
    decision = read_json(options, f"{GATE_16}/gate_decision.json")
    base = read_json(options, f"{GATE_16}/metrics_current.json")
    student = read_json(options, f"{GATE_16}/metrics_candidate.json")
    checks_passed = sum(1 for check in decision["checks"] if check["passed"])

    def before_after(base_value: str, student_value: str) -> str:
        return f"{base_value} -> {student_value}"

    return [
        Claim("gate 16 decision", "promote", decision["decision"]),
        Claim("gate checks passed", "8 / 8", f"{checks_passed} / {len(decision['checks'])}"),
        Claim("valid runs per system", "60 / 60",
              f"{min(base['num_valid_runs'], student['num_valid_runs'])} / {RUNS_PER_SYSTEM}"),
        Claim("task success, pass^1 (held-out)", "10% -> 15%",
              before_after(percent(base["pass_1"]), percent(student["pass_1"]))),
        Claim("pass^k, k = 3", "10% -> 10%", before_after(percent(base["pass_k"]), percent(student["pass_k"]))),
        Claim("tool-call error rate", "44.3% -> 5.7%",
              before_after(percent(base["tool_call_error_rate"], 1), percent(student["tool_call_error_rate"], 1))),
        Claim("BFCL v3 points", "45.0 -> 73.3",
              before_after(f"{base['bfcl']['accuracy_points']:.1f}", f"{student['bfcl']['accuracy_points']:.1f}")),
        Claim("runs that invented an order or item id", "40 -> 0",
              before_after(str(runs_with_label(base, "invented_id")), str(runs_with_label(student, "invented_id")))),
        Claim("runs handed to a human too early", "11 -> 0",
              before_after(str(runs_with_label(base, "premature_transfer")),
                           str(runs_with_label(student, "premature_transfer")))),
        Claim("runs with a made-up answer", "8 -> 0",
              before_after(str(runs_with_label(base, "invented_answer")),
                           str(runs_with_label(student, "invented_answer")))),
        Claim("runs with a malformed or unknown tool call", "6 -> 0",
              before_after(str(runs_with_label(base, "malformed_args") + runs_with_label(base, "unknown_tool")),
                           str(runs_with_label(student, "malformed_args")
                               + runs_with_label(student, "unknown_tool")))),
        Claim("runs with the wrong tool", "11 -> 4",
              before_after(str(runs_with_label(base, "wrong_tool")), str(runs_with_label(student, "wrong_tool")))),
        Claim("still failing: skipped a policy step", "25 -> 31",
              before_after(str(runs_with_label(base, "skipped_policy_step")),
                           str(runs_with_label(student, "skipped_policy_step")))),
        Claim("still failing: went in circles", "8 -> 12",
              before_after(str(runs_with_label(base, "loop")), str(runs_with_label(student, "loop")))),
    ]


def hidden_split_claims(options: argparse.Namespace) -> list[Claim]:
    base_traces = read_traces(options, f"{HIDDEN}/base/traces.jsonl")
    student_traces = read_traces(options, f"{HIDDEN}/{HIDDEN_STUDENT}/traces.jsonl")
    base_summary = read_json(options, f"{HIDDEN}/base/summary.json")
    student_summary = read_json(options, f"{HIDDEN}/{HIDDEN_STUDENT}/summary.json")

    def passed_by_pair(traces: list[dict]) -> dict[tuple[str, str], bool]:
        return {(str(trace["task_id"]), str(trace.get("trial"))): bool(trace.get("passed")) for trace in traces}

    base_passed = passed_by_pair(base_traces)
    student_passed = passed_by_pair(student_traces)
    shared_pairs = set(base_passed) & set(student_passed)
    only_student = sum(1 for pair in shared_pairs if student_passed[pair] and not base_passed[pair])
    only_base = sum(1 for pair in shared_pairs if base_passed[pair] and not student_passed[pair])

    def pass_rate(passed: dict) -> float:
        return sum(passed.values()) / len(passed)

    def tool_error_rate(summary: dict) -> float:
        return summary["tool_errors"] / summary["tool_calls"]

    return [
        Claim("hidden: paired runs", "60", str(len(shared_pairs))),
        Claim("hidden: task success", "0% -> 10%",
              f"{percent(pass_rate(base_passed))} -> {percent(pass_rate(student_passed))}"),
        Claim("hidden: paired wins / losses", "6 / 0", f"{only_student} / {only_base}"),
        Claim("hidden: exact McNemar p", "0.031", f"{exact_mcnemar_p(only_student, only_base):.3f}"),
        Claim("hidden: tool-call error rate", "56% -> 5%",
              f"{percent(tool_error_rate(base_summary))} -> {percent(tool_error_rate(student_summary))}"),
    ]


def print_claims(title: str, claims: list[Claim]) -> None:
    print(f"\n{title}")
    print(f"  {'claim':44s} {'write-up':>16s} {'recomputed':>16s}")
    for claim in claims:
        verdict = "MATCH" if claim.matches else "MISMATCH"
        print(f"  {claim.what:44s} {claim.written:>16s} {claim.recomputed:>16s}  {verdict}")


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    import logging

    for client_logger in ("httpx", "httpx2", "huggingface_hub"):  # one line per downloaded file would bury the table
        logging.getLogger(client_logger).setLevel(logging.WARNING)
    sections = [
        ("Gate 16: round 16 against the base, 20 held-out tasks x 3 trials (WRITEUP.md sections 1 and 2)",
         gate_16_claims(options)),
        ("Hidden split, opened once: round 7 adapter + guards against the base, 20 tasks x 3 trials",
         hidden_split_claims(options)),
    ]
    for title, claims in sections:
        print_claims(title, claims)
    mismatches = [claim for _, claims in sections for claim in claims if not claim.matches]
    total = sum(len(claims) for _, claims in sections)
    print(f"\n{total - len(mismatches)} of {total} claims match the run files.")
    if mismatches:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="reproduce", meta=True))

"""What the control room needs to know about a round, without Streamlit: which run dirs belong to it, what the
gate said in plain words, and how two systems did on each task.

A round is one pass of the loop. Its dirs follow the naming of the loop (shipwright.loop.config.run_name):

  before   rN-baseline            the base student on the training tasks (a loop round, N >= 4, starts from
           rN-cand-train          the previous adapter on the training tasks instead)
  teacher  rN-teacher / rN-sft    the corrections the student is trained on
  gate     gate-rN                the gate's results (the sidebar button pulls them: hub_runs.pull_gate_results)
  after    r(N+1)-cand-train      this round's adapter on the same training tasks, run at the start of the next round

Each round's manifest, runs/round-N.json, says which stages ran and what the gate decided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The failure labels in words a person would use.
LABEL_WORDS = {
    "invented_id": "making up ids",
    "tool_example_copied": "copying an example from the tool list",
    "premature_transfer": "transferring to a human too early",
    "loop": "repeating the same call",
    "tool_error_ignored": "carrying on after a tool error",
    "context_overflow": "running out of room in the conversation",
    "unknown_tool": "calling tools that do not exist",
    "malformed_args": "badly formed tool calls",
    "wrong_tool": "using the wrong tool",
    "bad_arguments": "wrong arguments",
    "wrong_order": "doing steps in the wrong order",
    "skipped_policy_step": "skipping a step the policy requires",
    "invented_answer": "inventing an answer",
    "unclassified": "failures nobody could label",
}
SIGNIFICANT_RATE_CHANGE = 0.10  # a label must move this many points of runs to be worth a sentence


@dataclass
class RoundEntry:
    number: int
    outcome: str  # "rejected", "promoted" or "unfinished"
    before: str | None = None
    teacher: str | None = None
    gate: str | None = None
    after: str | None = None
    candidate: str | None = None
    note: str = ""

    @property
    def label(self) -> str:
        words = {"rejected": "the gate said no", "promoted": "the gate said yes", "unfinished": "not finished"}
        return f"Round {self.number}: {words[self.outcome]}" + (f" ({self.note})" if self.note else "")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _existing(runs_directory: Path, name: str) -> str | None:
    return name if (runs_directory / name).is_dir() else None


def discover_rounds(runs_directory: Path) -> list[RoundEntry]:
    """The real rounds that have a manifest and a before-run on disk, newest first."""
    entries = []
    for manifest_path in runs_directory.glob("round-*.json"):
        number = int(manifest_path.stem.removeprefix("round-"))
        if number == 0:
            continue  # round 0 is the offline dry run of the plumbing
        manifest = _read_json(manifest_path) or {}
        stages = manifest.get("stages") or {}

        failures_run = manifest.get("failures_run")  # a round that reuses an earlier candidate-on-train run
        is_loop_round = "candidate_train" in stages or bool(failures_run)
        before_name = failures_run or (f"r{number}-cand-train" if is_loop_round else f"r{number}-baseline")
        before = _existing(runs_directory, before_name)
        if before is None:
            continue
        teacher = _existing(runs_directory, f"r{number}-teacher") or _existing(runs_directory, f"r{number}-sft")
        gate = f"gate-r{number}" if (runs_directory / f"gate-r{number}" / "gate_decision.json").exists() else None

        decision = gate_stage_decision(stages)
        if decision is None and gate:
            decision = (_read_json(runs_directory / gate / "gate_decision.json") or {}).get("decision")
        outcome = {"reject": "rejected", "promote": "promoted"}.get(decision, "unfinished")

        entries.append(RoundEntry(
            number=number, outcome=outcome, before=before, teacher=teacher, gate=gate,
            after=_existing(runs_directory, f"r{number + 1}-cand-train"),
            candidate=f"r{number}-weight" if outcome != "unfinished" or gate else None,
            note=_round_note(stages, is_loop_round),
        ))
    return sorted(entries, key=lambda entry: -entry.number)


def gate_stage_decision(stages: dict) -> str | None:
    """"promote" or "reject" from a manifest's gate stage. The loop's manifests (since 2026-11-14) keep the whole
    decision there as an object; earlier ones keep the word."""
    decision = (stages.get("gate") or {}).get("decision")
    if isinstance(decision, dict):
        return decision.get("decision")
    return decision


def gate_stage_pass_1(stages: dict) -> dict:
    """{"current": ..., "candidate": ...} for pass^1 from a manifest's gate stage, in either manifest shape."""
    gate = stages.get("gate") or {}
    if gate.get("pass_1"):
        return gate["pass_1"]
    decision = gate.get("decision")
    if isinstance(decision, dict):
        for check in decision.get("checks") or []:
            if check.get("name") == "pass_1":
                return {"current": check.get("current"), "candidate": check.get("candidate")}
    return {}


def _round_note(stages: dict, is_loop_round: bool) -> str:
    pass_1 = gate_stage_pass_1(stages)
    if pass_1.get("current") is not None and pass_1.get("candidate") is not None:
        return f"pass^1 {pass_1['current']:.0%} to {pass_1['candidate']:.0%}"
    if is_loop_round:
        return "started from the last rejection"
    baseline = stages.get("baseline") or {}
    if baseline.get("num_runs"):
        return f"base student passed {baseline.get('num_passed')} of {baseline['num_runs']}"
    return ""


# ---------------------------------------------------------------- what the gate said, in words

@dataclass
class Finding:
    kind: str  # "good", "bad" or "neutral"
    text: str


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def what_changed(decision: dict) -> list[Finding]:
    """The gate's measured differences between the live system and the candidate, as sentences."""
    current = decision.get("metrics_current") or {}
    candidate = decision.get("metrics_candidate") or {}
    findings: list[Finding] = []

    for name, words in (("pass_1", "tasks passed on a single try"), ("pass_k", "tasks passed on every one of their tries")):
        before, after = current.get(name), candidate.get(name)
        if before is not None and after is not None and before != after:
            kind = "good" if after > before else "bad"
            findings.append(Finding(kind, f"{words.capitalize()}: {_percent(before)} to {_percent(after)}."))

    before_errors, after_errors = current.get("tool_call_error_rate"), candidate.get("tool_call_error_rate")
    if before_errors is not None and after_errors is not None and abs(after_errors - before_errors) >= 0.05:
        kind = "good" if after_errors < before_errors else "bad"
        findings.append(Finding(kind, f"Tool calls that returned an error: {_percent(before_errors)} to {_percent(after_errors)}."))

    before_rates, after_rates = current.get("label_rates") or {}, candidate.get("label_rates") or {}
    for label in sorted(set(before_rates) | set(after_rates), key=lambda name: -abs(after_rates.get(name, 0) - before_rates.get(name, 0))):
        before, after = before_rates.get(label, 0.0), after_rates.get(label, 0.0)
        if abs(after - before) < SIGNIFICANT_RATE_CHANGE:
            continue
        words = LABEL_WORDS.get(label, label.replace("_", " "))
        if after < before:
            findings.append(Finding("good", f"Runs that failed by {words}: {_percent(before)} to {_percent(after)}."))
        else:
            findings.append(Finding("bad", f"Runs that failed by {words}: {_percent(before)} to {_percent(after)}."))

    steps_before, steps_after = current.get("avg_steps"), candidate.get("avg_steps")
    if steps_before and steps_after and abs(steps_after - steps_before) / steps_before >= 0.25:
        kind = "good" if steps_after < steps_before else "bad"
        findings.append(Finding(kind, f"Steps per conversation: {steps_before:.0f} to {steps_after:.0f}."))

    bfcl_before, bfcl_after = _bfcl_points(current), _bfcl_points(candidate)
    if bfcl_before is not None and bfcl_after is not None and bfcl_before != bfcl_after:
        kind = "good" if bfcl_after >= bfcl_before else "bad"
        findings.append(Finding(kind, f"General tool-calling quiz (BFCL): {bfcl_before:.1f} to {bfcl_after:.1f} points."))
    return findings


def _bfcl_points(metrics: dict) -> float | None:
    bfcl = metrics.get("bfcl")
    if isinstance(bfcl, dict):
        return bfcl.get("accuracy_points")
    return None


# ---------------------------------------------------------------- task by task

@dataclass
class TaskChange:
    task_id: str
    before_passed: int
    before_trials: int
    after_passed: int
    after_trials: int
    change: str  # "fixed", "better", "same", "worse" or "broke"
    group: str = ""  # e.g. "0-1 writes"


CHANGE_ORDER = {"fixed": 0, "broke": 1, "better": 2, "worse": 3, "same": 4}
CHANGE_WORDS = {"fixed": "now passes", "broke": "used to pass, now fails", "better": "passes more often",
                "worse": "passes less often", "same": "no change"}


def _passes_by_task(traces: list[dict]) -> dict[str, tuple[int, int]]:
    """task id -> (passes, trials that count). An infrastructure error is not the student's result."""
    counts: dict[str, list[int]] = {}
    for trace in traces:
        if trace.get("termination_reason") == "infrastructure_error":
            continue
        task_id = str(trace.get("task_id"))
        passed, trials = counts.setdefault(task_id, [0, 0])
        counts[task_id] = [passed + (1 if (trace.get("reward") or 0) >= 1 else 0), trials + 1]
    return {task_id: (passed, trials) for task_id, (passed, trials) in counts.items()}


def compare_tasks(before_traces: list[dict], after_traces: list[dict], group_by_task: dict[str, str] | None = None) -> list[TaskChange]:
    """Every task both runs have, with how its pass count moved. Changed tasks first."""
    before, after = _passes_by_task(before_traces), _passes_by_task(after_traces)
    changes = []
    for task_id in sorted(set(before) & set(after), key=lambda task: (len(task), task)):
        before_passed, before_trials = before[task_id]
        after_passed, after_trials = after[task_id]
        before_rate, after_rate = before_passed / before_trials, after_passed / after_trials
        if after_rate == before_rate:
            change = "same"
        elif before_passed == 0:
            change = "fixed"
        elif after_passed == 0:
            change = "broke"
        else:
            change = "better" if after_rate > before_rate else "worse"
        changes.append(TaskChange(task_id, before_passed, before_trials, after_passed, after_trials, change,
                                  (group_by_task or {}).get(task_id, "")))
    return sorted(changes, key=lambda item: CHANGE_ORDER[item.change])


# ---------------------------------------------------------------- every gate, in order

CHECK_WORDS = {
    "comparable": "different tasks", "valid_runs": "too few valid runs", "pass_1": "task success fell",
    "pass_k": "consistency fell", "target": "targeted failures did not fall enough",
    "non_target": "a new kind of failure rose", "bfcl": "general tool calling fell",
    "agenttune_gate": "AgentTune's gate saw a regression",
}


@dataclass
class GateRecord:
    round_number: int
    decision: str
    current_pass_1: float | None
    candidate_pass_1: float | None
    checks_passed: int
    checks_total: int
    failed_checks: list[str]
    agenttune_approved: bool | None

    @property
    def why_refused(self) -> str:
        return "; ".join(CHECK_WORDS.get(name, name) for name in self.failed_checks)


def gate_history(runs_directory: Path) -> list[GateRecord]:
    """Every gate decision on disk (runs/gate-rN/gate_decision.json), oldest first."""
    records = []
    for decision_path in runs_directory.glob("gate-r*/gate_decision.json"):
        number = decision_path.parent.name.removeprefix("gate-r")
        if not number.isdigit():
            continue
        decision = _read_json(decision_path) or {}
        checks = decision.get("checks") or []
        by_name = {check.get("name"): check for check in checks}
        pass_1 = by_name.get("pass_1") or {}
        agenttune = by_name.get("agenttune_gate")
        records.append(GateRecord(
            round_number=int(number), decision=decision.get("decision") or "unknown",
            current_pass_1=pass_1.get("current"), candidate_pass_1=pass_1.get("candidate"),
            checks_passed=sum(1 for check in checks if check.get("passed")), checks_total=len(checks),
            failed_checks=[check.get("name") for check in checks if not check.get("passed")],
            agenttune_approved=agenttune.get("passed") if agenttune else None,
        ))
    return sorted(records, key=lambda record: record.round_number)

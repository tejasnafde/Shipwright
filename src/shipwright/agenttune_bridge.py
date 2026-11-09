"""AgentTune is the engine of DETECT, CLASSIFY, the retrain trigger and the GATE.

Our own logic (rule labels, the 3-way router, AuditKIT thresholds) adds to these AgentTune
classes; it does not replace them:

- DETECT: ``trace_audit_records`` turns a tau2 trace into AgentTune audit records (the schema
  of ``EventLog.to_audit_records()``, one record per tool call). ``detect`` runs
  ``agenttune.decide.closed_loop.failure_detector.FailureDetector.scan_audit_log`` on them.
  The failure set is runs with reward < 1 (outcome) UNION runs the detector flags.
- CLASSIFY: ``ShipwrightFailureClassifier`` is AgentTune's ``FailureClassifier`` with a fuller
  prompt (the default cuts the context to 2000 characters). It runs through
  ``agenttune.agentic.heal_loop.as_sync_classifier``. ``AGENTTUNE_ROOT_CAUSE_MAP`` maps its
  root cause to our label list and fault layer.
- RETRAIN TRIGGER: ``SelfHealLoop`` with a ``RetrainingTrigger`` (buffer + 6 conditions +
  2 safety gates) decides if a weight-fix round trains (``scripts/round_status.py``).
- GATE: ``agenttune_gate_check`` calls ``DeploymentGate.evaluate_decision`` on the paired
  held-out runs. It cannot approve by default: fewer than ``min_test_samples`` cases rejects.

Approach for the classifier subclass, the gate floor and the loop wiring ported from
ig/main (Ishank): src/shipwright/self_heal/classifier.py, gate.py, loop.py.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no price-list fetch at litellm import

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure, TrainingExample
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate
from agenttune.decide.closed_loop.failure_detector import FailureDetector
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

from shipwright import vertex
from shipwright.classify import RULE_LAYERS, compact_trace, expected_actions, route

# AgentTune FailureClassifier.ROOT_CAUSES -> (our label, fault layer). AgentTune has no
# tool-layer or environment root cause: only our rules (tool_example_copied, context_overflow),
# our judge and the infra-error check can route a run to tool_fix or no_fix.
AGENTTUNE_ROOT_CAUSE_MAP = {
    "wrong_tool": ("wrong_tool", "model"),
    "wrong_routing": ("wrong_order", "model"),  # took the wrong branch of the policy
    "incomplete_reasoning": ("skipped_policy_step", "model"),
    "hallucinated_output": ("invented_answer", "model"),
    "loop_collapse": ("loop", "model"),
}
# Detector failure types that are also one of our labels.
DETECTOR_LABELS = {"loop_collapse": "loop"}
# An identical call (same tool, same arguments) made a 3rd time is a loop, like our loop rule.
DETECTOR_MAX_REVISITS = 2
CLASSIFIER_CONTEXT_CHARS = 12000


# ---------------------------------------------------------------- DETECT

def trajectory_id(trace: dict, index: int) -> str:
    sim = trace.get("simulation_id")
    return str(sim) if sim else f"run-{index}"


def trace_audit_records(trace: dict, traj_id: str) -> list[dict]:
    """AgentTune audit records for one trace, one per agent tool call.

    Same fields as ``EventLog.to_audit_records()``: trajectory_id, stage_name (tool name plus
    sorted JSON arguments, so only an identical call counts as a revisit), tool_name,
    stage_type, state_snapshot, status, error_details. A malformed call, an unknown tool or
    a tool result with an error is status ``error``. The snapshot also keeps the task id,
    the message index and the raw model output of that turn (AGENTS.md: log raw output).
    """
    records = []
    for turn in trace.get("agent_turns") or []:
        results = {r.get("id"): r for r in turn.get("tool_results") or []}
        for tc in turn.get("parsed_tool_calls") or []:
            name = tc.get("name") or "unknown"
            args = tc.get("arguments") or {}
            stage = f"{name} {json.dumps(args, sort_keys=True, default=str)}" if args else name
            res = results.get(tc.get("id")) or {}
            error = None
            if res.get("error"):
                error = str(res.get("content") or "tool returned an error")[:500]
            elif tc.get("unknown_tool"):
                error = f"unknown tool {name}"
            elif tc.get("malformed_arguments"):
                error = "malformed arguments"
            rec = {
                "trajectory_id": traj_id,
                "stage_name": stage,
                "tool_name": name,
                "stage_type": "tool_call",
                "state_snapshot": {"arguments": args, "task_id": str(trace.get("task_id")),
                                   "message_index": turn.get("message_index"),
                                   "raw_output": (turn.get("raw_output") or "")[:2000]},
                "status": "error" if error else "ok",
            }
            if error:
                rec["error_details"] = error
            records.append(rec)
    return records


def write_audit_log(traces: list[dict], path: Path) -> Path:
    """Write the run's AgentTune audit log (one JSONL file per run dir)."""
    with open(path, "w") as f:
        for i, t in enumerate(traces):
            for rec in trace_audit_records(t, trajectory_id(t, i)):
                f.write(json.dumps(rec, default=str) + "\n")
    return path


def detect(audit_path: Path, max_revisits: int = DETECTOR_MAX_REVISITS) -> dict[str, list[Failure]]:
    """AgentTune ``FailureDetector.scan_audit_log`` over the whole file, grouped by trajectory.

    The detector resumes from ``<file>.offset``; we remove it before and after so a second
    scan of the same run dir finds the same failures.
    """
    offset = Path(str(audit_path) + ".offset")
    offset.unlink(missing_ok=True)
    try:
        found = list(FailureDetector(max_revisits=max_revisits).scan_audit_log(str(audit_path)))
    finally:
        offset.unlink(missing_ok=True)
    by_traj: dict[str, list[Failure]] = defaultdict(list)
    for f in found:
        by_traj[f.trajectory_id].append(f)
    return dict(by_traj)


def failure_summary(f: Failure) -> dict:
    return {"failure_type": f.failure_type, "stage": f.failed_stage_name[:200],
            "error": (f.error_message or None) and str(f.error_message)[:300]}


# ---------------------------------------------------------------- CLASSIFY

def _failure_classifier_cls():
    from agenttune.decide.closed_loop.failure_classifier import FailureClassifier

    return FailureClassifier


def make_classifier(model_name: str, api_base: str | None = None):
    """AgentTune ``FailureClassifier`` with our prompt context.

    ``model_name``: a ``vertex/<publisher>/<model>`` id (Vertex AI through litellm, api_base and
    a fresh gcloud token per call, see ``shipwright.vertex``) or any other litellm id."""
    base = _failure_classifier_cls()

    class ShipwrightFailureClassifier(base):
        """Same system prompt, root causes and JSON reply as AgentTune's. The user message
        keeps AgentTune's fields and adds the task goal, the expected actions, the whole
        compact run and our rule hits as hints (up to 12000 characters, not 2000)."""

        def _build_prompt(self, failure: Failure) -> list[dict[str, str]]:
            messages = super()._build_prompt(failure)
            ctx = failure.context or {}
            hints = "\n".join(f"- {h['label']} at turn {h['turn']}: {h['detail']}"
                              for h in ctx.get("rule_hits") or []) or "(none)"
            detector = "\n".join(f"- {d['failure_type']}: {d['stage']}" + (f" ({d['error']})" if d.get("error") else "")
                                 for d in ctx.get("detector_hits") or []) or "(none)"
            user = (
                f"Failure Type: {failure.failure_type}\n"
                f"Failed Stage: {failure.failed_stage_name}\n"
                f"Error Message: {failure.error_message or 'None'}\n"
                f"Task reward: {failure.judge_score}\n\n"
                f"User goal:\n{ctx.get('instruction') or '(unknown)'}\n\n"
                f"Expected agent actions:\n{ctx.get('expected_actions') or '(none)'}\n\n"
                f"Deterministic rule hits (hints, already verified):\n{hints}\n\n"
                f"Failure detector hits:\n{detector}\n\n"
                f"Run (turn numbers in brackets):\n{ctx.get('run') or ''}"
            )
            messages[-1] = {"role": "user", "content": user[:CLASSIFIER_CONTEXT_CHARS]}
            return messages

        async def _classify_single(self, failure: Failure) -> ClassifiedFailure:
            """A non-vertex id runs AgentTune's own method. A ``vertex/`` id needs a bearer
            token per call (AgentTune's method passes no api_key), so this does the same
            request and parsing with ``shipwright.vertex.route_kwargs`` and one retry on 401."""
            if not vertex.is_vertex_model(self.model_name):
                return await super()._classify_single(failure)
            import litellm

            kwargs = {"model": self.model_name, "messages": self._build_prompt(failure),
                      "response_format": {"type": "json_object"}}
            tokens = vertex.provider()
            try:
                try:
                    response = await litellm.acompletion(**vertex.route_kwargs(kwargs, tokens))
                except Exception as e:  # noqa: BLE001
                    if not vertex.is_auth_error(e):
                        raise
                    tokens.invalidate()
                    response = await litellm.acompletion(**vertex.route_kwargs(kwargs, tokens))
                parsed = json.loads(response.choices[0].message.content)
                root = parsed.get("root_cause", "")
                return ClassifiedFailure(
                    failure=failure,
                    root_cause=root if root in self.ROOT_CAUSES else "incomplete_reasoning",
                    confidence=float(parsed.get("confidence", 0.5)),
                    analysis=parsed.get("analysis", "No analysis provided"))
            except Exception as e:  # noqa: BLE001 - same fallback as AgentTune's
                return ClassifiedFailure(failure=failure, root_cause="incomplete_reasoning", confidence=0.0,
                                         analysis=f"Classification failed due to error: {type(e).__name__}: {e}")

    return ShipwrightFailureClassifier(model_name=model_name, api_base=api_base)


def build_failure(trace: dict, traj_id: str, row: dict, detector_hits: list[Failure],
                  instruction: str | None) -> Failure:
    """The AgentTune ``Failure`` for one failed run."""
    first = detector_hits[0] if detector_hits else None
    hits = row.get("rule_hits") or []
    return Failure(
        trajectory_id=traj_id,
        failure_type=first.failure_type if first else "task_failed",
        failed_stage_name=first.failed_stage_name if first else (
            f"turn {hits[0]['turn']}" if hits else "final_state"),
        error_message=first.error_message if first else None,
        judge_score=trace.get("reward"),
        context={
            "task_id": str(trace.get("task_id")),
            "instruction": instruction,
            "expected_actions": expected_actions(trace),
            "rule_hits": hits[:10],
            "detector_hits": [failure_summary(f) for f in detector_hits[:10]],
            "run": compact_trace(trace),
        },
    )


def classification_failed(c: ClassifiedFailure) -> bool:
    """AgentTune returns confidence 0 and a fixed text instead of raising."""
    return c.confidence == 0.0 and str(c.analysis).startswith("Classification failed")


def merge_classification(row: dict, c: ClassifiedFailure) -> dict:
    """Add AgentTune's verdict to a failures.jsonl row.

    Labels: rule labels + judge label + the mapped AgentTune label. Fault layer, in order:
    a tool-layer rule (deterministic, wins), then our judge's verdict, then AgentTune's
    mapped root cause. AgentTune's root cause, confidence and analysis are always kept.
    """
    row["agenttune_root_cause"] = c.root_cause
    row["agenttune_confidence"] = c.confidence
    row["agenttune_analysis"] = str(c.analysis)[:500]
    if classification_failed(c):
        row["agenttune_error"] = row.pop("agenttune_analysis")
        row["agenttune_root_cause"] = None
        return row
    label, layer = AGENTTUNE_ROOT_CAUSE_MAP[c.root_cause]
    labels = [lab for lab in row["labels"] if lab != "unclassified"]
    row["labels"] = list(dict.fromkeys(labels + [label]))
    tool_rule = row["source"] == "rules" and row["fault_layer"] == "tool"
    if not tool_rule and row["source"] != "judge":
        if row["source"] == "rules":
            row["explanation"] = f"{c.analysis} (rules: {row['explanation']})"
        else:
            row["explanation"] = str(c.analysis)
        row["fault_layer"], row["source"] = layer, "agenttune"
        row["route"] = route(layer)
    return row


def load_heal_loop():
    """``agenttune.agentic.heal_loop`` (SelfHealLoop, as_sync_classifier).

    The module itself needs only AgentTune's contracts, but the ``agenttune.agentic``
    package imports torch. Our CPU harness installs agenttune without torch, so on an
    ImportError we load the module file directly, without the package ``__init__``.
    """
    name = "agenttune.agentic.heal_loop"
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    if name in sys.modules:
        return sys.modules[name]
    import agenttune

    path = Path(agenttune.__file__).parent / "agentic" / "heal_loop.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def classify_failures(failures: list[Failure], classifier) -> list[ClassifiedFailure]:
    """Run AgentTune's classifier on all failures (concurrently, through as_sync_classifier)."""
    if not failures:
        return []
    return load_heal_loop().as_sync_classifier(classifier)(failures)


# ---------------------------------------------------------------- whole run

def classify_traces(traces: list[dict], run_dir: Path, *, instructions: dict[str, str] | None = None,
                    chat=None, judge_all: bool = False, example_values: set[str] | None = None,
                    classifier=None) -> list[dict]:
    """failures.jsonl rows for a run: AgentTune detector + our rules (+ judge) + AgentTune classifier.

    Writes ``run_dir/audit.jsonl`` (AgentTune audit records). Each row has ``detected_by``
    (``outcome``: reward < 1 or infra error; ``agenttune_detector``: FailureDetector hit) and
    ``outcome_failed``. A passed run that only the detector flags gets route ``no_fix``: it
    is evidence, not a failure to fix, and the gate's label rates skip it.
    """
    from shipwright.classify import classify_run, rule_labels

    instructions = instructions or {}
    audit = write_audit_log(traces, Path(run_dir) / "audit.jsonl")
    hits_by_traj = detect(audit)

    rows, to_classify = [], []
    for i, t in enumerate(traces):
        tid = trajectory_id(t, i)
        det = hits_by_traj.get(tid, [])
        instruction = instructions.get(str(t.get("task_id")))
        row = classify_run(t, instruction, chat, judge_all, example_values)
        if row is None and not det:
            continue
        if row is None:  # passed run, detector hit only
            hits = rule_labels(t, example_values)
            row = {"task_id": t.get("task_id"), "trial": t.get("trial"),
                   "simulation_id": t.get("simulation_id"), "reward": t.get("reward"),
                   "labels": list(dict.fromkeys(h["label"] for h in hits)), "rule_hits": hits,
                   "first_wrong_turn": None, "fault_layer": "model", "route": "no_fix",
                   "source": "agenttune_detector", "outcome_failed": False,
                   "explanation": "the run passed; AgentTune's detector flagged it"}
            row["detected_by"] = ["agenttune_detector"]
        else:
            row["outcome_failed"] = True
            row["detected_by"] = ["outcome"] + (["agenttune_detector"] if det else [])
        for f in det:
            label = DETECTOR_LABELS.get(f.failure_type)
            if label and row["route"] != "no_fix" and label not in row["labels"]:
                row["labels"] = [lab for lab in row["labels"] if lab != "unclassified"] + [label]
                if row["source"] == "unclassified":
                    row["fault_layer"], row["source"] = RULE_LAYERS[label], "agenttune_detector"
                    row["route"] = route(row["fault_layer"])
        row["agenttune_failures"] = [failure_summary(f) for f in det]
        row["agenttune_root_cause"] = None
        rows.append(row)
        if classifier is not None and row["outcome_failed"] and row["fault_layer"] != "environment":
            to_classify.append((row, build_failure(t, tid, row, det, instruction)))

    classified = classify_failures([f for _, f in to_classify], classifier) if to_classify else []
    for (row, _), c in zip(to_classify, classified):
        merge_classification(row, c)
    return rows


# ---------------------------------------------------------------- RETRAIN TRIGGER

def trigger_config(values: dict | None = None) -> TriggerConfig:
    values = dict(values or {})
    unknown = set(values) - set(TriggerConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown retrain trigger keys: {sorted(unknown)}")
    return TriggerConfig(**values)


def classified_from_row(row: dict) -> ClassifiedFailure:
    """A failures.jsonl row as AgentTune's ClassifiedFailure (classification already done)."""
    root = row.get("agenttune_root_cause") or (row.get("labels") or ["unclassified"])[0]
    failure = Failure(trajectory_id=str(row.get("simulation_id")), failure_type="task_failed",
                      failed_stage_name=f"turn {row.get('first_wrong_turn')}",
                      context={"task_id": str(row.get("task_id")), "labels": row.get("labels") or []},
                      judge_score=row.get("reward"))
    return ClassifiedFailure(failure=failure, root_cause=root,
                             confidence=float(row.get("agenttune_confidence") or 1.0),
                             analysis=str(row.get("explanation") or ""))


def correction_generator(corrections: dict[str, list[dict]]):
    """SelfHealLoop generator: one TrainingExample per weight-fix failure.

    ``chosen`` is the teacher's correct run of the same task (an SFT example from
    scripts/build_sft.py), if one exists. A failure with no correction has neither chosen
    nor rejected, so AgentTune's buffer counts it as attempted but not accepted, and its
    drop-rate gate blocks training on a buffer that is mostly uncorrected.
    """
    def generate(classified: list[ClassifiedFailure]) -> list[TrainingExample]:
        out = []
        for c in classified:
            task = c.failure.context.get("task_id")
            fix = corrections.get(task) or []
            first = next((i for i, m in enumerate(fix) if m.get("role") == "assistant"), len(fix))
            out.append(TrainingExample(
                trajectory_id=c.failure.trajectory_id, original_failure_type=c.failure.failure_type,
                root_cause=c.root_cause, prompt=fix[:first], chosen=fix[first:] or None,
                is_negative_only=not fix))
        return out

    return generate


def retrain_status(rows: list[dict], corrections: dict[str, list[dict]], config: dict | None = None,
                   baseline_rewards: list[float] | None = None, rewards: list[float] | None = None) -> dict:
    """Run AgentTune's SelfHealLoop over the weight-fix failures with a RetrainingTrigger.

    No trainer is passed (training is a separate HF Job, scripts/train_job.py), so the loop
    only fills the trigger's buffer and asks ``should_trigger()``. ``baseline_rewards``
    lock the reward-drift baseline; ``rewards`` are this run's episode rewards.
    """
    heal = load_heal_loop()
    cfg = trigger_config(config)
    if baseline_rewards:
        cfg.reward_drift_baseline_size = len(baseline_rewards)
        cfg.reward_drift_window = max(1, len(rewards or []))
    trigger = RetrainingTrigger(config=cfg)
    for r in list(baseline_rewards or []) + (list(rewards or []) if baseline_rewards else []):
        trigger.push_reward(r)
    weight = [r for r in rows if r.get("route") == "weight_fix" and r.get("outcome_failed", True)]
    loop = heal.SelfHealLoop(classifier=lambda fs: [classified_from_row(r) for r in weight],
                             generator=correction_generator(corrections), retraining_trigger=trigger)
    result = loop.run([classified_from_row(r).failure for r in weight])
    health = trigger.buffer.health()
    return {
        "should_train": bool(result["trigger_fired"]),
        "reason": result["trigger_reason"],
        "num_weight_fix_failures": len(weight),
        "num_with_correction": sum(1 for ex in result["generated"] if ex.chosen),
        "buffer_health": {k: v for k, v in health.__dict__.items()},
        "reward_drift": {"baseline_mean": trigger.reward_tracker._baseline_mean,
                         "current_mean": trigger.reward_tracker.current_mean(),
                         "drifting": trigger.reward_tracker.is_drifting(cfg.reward_drift_sigma)},
        "trigger_config": dict(cfg.__dict__),
        "agenttune_classes": ["SelfHealLoop", "RetrainingTrigger", "TrainingBuffer", "RewardDriftTracker"],
    }


# ---------------------------------------------------------------- GATE

def paired_test_set(cur_tasks: dict, cand_tasks: dict) -> tuple[list[dict], dict, dict]:
    """DeploymentGate test cases from the per-task AuditKIT results of both systems.

    One case per decided trial that both systems have on the same task (task id, trial
    slot). Each system's verdict for a case is PASS for its first n_success slots, else
    FAIL, so a system's score on the set is its pass rate on the paired runs.
    """
    cases, old, new = [], {}, {}
    for tid in sorted(set(cur_tasks) & set(cand_tasks), key=lambda x: (len(x), x)):
        a, b = cur_tasks[tid], cand_tasks[tid]
        for j in range(min(a.get("n_decided") or 0, b.get("n_decided") or 0)):
            key = f"{tid}#{j}"
            cases.append({"pipeline_id": key, "input_text": key, "expected_verdict": "PASS"})
            old[key] = "PASS" if j < (a.get("n_success") or 0) else "FAIL"
            new[key] = "PASS" if j < (b.get("n_success") or 0) else "FAIL"
    return cases, old, new


def agenttune_gate_check(current: dict, candidate: dict, min_test_samples: int,
                         task_regression_tol: float) -> dict:
    """AgentTune ``DeploymentGate.evaluate_decision`` as one gate check.

    AgentTune approves when there is no regression, and also when the test set is empty
    (both scores fall back to 0). We add the floor: fewer than ``min_test_samples`` paired
    cases fails the check, whatever AgentTune says. Both verdicts are kept.
    """
    cases, old, new = paired_test_set(current.get("per_task") or {}, candidate.get("per_task") or {})
    d = DeploymentGate(seed=0).evaluate_decision(
        cases, old.get, new.get, task_regression_tol=task_regression_tol)
    enough = len(cases) >= min_test_samples
    passed = d.approved and enough
    if not enough:
        detail = (f"AgentTune DeploymentGate: only {len(cases)} paired held-out runs, needs"
                  f" {min_test_samples} (AgentTune alone said: {d.reason})")
    else:
        detail = (f"AgentTune DeploymentGate {d.reason}: pass rate {d.task_new:.4f} vs {d.task_old:.4f}"
                  f" on {len(cases)} paired runs (tol {task_regression_tol})")
    return {"name": "agenttune_gate", "passed": passed, "detail": detail,
            "agenttune_approved": d.approved, "agenttune_reason": d.reason,
            "num_cases": len(cases), "min_test_samples": min_test_samples,
            "current": d.task_old, "candidate": d.task_new, "delta": d.task_delta}


__all__ = [
    "AGENTTUNE_ROOT_CAUSE_MAP", "agenttune_gate_check", "classify_traces", "detect", "make_classifier",
    "merge_classification", "retrain_status", "trace_audit_records", "write_audit_log",
]

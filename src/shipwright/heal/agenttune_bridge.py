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
  2 safety gates) decides if a weight-fix round trains (the loop's train stage, ``scripts/round_status.py``).
- GATE: ``agenttune_gate_check`` calls ``DeploymentGate.evaluate_decision`` on the paired
  held-out runs. It cannot approve by default: fewer than ``min_test_samples`` cases rejects.

Approach for the classifier subclass, the gate floor and the loop wiring ported from
ig/main (Ishank): src/shipwright/self_heal/classifier.py, gate.py, loop.py.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no price-list fetch at litellm import

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure, TrainingExample
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate
from agenttune.decide.closed_loop.failure_detector import FailureDetector
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

from shipwright.llm import vertex
from shipwright.heal.traces import read_jsonl
from shipwright.heal.classify import RULE_LAYERS, compact_trace, expected_actions, route

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


# AgentTune's classify_batch sends every failure at once. Vertex gpt-oss answers 429 at that rate: in round 3, 30 of 73
# classifications failed and fell back to rules. So at most this many calls run at a time, and a 429 or a timeout waits
# and tries again (``vertex.RATE_LIMIT_WAITS``).
CLASSIFIER_CONCURRENCY = 4


def make_classifier(model_name: str, api_base: str | None = None):
    """AgentTune ``FailureClassifier`` with our prompt context.

    ``model_name``: a ``vertex/<publisher>/<model>`` id (Vertex AI through litellm, api_base and
    a fresh gcloud token per call, see ``shipwright.llm.vertex``) or any other litellm id."""
    agenttune_classifier_class = _failure_classifier_cls()

    class ShipwrightFailureClassifier(agenttune_classifier_class):
        """Same system prompt, root causes and JSON reply as AgentTune's. The user message
        keeps AgentTune's fields and adds the task goal, the expected actions, the whole
        compact run and our rule hits as hints (up to 12000 characters, not 2000)."""

        def _build_prompt(self, failure: Failure) -> list[dict[str, str]]:
            messages = super()._build_prompt(failure)
            context = failure.context or {}
            rule_hits_text = "\n".join(
                f"- {hit['label']} at turn {hit['turn']}: {hit['detail']}"
                for hit in context.get("rule_hits") or []
            ) or "(none)"
            detector_hits_text = "\n".join(
                f"- {hit['failure_type']}: {hit['stage']}" + (f" ({hit['error']})" if hit.get("error") else "")
                for hit in context.get("detector_hits") or []
            ) or "(none)"
            user_message = (
                f"Failure Type: {failure.failure_type}\n"
                f"Failed Stage: {failure.failed_stage_name}\n"
                f"Error Message: {failure.error_message or 'None'}\n"
                f"Task reward: {failure.judge_score}\n\n"
                f"User goal:\n{context.get('instruction') or '(unknown)'}\n\n"
                f"Expected agent actions:\n{context.get('expected_actions') or '(none)'}\n\n"
                f"Deterministic rule hits (hints, already verified):\n{rule_hits_text}\n\n"
                f"Failure detector hits:\n{detector_hits_text}\n\n"
                f"Run (turn numbers in brackets):\n{context.get('run') or ''}"
            )
            messages[-1] = {"role": "user", "content": user_message[:CLASSIFIER_CONTEXT_CHARS]}
            return messages

        async def _ask_vertex(self, litellm, request_options: dict, token_provider, call_record: dict):
            """One classifier request: a new token after a 401, a wait and another try after a 429 or a timeout."""
            rate_limit_waits = list(vertex.RATE_LIMIT_WAITS)
            auth_already_refreshed = False
            while True:
                try:
                    return await litellm.acompletion(**vertex.route_kwargs(request_options, token_provider))
                except Exception as call_error:  # noqa: BLE001
                    if vertex.is_auth_error(call_error) and not auth_already_refreshed:
                        token_provider.invalidate()
                        auth_already_refreshed = True
                    elif (vertex.is_rate_limit(call_error) or vertex.is_timeout(call_error)) and rate_limit_waits:
                        await asyncio.sleep(rate_limit_waits.pop(0) * (1 + random.random() / 2))
                    else:
                        raise
                    call_record["retries"] += 1

        async def _classify_single(self, failure: Failure) -> ClassifiedFailure:
            """A non-vertex id runs AgentTune's own method. A ``vertex/`` id needs a bearer
            token per call (AgentTune's method passes no api_key), so this does the same
            request and parsing with ``shipwright.llm.vertex.route_kwargs`` and one retry on 401."""
            from shipwright.observability import instrument

            if not vertex.is_vertex_model(self.model_name):
                with instrument.external("llm_classifier", self.model_name, role="classifier"):
                    return await super()._classify_single(failure)
            import litellm

            request_options = {
                "model": self.model_name,
                "messages": self._build_prompt(failure),
                "response_format": {"type": "json_object"},
                "num_retries": 0,  # litellm's own retries would stack on ours
            }
            token_provider = vertex.provider()
            request_slots = self.__dict__.setdefault("_request_slots", asyncio.Semaphore(CLASSIFIER_CONCURRENCY))
            try:
                with instrument.external("llm_classifier", self.model_name, role="classifier") as call_record:
                    call_record["retries"] = 0
                    async with request_slots:
                        response = await self._ask_vertex(litellm, request_options, token_provider, call_record)
                    call_record.update(instrument.usage_fields(response))
                parsed_reply = json.loads(response.choices[0].message.content)
                root_cause = parsed_reply.get("root_cause", "")
                if root_cause not in self.ROOT_CAUSES:
                    root_cause = "incomplete_reasoning"
                return ClassifiedFailure(
                    failure=failure,
                    root_cause=root_cause,
                    confidence=float(parsed_reply.get("confidence", 0.5)),
                    analysis=parsed_reply.get("analysis", "No analysis provided"),
                )
            except Exception as classify_error:  # noqa: BLE001 - same fallback as AgentTune's
                return ClassifiedFailure(
                    failure=failure,
                    root_cause="incomplete_reasoning",
                    confidence=0.0,
                    analysis=f"Classification failed due to error: {type(classify_error).__name__}: {classify_error}",
                )

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
    module_name = "agenttune.agentic.heal_loop"
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass
    if module_name in sys.modules:
        return sys.modules[module_name]
    import agenttune

    module_path = Path(agenttune.__file__).parent / "agentic" / "heal_loop.py"
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def classify_failures(failures: list[Failure], classifier) -> list[ClassifiedFailure]:
    """Run AgentTune's classifier on all failures (concurrently, through as_sync_classifier)."""
    if not failures:
        return []
    return load_heal_loop().as_sync_classifier(classifier)(failures)


# ---------------------------------------------------------------- whole run

def classify_traces(
    traces: list[dict],
    run_dir: Path,
    *,
    instructions: dict[str, str] | None = None,
    chat=None,
    judge_all: bool = False,
    example_values: set[str] | None = None,
    classifier=None,
) -> list[dict]:
    """failures.jsonl rows for a run: AgentTune detector + our rules (+ judge) + AgentTune classifier.

    Writes ``run_dir/audit.jsonl`` (AgentTune audit records). Each row has ``detected_by``
    (``outcome``: reward < 1 or infra error; ``agenttune_detector``: FailureDetector hit) and
    ``outcome_failed``. A passed run that only the detector flags gets route ``no_fix``: it
    is evidence, not a failure to fix, and the gate's label rates skip it.
    """
    from shipwright.heal.classify import classify_run, rule_labels

    instructions = instructions or {}
    audit_path = write_audit_log(traces, Path(run_dir) / "audit.jsonl")
    detector_hits_by_trajectory = detect(audit_path)

    rows = []
    rows_to_classify = []  # (row, AgentTune Failure) pairs
    for trace_index, trace in enumerate(traces):
        trajectory = trajectory_id(trace, trace_index)
        detector_hits = detector_hits_by_trajectory.get(trajectory, [])
        instruction = instructions.get(str(trace.get("task_id")))

        row = classify_run(trace, instruction, chat, judge_all, example_values)
        if row is None and not detector_hits:
            continue  # passed, and the detector found nothing
        if row is None:  # passed run, detector hit only
            rule_hits = rule_labels(trace, example_values)
            row = {
                "task_id": trace.get("task_id"),
                "trial": trace.get("trial"),
                "simulation_id": trace.get("simulation_id"),
                "reward": trace.get("reward"),
                "labels": list(dict.fromkeys(hit["label"] for hit in rule_hits)),
                "rule_hits": rule_hits,
                "first_wrong_turn": None,
                "fault_layer": "model",
                "route": "no_fix",
                "source": "agenttune_detector",
                "outcome_failed": False,
                "explanation": "the run passed; AgentTune's detector flagged it",
                "detected_by": ["agenttune_detector"],
            }
        else:
            row["outcome_failed"] = True
            row["detected_by"] = ["outcome"] + (["agenttune_detector"] if detector_hits else [])

        for detector_hit in detector_hits:
            detector_label = DETECTOR_LABELS.get(detector_hit.failure_type)
            if detector_label and row["route"] != "no_fix" and detector_label not in row["labels"]:
                row["labels"] = [label for label in row["labels"] if label != "unclassified"] + [detector_label]
                if row["source"] == "unclassified":
                    row["fault_layer"] = RULE_LAYERS[detector_label]
                    row["source"] = "agenttune_detector"
                    row["route"] = route(row["fault_layer"])
        row["agenttune_failures"] = [failure_summary(hit) for hit in detector_hits]
        row["agenttune_root_cause"] = None
        rows.append(row)

        if classifier is not None and row["outcome_failed"] and row["fault_layer"] != "environment":
            failure = build_failure(trace, trajectory, row, detector_hits, instruction)
            rows_to_classify.append((row, failure))

    if rows_to_classify:
        classified_failures = classify_failures([failure for _, failure in rows_to_classify], classifier)
    else:
        classified_failures = []
    for (row, _), classified in zip(rows_to_classify, classified_failures):
        merge_classification(row, classified)
    return rows


# ---------------------------------------------------------------- RETRAIN TRIGGER

def trigger_config(values: dict | None = None) -> TriggerConfig:
    config_values = dict(values or {})
    unknown_keys = set(config_values) - set(TriggerConfig.__dataclass_fields__)
    if unknown_keys:
        raise ValueError(f"unknown retrain trigger keys: {sorted(unknown_keys)}")
    return TriggerConfig(**config_values)


def classified_from_row(row: dict) -> ClassifiedFailure:
    """A failures.jsonl row as AgentTune's ClassifiedFailure (classification already done)."""
    root_cause = row.get("agenttune_root_cause") or (row.get("labels") or ["unclassified"])[0]
    failure = Failure(
        trajectory_id=str(row.get("simulation_id")),
        failure_type="task_failed",
        failed_stage_name=f"turn {row.get('first_wrong_turn')}",
        context={"task_id": str(row.get("task_id")), "labels": row.get("labels") or []},
        judge_score=row.get("reward"),
    )
    return ClassifiedFailure(
        failure=failure,
        root_cause=root_cause,
        confidence=float(row.get("agenttune_confidence") or 1.0),
        analysis=str(row.get("explanation") or ""),
    )


def correction_generator(corrections: dict[str, list[dict]]):
    """SelfHealLoop generator: one TrainingExample per weight-fix failure.

    ``chosen`` is the teacher's correct run of the same task (an SFT example from
    scripts/build_sft.py), if one exists. A failure with no correction has neither chosen
    nor rejected, so AgentTune's buffer counts it as attempted but not accepted, and its
    drop-rate gate blocks training on a buffer that is mostly uncorrected.
    """
    def generate(classified_failures: list[ClassifiedFailure]) -> list[TrainingExample]:
        training_examples = []
        for classified in classified_failures:
            task_id = classified.failure.context.get("task_id")
            correction_messages = corrections.get(task_id) or []
            first_assistant_index = next(
                (index for index, message in enumerate(correction_messages) if message.get("role") == "assistant"),
                len(correction_messages),
            )
            training_examples.append(TrainingExample(
                trajectory_id=classified.failure.trajectory_id,
                original_failure_type=classified.failure.failure_type,
                root_cause=classified.root_cause,
                prompt=correction_messages[:first_assistant_index],
                chosen=correction_messages[first_assistant_index:] or None,
                is_negative_only=not correction_messages,
            ))
        return training_examples

    return generate


def retrain_status(
    rows: list[dict],
    corrections: dict[str, list[dict]],
    config: dict | None = None,
    baseline_rewards: list[float] | None = None,
    rewards: list[float] | None = None,
) -> dict:
    """Run AgentTune's SelfHealLoop over the weight-fix failures with a RetrainingTrigger.

    No trainer is passed (training is a separate HF Job, scripts/train_job.py), so the loop
    only fills the trigger's buffer and asks ``should_trigger()``. ``baseline_rewards``
    lock the reward-drift baseline; ``rewards`` are this run's episode rewards.
    """
    heal_loop_module = load_heal_loop()
    trigger_settings = trigger_config(config)
    if baseline_rewards:
        trigger_settings.reward_drift_baseline_size = len(baseline_rewards)
        trigger_settings.reward_drift_window = max(1, len(rewards or []))
    trigger = RetrainingTrigger(config=trigger_settings)

    if baseline_rewards:
        for reward in list(baseline_rewards) + list(rewards or []):
            trigger.push_reward(reward)

    weight_fix_rows = [row for row in rows if row.get("route") == "weight_fix" and row.get("outcome_failed", True)]
    heal_loop = heal_loop_module.SelfHealLoop(
        classifier=lambda failures: [classified_from_row(row) for row in weight_fix_rows],
        generator=correction_generator(corrections),
        retraining_trigger=trigger,
    )
    result = heal_loop.run([classified_from_row(row).failure for row in weight_fix_rows])
    buffer_health = trigger.buffer.health()
    return {
        "should_train": bool(result["trigger_fired"]),
        "reason": result["trigger_reason"],
        "num_weight_fix_failures": len(weight_fix_rows),
        "num_with_correction": sum(1 for example in result["generated"] if example.chosen),
        "buffer_health": dict(buffer_health.__dict__),
        "reward_drift": {
            "baseline_mean": trigger.reward_tracker._baseline_mean,
            "current_mean": trigger.reward_tracker.current_mean(),
            "drifting": trigger.reward_tracker.is_drifting(trigger_settings.reward_drift_sigma),
        },
        "trigger_config": dict(trigger_settings.__dict__),
        "agenttune_classes": ["SelfHealLoop", "RetrainingTrigger", "TrainingBuffer", "RewardDriftTracker"],
    }


# ---------------------------------------------------------------- GATE

def paired_test_set(current_tasks: dict, candidate_tasks: dict) -> tuple[list[dict], dict, dict]:
    """DeploymentGate test cases from the per-task AuditKIT results of both systems.

    One case per decided trial that both systems have on the same task (task id, trial
    slot). Each system's verdict for a case is PASS for its first n_success slots, else
    FAIL, so a system's score on the set is its pass rate on the paired runs.
    """
    test_cases = []
    current_verdicts = {}
    candidate_verdicts = {}
    shared_task_ids = sorted(set(current_tasks) & set(candidate_tasks), key=lambda task_id: (len(task_id), task_id))
    for task_id in shared_task_ids:
        current_task = current_tasks[task_id]
        candidate_task = candidate_tasks[task_id]
        paired_trial_count = min(current_task.get("n_decided") or 0, candidate_task.get("n_decided") or 0)
        for trial_slot in range(paired_trial_count):
            case_key = f"{task_id}#{trial_slot}"
            test_cases.append({"pipeline_id": case_key, "input_text": case_key, "expected_verdict": "PASS"})
            current_passes = trial_slot < (current_task.get("n_success") or 0)
            candidate_passes = trial_slot < (candidate_task.get("n_success") or 0)
            current_verdicts[case_key] = "PASS" if current_passes else "FAIL"
            candidate_verdicts[case_key] = "PASS" if candidate_passes else "FAIL"
    return test_cases, current_verdicts, candidate_verdicts


def agenttune_gate_check(
    current: dict,
    candidate: dict,
    min_test_samples: int,
    task_regression_tol: float,
) -> dict:
    """AgentTune ``DeploymentGate.evaluate_decision`` as one gate check.

    AgentTune approves when there is no regression, and also when the test set is empty
    (both scores fall back to 0). We add the floor: fewer than ``min_test_samples`` paired
    cases fails the check, whatever AgentTune says. Both verdicts are kept.
    """
    test_cases, current_verdicts, candidate_verdicts = paired_test_set(
        current.get("per_task") or {}, candidate.get("per_task") or {}
    )
    decision = DeploymentGate(seed=0).evaluate_decision(
        test_cases, current_verdicts.get, candidate_verdicts.get, task_regression_tol=task_regression_tol
    )
    has_enough_cases = len(test_cases) >= min_test_samples
    passed = decision.approved and has_enough_cases
    if not has_enough_cases:
        detail = (
            f"AgentTune DeploymentGate: only {len(test_cases)} paired held-out runs, needs"
            f" {min_test_samples} (AgentTune alone said: {decision.reason})"
        )
    else:
        detail = (
            f"AgentTune DeploymentGate {decision.reason}: pass rate {decision.task_new:.4f}"
            f" vs {decision.task_old:.4f} on {len(test_cases)} paired runs (tol {task_regression_tol})"
        )
    return {
        "name": "agenttune_gate",
        "passed": passed,
        "detail": detail,
        "agenttune_approved": decision.approved,
        "agenttune_reason": decision.reason,
        "num_cases": len(test_cases),
        "min_test_samples": min_test_samples,
        "current": decision.task_old,
        "candidate": decision.task_new,
        "delta": decision.task_delta,
    }



def retrain_status_for_run(run_directory: Path, sft_path: Path | None, heal_config_path: Path,
                           reward_baseline: Path | None = None) -> dict:
    """``retrain_status`` for a classified run dir (failures.jsonl) and an SFT file: a failure has a correction when the
    SFT data holds an example of its task. Writes and returns ``round_status.json``. The train stage of the loop and
    scripts/round_status.py both call this."""
    import yaml

    heal_config = yaml.safe_load(Path(heal_config_path).read_text())
    corrections: dict[str, list[dict]] = {}
    if sft_path is not None and Path(sft_path).exists():
        for example in read_jsonl(Path(sft_path)):
            corrections.setdefault(str(example.get("task_id")), example["messages"])
    baseline_rewards = None
    rewards = None
    if reward_baseline is not None:
        baseline_rewards = [float(t["reward"]) for t in read_jsonl(Path(reward_baseline) / "traces.jsonl") if t.get("reward") is not None]
        rewards = [float(t["reward"]) for t in read_jsonl(Path(run_directory) / "traces.jsonl") if t.get("reward") is not None]
    status = retrain_status(read_jsonl(Path(run_directory) / "failures.jsonl"), corrections,
                            heal_config.get("retrain_trigger"), baseline_rewards=baseline_rewards, rewards=rewards)
    status.update(run_dir=str(run_directory), sft=str(sft_path) if sft_path else None,
                  config_status=heal_config.get("status"))
    (Path(run_directory) / "round_status.json").write_text(json.dumps(status, indent=2, default=str))
    return status


__all__ = [
    "AGENTTUNE_ROOT_CAUSE_MAP", "agenttune_gate_check", "classify_traces", "detect", "make_classifier",
    "merge_classification", "retrain_status", "retrain_status_for_run", "trace_audit_records", "write_audit_log",
]

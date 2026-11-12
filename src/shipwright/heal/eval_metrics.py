"""Gate metrics for one run dir (traces.jsonl + failures.jsonl), computed with AuditKIT.

AuditKIT (``auditkit.agent_eval``) does the scoring:

- Each trace becomes an ``AgentEpisode`` with ``episode_from_openai_messages``. Our traces
  keep tau2's message format, so ``trace_to_openai_messages`` is a small adapter
  (tool calls get the OpenAI ``function`` shape, tool results get ``tool_call_id``).
- The outcome oracle is ``FinalStateAssertion("tau2_pass", equals=True)``. The state is
  tau2's own verified result (its evaluator checks the final DB state, the required
  actions and the required replies). An infra-error run has no state, so AuditKIT scores
  it ``unknown``: it is never counted as a pass or a fail.
- pass^1 and pass^k: ``AgentEvalRunner`` in recorded mode with ``trials=k`` and
  ``reliability_k=k``. Per task, ``all_k`` is AuditKIT's ``pass_hat_k`` = C(c, k) / C(n, k)
  over decided trials, which is tau2's pass^k. ``reset_confirmed=True`` because tau2
  starts every trial from a fresh environment.
- Trajectory diagnostics: ``tool_call_f1`` (``ToolCallF1`` against the task's expected
  assistant actions from ``reward_info.action_checks``), ``tool_call_validity`` (against
  the tool schemas in ``prompts.json``) and ``redundant_tool_calls``, one case per run.

Our own counts (from ``shipwright.heal.traces`` and ``failures.jsonl``): per-label failure
rates, tool-call error rate, average agent steps.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from shipwright.heal.traces import INFRA_ERROR

PASS_KEY = "tau2_pass"
DIAGNOSTIC_SCORERS = ["tool_call_f1", "tool_call_validity", "redundant_tool_calls"]


# ---------------------------------------------------------------- loading

def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run_dir: Path) -> dict:
    """traces, failures, run_config and tool schemas of a run dir."""
    run_dir = Path(run_dir)
    failures_path = run_dir / "failures.jsonl"
    if not failures_path.exists():
        raise FileNotFoundError(f"{failures_path} missing: run scripts/classify.py on the run first")

    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text())
    else:
        run_config = {}

    tools: list[dict] = []
    prompts_path = run_dir / "prompts.json"
    if prompts_path.exists():
        prompts_by_sha = json.loads(prompts_path.read_text())
        if prompts_by_sha:
            tools = next(iter(prompts_by_sha.values())).get("tools") or []
    return {
        "run_dir": str(run_dir),
        "traces": _jsonl(run_dir / "traces.jsonl"),
        "failures": _jsonl(failures_path),
        "run_config": run_config,
        "tools": tools,
    }


# ---------------------------------------------------------------- AuditKIT adapter

def is_infra_error(trace: dict) -> bool:
    return trace.get("termination_reason") == INFRA_ERROR


def trace_to_openai_messages(messages: list[dict]) -> list[dict]:
    """tau2 messages (as saved in traces.jsonl) -> OpenAI chat messages for AuditKIT."""
    openai_messages: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            openai_message: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                openai_message["tool_calls"] = [
                    {
                        "id": tool_call.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name"),
                            "arguments": json.dumps(tool_call.get("arguments") or {}),
                        },
                    }
                    for tool_call in tool_calls
                ]
            openai_messages.append(openai_message)
        elif role == "tool":
            openai_messages.append(
                {"role": "tool", "tool_call_id": message.get("id"), "content": message.get("content") or ""}
            )
        elif role in ("user", "system"):
            openai_messages.append({"role": role, "content": message.get("content") or ""})
    return openai_messages


def final_answer(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content") and not message.get("tool_calls"):
            return message["content"]
    return None


def reference_turns(trace: dict) -> list[list[dict]] | None:
    """The task's expected assistant actions (tau2 ``action_checks``), one call per turn."""
    action_checks = (trace.get("reward_info") or {}).get("action_checks")
    if action_checks is None:
        return None
    turns = []
    for action_check in action_checks:
        action = action_check.get("action") or {}
        if action.get("requestor", "assistant") == "assistant" and action.get("name"):
            turns.append([{"name": action["name"], "arguments": action.get("arguments") or {}}])
    return turns


def episode_from_trace(trace: dict, case_id: str | None = None):
    """One AuditKIT AgentEpisode for one trace. Infra-error runs get no final state."""
    from auditkit.agent_eval import episode_from_openai_messages

    messages = trace.get("messages") or []
    if case_id is None:
        case_id = trace.get("task_id")
    episode = episode_from_openai_messages(
        trace_to_openai_messages(messages),
        case_id=str(case_id),
        source_id=str(trace.get("simulation_id")),
        final_answer=final_answer(messages),
        source="shipwright_tau2",
    )
    episode.trial_id = str(trace.get("simulation_id"))
    episode.source_reward = trace.get("reward")
    if not is_infra_error(trace) and trace.get("reward") is not None:
        episode.final_state = {PASS_KEY: bool(trace.get("passed")), "tau2_reward": trace.get("reward")}
        episode.coverage["final_state"] = "observed"
    return episode


def _case(case_id: str, task: str, tools: list[dict] | None, reference_turn_list: list | None):
    from auditkit.agent_eval import AgentCase, FinalStateAssertion

    return AgentCase(
        id=case_id,
        task=task,
        category="tau2",
        allowed_tools=tools or None,
        reference_turns=reference_turn_list,
        outcome=FinalStateAssertion(PASS_KEY, equals=True),
    )


def reliability_rows(traces: list[dict], k: int) -> list:
    """AuditKIT rows, one per task, aggregating its trials (recorded mode)."""
    from auditkit.agent_eval import AgentEvalRunner, AgentEvalSpec

    traces_by_task_id: dict[str, list[dict]] = {}
    for trace in traces:
        traces_by_task_id.setdefault(str(trace.get("task_id")), []).append(trace)

    trials_per_task = max((len(task_traces) for task_traces in traces_by_task_id.values()), default=1)
    episodes = [
        episode_from_trace(trace) for task_traces in traces_by_task_id.values() for trace in task_traces
    ]
    cases = [_case(task_id, f"tau2 task {task_id}", None, None) for task_id in traces_by_task_id]
    spec = AgentEvalSpec(
        cases=cases,
        mode="recorded",
        episodes=episodes,
        trials=max(trials_per_task, 1),
        reliability_k=k,
        reset_confirmed=True,
        scorers=[],
        name="shipwright-gate-reliability",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AgentEvalRunner().run(spec).rows


def diagnostic_rows(traces: list[dict], tools: list[dict] | None) -> list:
    """AuditKIT rows, one per run, with the trajectory diagnostics."""
    from auditkit.agent_eval import AgentEvalRunner, AgentEvalSpec

    valid_traces = [trace for trace in traces if not is_infra_error(trace)]
    episodes, cases = [], []
    for trace in valid_traces:
        case_id = str(trace.get("simulation_id"))
        episodes.append(episode_from_trace(trace, case_id=case_id))
        cases.append(_case(case_id, f"tau2 task {trace.get('task_id')}", tools, reference_turns(trace)))
    if not cases:
        return []

    spec = AgentEvalSpec(
        cases=cases,
        mode="recorded",
        episodes=episodes,
        trials=1,
        scorers=DIAGNOSTIC_SCORERS,
        name="shipwright-gate-diagnostics",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AgentEvalRunner().run(spec).rows


# ---------------------------------------------------------------- metrics

def _mean(values: list[float]) -> float | None:
    if values:
        return sum(values) / len(values)
    return None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (default: 95%).

    More reliable than the normal approximation at small n or p near 0/1 - exactly the
    regime our held-out set (20 tasks) and gate samples sit in. A result like "20/20" is
    not the same claim as "the true rate is 100%"; this is the same framing a different
    Track 3 team used for their own 30-task held-out claim ("consistent with a true rate
    of 88-100%", github.com/Timidan/lexsi-hack) rather than reporting a bare percentage
    the sample size cannot support.
    """
    if n == 0:
        return None
    observed_rate = successes / n
    denominator = 1 + z * z / n
    center = (observed_rate + z * z / (2 * n)) / denominator
    margin = (z * ((observed_rate * (1 - observed_rate) / n + z * z / (4 * n * n)) ** 0.5)) / denominator
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def pass_metrics(rows: list, k: int) -> dict:
    """pass^1 and pass^k from AuditKIT rows (one per task).

    pass^1: mean over tasks of successes / decided trials. ``pass_1_ci`` is a 95% Wilson
    interval on the pooled (successes, decided trials) count across all tasks - the
    standard pass@1 convention of treating every decided trial as one Bernoulli draw, same
    as pass@k literature generally does, not a claim that trials within one task are fully
    independent of task difficulty.
    pass^k: mean over tasks of AuditKIT ``all_k``, only for tasks with at least k decided
    trials (AuditKIT would silently lower k otherwise). Other tasks are counted, not scored.
    """
    task_pass_1_rates = []
    task_pass_k_values = []
    total_successes = 0
    total_decided = 0
    per_task = {}
    for row in rows:
        reliability = row.reliability
        if reliability:
            success_count = reliability["n_success"]
            decided_count = reliability["n_decided"]
            has_k_decided_trials = reliability.get("k") == k and decided_count >= k
            pass_k = reliability.get("all_k") if has_k_decided_trials else None
        else:  # a single trial: no reliability dict
            verdict = (row.outcome or {}).get("verdict")
            success_count = int(verdict == "success")
            decided_count = int(verdict in ("success", "failure"))
            pass_k = float(success_count) if k == 1 and decided_count == 1 else None

        if decided_count:
            task_pass_1_rates.append(success_count / decided_count)
            total_successes += success_count
            total_decided += decided_count
        if pass_k is not None:
            task_pass_k_values.append(pass_k)
        per_task[row.case_id] = {"n_success": success_count, "n_decided": decided_count, "pass_k": pass_k}

    return {
        "pass_1": _mean(task_pass_1_rates),
        "pass_1_ci": wilson_interval(total_successes, total_decided),
        "pass_k": _mean(task_pass_k_values),
        "num_tasks_scored": len(task_pass_1_rates),
        "num_tasks_complete": len(task_pass_k_values),
        "per_task": per_task,
    }


def diagnostic_metrics(rows: list) -> dict:
    values_by_name: dict[str, list[float]] = {}
    ineligible_counts: Counter = Counter()
    for row in rows:
        for name, diagnostic in (row.diagnostics or {}).items():
            if isinstance(diagnostic, dict) and isinstance(diagnostic.get("value"), (int, float)):
                values_by_name.setdefault(name, []).append(float(diagnostic["value"]))
        for name, reason in (row.ineligible or {}).items():
            ineligible_counts[f"{name}: {reason}"] += 1

    reported_names = (
        "tool_call_f1", "tool_call_precision", "tool_call_recall", "tool_call_exact",
        "tool_call_validity", "redundant_tool_calls",
    )
    metrics: dict[str, Any] = {name: _mean(values_by_name.get(name, [])) for name in reported_names}
    metrics["ineligible"] = dict(ineligible_counts)
    return metrics


def label_rates(failures: list[dict], valid_ids: set[str], num_valid: int) -> dict[str, float]:
    """Share of valid runs whose failure row carries each label."""
    run_counts_by_label: Counter = Counter()
    for failure in failures:
        if str(failure.get("simulation_id")) not in valid_ids:
            continue
        run_counts_by_label.update(set(failure.get("labels") or []))
    if not num_valid:
        return {}
    return {label: run_count / num_valid for label, run_count in run_counts_by_label.most_common()}


def tool_counts(traces: list[dict]) -> dict:
    """Tool calls, failed tool calls (malformed, unknown tool, or a tool error result),
    prompted-JSON parse failures and agent turns, over the given traces."""
    call_count = error_count = parse_failure_count = turn_count = 0
    for trace in traces:
        for turn in trace.get("agent_turns") or []:
            turn_count += 1
            parse_failure_count += int(turn.get("tool_call_parse") == "parse_failed")
            failed_call_ids = {
                result.get("id") for result in turn.get("tool_results") or [] if result.get("error")
            }
            for tool_call in turn.get("parsed_tool_calls") or []:
                call_count += 1
                is_failed_call = (
                    tool_call.get("malformed_arguments")
                    or tool_call.get("unknown_tool")
                    or tool_call.get("id") in failed_call_ids
                )
                error_count += int(bool(is_failed_call))
    return {
        "tool_calls": call_count,
        "errors": error_count,
        "parse_failures": parse_failure_count,
        "agent_turns": turn_count,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator:
        return numerator / denominator
    return None


def slice_metrics(run_config: dict, per_task: dict) -> dict | None:
    """pass^1 and pass^k per task slice (``shipwright.heal.slices``); None when tau2 is not installed."""
    from shipwright.heal import slices

    try:
        slice_by_task = slices.write_slice_by_task(run_config.get("domain") or "retail", run_config.get("task_split") or "test")
    except ImportError:
        return None
    return slices.slice_pass_metrics(per_task, slice_by_task)


def compute_metrics(run: dict, k: int, system: str | None = None) -> dict:
    """The gate's metric dict for one run (see module docstring)."""
    traces = run["traces"]
    valid_traces = [trace for trace in traces if not is_infra_error(trace)]
    valid_ids = {str(trace.get("simulation_id")) for trace in valid_traces}
    counts = tool_counts(valid_traces)
    pass_scores = pass_metrics(reliability_rows(traces, k), k)
    run_config = run.get("run_config") or {}
    trials_per_task = Counter(str(trace.get("task_id")) for trace in traces)
    failed_valid_count = sum(1 for trace in valid_traces if not trace.get("passed"))
    # Rows for passed runs that only AgentTune's detector flagged are evidence, not failures.
    failures = [failure for failure in run["failures"] if failure.get("outcome_failed") is not False]
    failed_run_labels = [
        {"simulation_id": str(failure.get("simulation_id")), "labels": failure.get("labels") or []}
        for failure in failures
        if str(failure.get("simulation_id")) in valid_ids
    ]
    return {
        "system": system,
        "run_dir": run.get("run_dir"),
        "slices": slice_metrics(run_config, pass_scores["per_task"]),
        "agent_model": run_config.get("agent_model"),
        "task_split": run_config.get("task_split"),
        "active_tool_patches": run_config.get("active_tool_patches") or [],
        "k": k,
        "task_ids": sorted(trials_per_task, key=lambda task_id: (len(task_id), task_id)),
        "num_tasks": len(trials_per_task),
        "num_runs": len(traces),
        "num_valid_runs": len(valid_traces),
        "num_infra_errors": len(traces) - len(valid_traces),
        "pass_1": pass_scores["pass_1"],
        "pass_1_ci": pass_scores["pass_1_ci"],
        "pass_k": pass_scores["pass_k"],
        "num_tasks_scored": pass_scores["num_tasks_scored"],
        "num_tasks_complete": pass_scores["num_tasks_complete"],
        "per_task": pass_scores["per_task"],
        "failure_rate": _ratio(failed_valid_count, len(valid_traces)),
        "label_rates": label_rates(failures, valid_ids, len(valid_traces)),
        "tool_calls": counts["tool_calls"],
        "tool_call_errors": counts["errors"],
        "tool_call_error_rate": _ratio(counts["errors"], counts["tool_calls"]),
        "tool_call_parse_failures": counts["parse_failures"],
        "avg_steps": _ratio(counts["agent_turns"], len(valid_traces)),
        "auditkit": diagnostic_metrics(diagnostic_rows(traces, run.get("tools"))),
        # Labels of each failed valid run, for the gate's target / non-target checks.
        "failed_run_labels": failed_run_labels,
    }

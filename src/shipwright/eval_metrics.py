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

Our own counts (from ``shipwright.traces`` and ``failures.jsonl``): per-label failure
rates, tool-call error rate, average agent steps.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from shipwright.traces import INFRA_ERROR

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
    cfg_path = run_dir / "run_config.json"
    prompts_path = run_dir / "prompts.json"
    tools: list[dict] = []
    if prompts_path.exists():
        prompts = json.loads(prompts_path.read_text())
        if prompts:
            tools = next(iter(prompts.values())).get("tools") or []
    return {
        "run_dir": str(run_dir),
        "traces": _jsonl(run_dir / "traces.jsonl"),
        "failures": _jsonl(failures_path),
        "run_config": json.loads(cfg_path.read_text()) if cfg_path.exists() else {},
        "tools": tools,
    }


# ---------------------------------------------------------------- AuditKIT adapter

def is_infra_error(trace: dict) -> bool:
    return trace.get("termination_reason") == INFRA_ERROR


def trace_to_openai_messages(messages: list[dict]) -> list[dict]:
    """tau2 messages (as saved in traces.jsonl) -> OpenAI chat messages for AuditKIT."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content")}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [{
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {"name": tc.get("name"),
                                 "arguments": json.dumps(tc.get("arguments") or {})},
                } for tc in calls]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id"), "content": m.get("content") or ""})
        elif role in ("user", "system"):
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def final_answer(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
            return m["content"]
    return None


def reference_turns(trace: dict) -> list[list[dict]] | None:
    """The task's expected assistant actions (tau2 ``action_checks``), one call per turn."""
    checks = (trace.get("reward_info") or {}).get("action_checks")
    if checks is None:
        return None
    turns = []
    for c in checks:
        a = c.get("action") or {}
        if a.get("requestor", "assistant") == "assistant" and a.get("name"):
            turns.append([{"name": a["name"], "arguments": a.get("arguments") or {}}])
    return turns


def episode_from_trace(trace: dict, case_id: str | None = None):
    """One AuditKIT AgentEpisode for one trace. Infra-error runs get no final state."""
    from auditkit.agent_eval import episode_from_openai_messages

    messages = trace.get("messages") or []
    ep = episode_from_openai_messages(
        trace_to_openai_messages(messages),
        case_id=str(case_id if case_id is not None else trace.get("task_id")),
        source_id=str(trace.get("simulation_id")),
        final_answer=final_answer(messages),
        source="shipwright_tau2",
    )
    ep.trial_id = str(trace.get("simulation_id"))
    ep.source_reward = trace.get("reward")
    if not is_infra_error(trace) and trace.get("reward") is not None:
        ep.final_state = {PASS_KEY: bool(trace.get("passed")), "tau2_reward": trace.get("reward")}
        ep.coverage["final_state"] = "observed"
    return ep


def _case(case_id: str, task: str, tools: list[dict] | None, ref: list | None):
    from auditkit.agent_eval import AgentCase, FinalStateAssertion

    return AgentCase(id=case_id, task=task, category="tau2",
                     allowed_tools=tools or None, reference_turns=ref,
                     outcome=FinalStateAssertion(PASS_KEY, equals=True))


def reliability_rows(traces: list[dict], k: int) -> list:
    """AuditKIT rows, one per task, aggregating its trials (recorded mode)."""
    from auditkit.agent_eval import AgentEvalRunner, AgentEvalSpec

    by_task: dict[str, list[dict]] = {}
    for t in traces:
        by_task.setdefault(str(t.get("task_id")), []).append(t)
    trials = max((len(v) for v in by_task.values()), default=1)
    episodes = [episode_from_trace(t) for ts in by_task.values() for t in ts]
    cases = [_case(tid, f"tau2 task {tid}", None, None) for tid in by_task]
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes, trials=max(trials, 1),
                         reliability_k=k, reset_confirmed=True, scorers=[], name="shipwright-gate-reliability")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AgentEvalRunner().run(spec).rows


def diagnostic_rows(traces: list[dict], tools: list[dict] | None) -> list:
    """AuditKIT rows, one per run, with the trajectory diagnostics."""
    from auditkit.agent_eval import AgentEvalRunner, AgentEvalSpec

    valid = [t for t in traces if not is_infra_error(t)]
    episodes, cases = [], []
    for t in valid:
        cid = str(t.get("simulation_id"))
        episodes.append(episode_from_trace(t, case_id=cid))
        cases.append(_case(cid, f"tau2 task {t.get('task_id')}", tools, reference_turns(t)))
    if not cases:
        return []
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes, trials=1,
                         scorers=DIAGNOSTIC_SCORERS, name="shipwright-gate-diagnostics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AgentEvalRunner().run(spec).rows


# ---------------------------------------------------------------- metrics

def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pass_metrics(rows: list, k: int) -> dict:
    """pass^1 and pass^k from AuditKIT rows (one per task).

    pass^1: mean over tasks of successes / decided trials.
    pass^k: mean over tasks of AuditKIT ``all_k``, only for tasks with at least k decided
    trials (AuditKIT would silently lower k otherwise). Other tasks are counted, not scored.
    """
    p1, pk = [], []
    per_task = {}
    for row in rows:
        rel = row.reliability
        if rel:
            n_s, n_d = rel["n_success"], rel["n_decided"]
            all_k = rel.get("all_k") if rel.get("k") == k and n_d >= k else None
        else:  # a single trial: no reliability dict
            verdict = (row.outcome or {}).get("verdict")
            n_s, n_d = int(verdict == "success"), int(verdict in ("success", "failure"))
            all_k = float(n_s) if k == 1 and n_d == 1 else None
        if n_d:
            p1.append(n_s / n_d)
        if all_k is not None:
            pk.append(all_k)
        per_task[row.case_id] = {"n_success": n_s, "n_decided": n_d, "pass_k": all_k}
    return {"pass_1": _mean(p1), "pass_k": _mean(pk), "num_tasks_scored": len(p1),
            "num_tasks_complete": len(pk), "per_task": per_task}


def diagnostic_metrics(rows: list) -> dict:
    vals: dict[str, list[float]] = {}
    ineligible: Counter = Counter()
    for row in rows:
        for name, d in (row.diagnostics or {}).items():
            if isinstance(d, dict) and isinstance(d.get("value"), (int, float)):
                vals.setdefault(name, []).append(float(d["value"]))
        for name, why in (row.ineligible or {}).items():
            ineligible[f"{name}: {why}"] += 1
    keep = ("tool_call_f1", "tool_call_precision", "tool_call_recall", "tool_call_exact",
            "tool_call_validity", "redundant_tool_calls")
    out: dict[str, Any] = {name: _mean(vals.get(name, [])) for name in keep}
    out["ineligible"] = dict(ineligible)
    return out


def label_rates(failures: list[dict], valid_ids: set[str], num_valid: int) -> dict[str, float]:
    """Share of valid runs whose failure row carries each label."""
    counts: Counter = Counter()
    for row in failures:
        if str(row.get("simulation_id")) not in valid_ids:
            continue
        counts.update(set(row.get("labels") or []))
    return {lab: n / num_valid for lab, n in counts.most_common()} if num_valid else {}


def tool_counts(traces: list[dict]) -> dict:
    """Tool calls, failed tool calls (malformed, unknown tool, or a tool error result),
    prompted-JSON parse failures and agent turns, over the given traces."""
    calls = errors = parse_failures = turns = 0
    for t in traces:
        for turn in t.get("agent_turns") or []:
            turns += 1
            parse_failures += int(turn.get("tool_call_parse") == "parse_failed")
            failed = {r.get("id") for r in turn.get("tool_results") or [] if r.get("error")}
            for tc in turn.get("parsed_tool_calls") or []:
                calls += 1
                bad = tc.get("malformed_arguments") or tc.get("unknown_tool") or tc.get("id") in failed
                errors += int(bool(bad))
    return {"tool_calls": calls, "errors": errors, "parse_failures": parse_failures, "agent_turns": turns}


def compute_metrics(run: dict, k: int, system: str | None = None) -> dict:
    """The gate's metric dict for one run (see module docstring)."""
    traces = run["traces"]
    valid = [t for t in traces if not is_infra_error(t)]
    valid_ids = {str(t.get("simulation_id")) for t in valid}
    counts = tool_counts(valid)
    tool_calls = counts["tool_calls"]
    passes = pass_metrics(reliability_rows(traces, k), k)
    cfg = run.get("run_config") or {}
    trials_per_task = Counter(str(t.get("task_id")) for t in traces)
    failed_valid = sum(1 for t in valid if not t.get("passed"))
    # Rows for passed runs that only AgentTune's detector flagged are evidence, not failures.
    failures = [r for r in run["failures"] if r.get("outcome_failed") is not False]
    return {
        "system": system,
        "run_dir": run.get("run_dir"),
        "agent_model": cfg.get("agent_model"),
        "task_split": cfg.get("task_split"),
        "active_tool_patches": cfg.get("active_tool_patches") or [],
        "k": k,
        "task_ids": sorted(trials_per_task, key=lambda x: (len(x), x)),
        "num_tasks": len(trials_per_task),
        "num_runs": len(traces),
        "num_valid_runs": len(valid),
        "num_infra_errors": len(traces) - len(valid),
        "pass_1": passes["pass_1"],
        "pass_k": passes["pass_k"],
        "num_tasks_scored": passes["num_tasks_scored"],
        "num_tasks_complete": passes["num_tasks_complete"],
        "per_task": passes["per_task"],
        "failure_rate": failed_valid / len(valid) if valid else None,
        "label_rates": label_rates(failures, valid_ids, len(valid)),
        "tool_calls": tool_calls,
        "tool_call_errors": counts["errors"],
        "tool_call_error_rate": counts["errors"] / tool_calls if tool_calls else None,
        "tool_call_parse_failures": counts["parse_failures"],
        "avg_steps": counts["agent_turns"] / len(valid) if valid else None,
        "auditkit": diagnostic_metrics(diagnostic_rows(traces, run.get("tools"))),
        # Labels of each failed valid run, for the gate's target / non-target checks.
        "failed_run_labels": [{"simulation_id": str(r.get("simulation_id")), "labels": r.get("labels") or []}
                      for r in failures if str(r.get("simulation_id")) in valid_ids],
    }


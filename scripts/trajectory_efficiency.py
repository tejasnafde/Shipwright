"""Trajectory efficiency: how much of what the agent did was needed.

    python scripts/trajectory_efficiency.py --gold <tau2 results.json> [--gold ...] \
        --run "base=runs/gate-r16/base" --run "round 16=runs/gate-r16/r16-weight" \
        --run "teacher=runs/r10-ceiling/results.json" [--output efficiency.json]

The reference is tau2's gold action list for each task: the calls an expert agent makes, lookups and writes. A run
is a run dir with ``traces.jsonl`` (our harness) or a tau2 ``results.json`` (the teacher). Per conversation:

  attempts   tool calls the model produced, including those the tool guards refused (they cost a model call)
  on_path    calls matching a gold action (same tool, same arguments), each gold action counted once
  redundant  a call identical to an earlier one with no successful write in between (re-checking)
  failed     calls the tool rejected with an error
  blocked    calls the tool guards refused before they reached the shop
  precision  on_path / calls      the share of calls that were needed
  recall     on_path / gold       the share of the needed steps it took
  efficient  passed, with nothing redundant, failed or blocked, in at most 1.5 x the gold number of calls

Summaries: excess = attempts / gold (1.0 is the expert's path), wasted share = (redundant + failed + blocked) /
attempts, and prompt tokens per solved task (what a solved ticket costs in context). Evaluation only: no run is
changed and nothing here feeds training or the gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

WRITE_PREFIXES = ("cancel_", "modify_", "exchange_", "return_", "transfer_")


def canonical(name: str, arguments: dict | None) -> tuple[str, str]:
    return name, json.dumps(arguments or {}, sort_keys=True)


def load_gold(results_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Task id -> the gold calls the agent should make, from a tau2 results.json."""
    results = json.loads(Path(results_path).read_text())
    return {str(task["id"]): [canonical(action["name"], action.get("arguments"))
                              for action in task["evaluation_criteria"]["actions"]
                              if action.get("requestor", "assistant") == "assistant"]
            for task in results["tasks"]}


def runs(path: Path):
    """(task id, passed, messages, guard-blocked attempts) for each conversation in a run dir or results.json."""
    path = Path(path)
    if path.is_dir():
        for line in (path / "traces.jsonl").read_text().splitlines():
            if line.strip():
                trace = json.loads(line)
                blocked = sum(len(turn.get("guard_rejections") or []) for turn in trace["agent_turns"])
                yield str(trace["task_id"]), trace["passed"] in (True, "True"), trace["messages"], blocked
    else:
        for simulation in json.loads(path.read_text())["simulations"]:
            reward = (simulation.get("reward_info") or {}).get("reward") or 0.0
            yield str(simulation["task_id"]), reward >= 0.999, simulation["messages"], 0


def measure(passed: bool, messages: list[dict], gold: list[tuple[str, str]], blocked: int = 0) -> dict:
    calls, errored = [], {}
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls += [(call["id"], canonical(call["name"], call.get("arguments"))) for call in message["tool_calls"]]
        elif message["role"] == "tool":
            errored[message.get("id")] = bool(message.get("error"))
    remaining, seen = list(gold), set()
    on_path = redundant = failed = 0
    for call_id, key in calls:
        failed += errored.get(call_id, False)
        redundant += key in seen
        seen.add(key)
        if key[0].startswith(WRITE_PREFIXES) and not errored.get(call_id):
            seen = set()                        # a change can change what a lookup returns
        if key in remaining:
            remaining.remove(key)
            on_path += 1
    prompt_tokens = sum((message.get("usage") or {}).get("prompt_tokens") or 0
                        for message in messages if message["role"] == "assistant")
    count, attempts = len(calls), len(calls) + blocked
    return {"passed": passed, "calls": count, "blocked": blocked, "attempts": attempts, "gold": len(gold),
            "on_path": on_path, "redundant": redundant, "failed": failed,
            "precision": on_path / count if count else 0.0, "recall": on_path / max(len(gold), 1),
            "prompt_tokens": prompt_tokens,
            "efficient": passed and not (redundant or failed or blocked) and attempts <= 1.5 * len(gold)}


def summarise(name: str, rows: list[dict]) -> dict:
    def mean(key, subset=rows):
        return statistics.mean(row[key] for row in subset) if subset else 0.0
    solved = [row for row in rows if row["passed"]]
    attempts = sum(row["attempts"] for row in rows)
    return {"system": name, "runs": len(rows), "pass_1": mean("passed"),
            "attempts": mean("attempts"), "blocked": mean("blocked"), "gold": mean("gold"),
            "excess": mean("attempts") / mean("gold") if mean("gold") else 0.0,
            "precision": mean("precision"), "recall": mean("recall"),
            "wasted_share": sum(row["redundant"] + row["failed"] + row["blocked"] for row in rows) / attempts
            if attempts else 0.0,
            "efficient": mean("efficient"), "efficient_of_solved": mean("efficient", solved),
            "prompt_tokens": mean("prompt_tokens"),
            "prompt_tokens_per_solve": sum(row["prompt_tokens"] for row in rows) / len(solved) if solved else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", action="append", required=True, help="tau2 results.json holding the tasks")
    parser.add_argument("--run", action="append", required=True, help="NAME=PATH (run dir or results.json)")
    parser.add_argument("--output", help="write the summaries as JSON here")
    args = parser.parse_args()
    gold: dict = {}
    for path in args.gold:
        gold.update(load_gold(Path(path)))
    summaries = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        summaries.append(summarise(name, [measure(passed, messages, gold[task], blocked)
                                          for task, passed, messages, blocked in runs(Path(path))]))
    print(f"{'system':24s} {'pass^1':>6s} {'excess':>6s} {'prec':>5s} {'recall':>6s} {'wasted':>6s} "
          f"{'blocked':>7s} {'kTok/solve':>10s}")
    for s in summaries:
        per_solve = f"{s['prompt_tokens_per_solve'] / 1000:.0f}" if s["prompt_tokens_per_solve"] else "-"
        print(f"{s['system']:24s} {s['pass_1']:6.1%} {s['excess']:6.2f} {s['precision']:5.0%} {s['recall']:6.0%} "
              f"{s['wasted_share']:6.0%} {s['blocked']:7.2f} {per_solve:>10s}")
    if args.output:
        Path(args.output).write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()

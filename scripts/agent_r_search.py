"""Agent-R step 1: MCTS over tau2 retail runs of the current actor (arxiv 2501.11425).

For each task, a search tree is grown from rollouts of the actor (Tiny Aya, or the last
round's LoRA served on --agent-api-base). A node is a conversation prefix where the agent
acts next. Each iteration selects a node by UCT (c = 0.25), starts --children (4) rollouts
from it at temperature 1 (tau2 replays the prefix: Task.initial_state.message_history),
and backs up each rollout's tau2 reward along its path. --simulations caps the rollouts
per task. Details and the paper mapping: src/shipwright/heal/agent_r.py,
docs/pipeline/agent-r.md.

Task ids (train split only, never held-out or hidden):
  --task-ids 1 2 3     explicit ids
  --from-run RUN_DIR   the task ids routed to weight_fix in RUN_DIR/failures.jsonl
  --all-train          every task of the train split

Writes the normal run outputs (traces.jsonl with one line per rollout, summary.json,
prompts.json, run_config.json, ...) plus
  trees/<task_id>.json   the search tree (visits, Q, rewards, one line per action)
  search_summary.json    rollouts, nodes, leaves, passed leaves and possible pairs per task
Each trace has an ``mcts`` field: {"path": node ids from the root, "task_id"}.
scripts/agent_r_revise.py reads only traces.jsonl and prompts.json.

Spends credits (one rollout = one full tau2 conversation with the actor and the user
simulator): at most tasks x --simulations conversations. --dry-run is offline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import baseline  # noqa: E402
from shipwright.heal import agent_r  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("agent_r_search")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--from-run", type=Path, default=None, help="run dir with failures.jsonl (weight_fix ids)")
    p.add_argument("--all-train", action="store_true")
    p.add_argument("--agent-model", default=baseline.DEFAULT_AGENT, help="the actor (default Tiny Aya Global)")
    p.add_argument("--agent-api-base", default=None,
                   help="OpenAI-compatible endpoint for the actor, e.g. vLLM serving the last round's LoRA")
    p.add_argument("--user-model", required=True,
                   help="user simulator and judge (no default; team choice vertex/openai/gpt-oss-120b-maas)")
    p.add_argument("--simulations", type=int, default=24, help="rollouts per task (paper: 200-300)")
    p.add_argument("--children", type=int, default=agent_r.CHILDREN, help="rollouts per expansion")
    p.add_argument("--c-uct", type=float, default=agent_r.C_UCT)
    p.add_argument("--max-depth", type=int, default=agent_r.MAX_DEPTH, help="agent steps; deeper nodes are not expanded")
    p.add_argument("--temperature", type=float, default=agent_r.ROLLOUT_TEMPERATURE)
    p.add_argument("--max-steps", type=int, default=100, help="tau2 max steps per rollout")
    p.add_argument("--task-concurrency", type=int, default=1, help="tasks searched at once")
    p.add_argument("--tool-patches", type=Path, default=None, help="ledger; its applied patches are active")
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="noisy-oracle fake actor, no network")
    p.add_argument("--dry-run-oracle-p", type=float, default=0.75,
                   help="dry run: chance that the fake actor takes the task's next gold action")
    return p.parse_args(argv)


def resolve_task_ids(args: argparse.Namespace) -> list[str] | None:
    if args.all_train:
        return None
    ids = [str(i) for i in (args.task_ids or [])]
    if args.from_run:
        from teacher_runs import weight_fix_task_ids

        ids += [i for i in weight_fix_task_ids(args.from_run) if i not in ids]
    if not ids:
        raise SystemExit("No task ids: pass --task-ids, --from-run or --all-train.")
    return ids


# ---- dry run: a noisy oracle actor, so offline trees have passing and failing leaves ----

_current = threading.local()


def make_oracle_completion(agent_model: str, p_gold: float, seed: int):
    """Fake LLM for --dry-run. The actor takes the task's next gold action with chance
    p_gold, else a plausible wrong one; it says it is done after the gold actions. The
    user asks for help, then stops once the agent says it is done."""
    from shipwright.llm.dry_run import _response
    from shipwright.llm.prompted_tools import format_tool_call

    rng = random.Random(seed)
    lock = threading.Lock()
    wrong = [
        ("transfer_to_human_agents", {"summary": "dry run"}),
        ("get_user_details", {"user_id": "sara_doe_496"}),
        ("cancel_pending_order", {"order_id": "#W0000000", "reason": "no longer needed"}),
    ]

    def fake(model: str, messages: list[dict], **kwargs) -> object:
        if messages and "metExpectation" in str(messages[0].get("content") or ""):
            verdict = {"results": [{"expectedOutcome": "x", "reasoning": "dry run", "metExpectation": True}]}
            return _response(model, "```json\n" + json.dumps(verdict) + "\n```", None)
        if model == agent_model:
            gold = getattr(_current, "gold", [])
            done = sum(1 for m in messages if m.get("role") == "assistant" and "```json" in (m.get("content") or ""))
            if done >= len(gold):
                return _response(model, "All done. Your request is complete.", None)
            with lock:
                take_gold = rng.random() < p_gold
                bad = rng.choice(wrong)
            name, arguments = gold[done] if take_gold else bad
            return _response(model, format_tool_call(name, arguments), None)
        last = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        if "All done" in last:
            return _response(model, "Thanks. ###STOP###", None)
        return _response(model, "Hi, I need help with my order.", None)

    return fake


# ---- rollouts from a prefix ----


def make_rollout(config, task, seeds, sims_out: list, lock: threading.Lock, oracle: bool):
    """rollout(prefix, n) for agent_r.run_search: n tau2 runs of ``task`` from a prefix."""
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.tasks import InitialState
    from tau2.runner.batch import run_single_task

    by_role = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    gold = [(a.name, a.arguments) for a in ((task.evaluation_criteria.actions or [])
                                            if task.evaluation_criteria else []) if a.requestor == "assistant"]

    def one(prefix: list[dict]) -> dict | None:
        t = task.model_copy(deep=True)
        if prefix:
            init = task.initial_state
            t.initial_state = InitialState(
                initialization_data=init.initialization_data if init else None,
                initialization_actions=init.initialization_actions if init else None,
                message_history=[by_role[m["role"]].model_validate(m) for m in prefix],
            )
        with lock:
            seed = next(seeds)
        if oracle:
            _current.gold = gold
        try:
            sim = run_single_task(config, t, seed=seed)
        except Exception as e:  # noqa: BLE001 - one failed rollout must not stop the search
            logger.warning(f"rollout failed: {e}", task_id=task.id, prefix_messages=len(prefix))
            return None
        d = sim.model_dump(mode="json")
        reward = (d.get("reward_info") or {}).get("reward")
        with lock:
            sims_out.append(d)
        if reward is None or d.get("termination_reason") == "infrastructure_error":
            return None
        return {"messages": d["messages"], "reward": float(reward), "simulation_id": d["id"], "ended": True}

    def rollout(prefix: list[dict], n: int) -> list[dict]:
        with ThreadPoolExecutor(max_workers=n) as ex:
            return [r for r in ex.map(one, [prefix] * n) if r]

    return rollout


def search_task(task, args, config, seeds, lock, sims_out: list, mcts_meta: dict, out_dir: Path) -> dict:
    tree = agent_r.Tree(task.id, c_uct=args.c_uct, children=args.children, max_depth=args.max_depth)
    rollout = make_rollout(config, task, seeds, sims_out, lock, args.dry_run)

    def record(r: dict, path: list[int]) -> None:
        with lock:
            mcts_meta[r["simulation_id"]] = {"mcts": {"task_id": str(task.id), "path": path}}

    counts = agent_r.run_search(tree, rollout, args.simulations, on_rollout=record)
    (out_dir / "trees").mkdir(exist_ok=True)
    (out_dir / "trees" / f"{task.id}.json").write_text(json.dumps(tree.to_json(), indent=1))
    rewards = [lf["reward"] for lf in tree.leaves]
    pairs = agent_r.select_pairs(tree.leaves, alpha=1.0)
    row = {**counts, "passed_leaves": sum(r >= 1.0 for r in rewards),
           "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
           "possible_pairs_alpha_1": len(pairs), "root_q": round(tree.root.q, 3)}
    logger.info("task searched", task_id=task.id, **row)
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_ids = resolve_task_ids(args)
    run_argv = ["--task-split", "train", "--agent-model", args.agent_model, "--user-model", args.user_model,
                "--agent-temperature", str(args.temperature), "--max-steps", str(args.max_steps),
                "--max-concurrency", str(args.children), "--seed", str(args.seed), "--num-tasks", "0"]
    if task_ids:
        run_argv += ["--task-ids", *task_ids]
    if args.output_dir:
        run_argv += ["--output-dir", str(args.output_dir)]
    if args.agent_api_base:
        run_argv += ["--agent-api-base", args.agent_api_base]
    if args.tool_patches:
        run_argv += ["--tool-patches", str(args.tool_patches)]
    if args.dry_run:
        run_argv.append("--dry-run")
    run_args = baseline.parse_args(run_argv)

    out_dir = baseline.make_run_dir(run_args, "agent-r-search")
    fake = make_oracle_completion(args.agent_model, args.dry_run_oracle_p, args.seed) if args.dry_run else None
    setup = baseline.prepare(run_args, out_dir, fake_completion=fake)
    if setup is None:
        return 2
    cfg = json.loads((out_dir / "run_config.json").read_text())
    cfg["agent_r_search"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    from tau2.registry import registry

    config = baseline.text_run_config(run_args, out_dir)
    tasks = registry.get_tasks_loader(run_args.domain)(task_split_name="train")
    if task_ids:
        by_id = {str(t.id): t for t in tasks}
        missing = [i for i in task_ids if i not in by_id]
        if missing:
            raise SystemExit(f"Not in the train split: {missing}")
        tasks = [by_id[i] for i in task_ids]

    seeds = itertools.count(args.seed * 1000)
    lock = threading.Lock()
    sims: list[dict] = []
    meta: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, args.task_concurrency)) as ex:
        rows = list(ex.map(lambda t: search_task(t, args, config, seeds, lock, sims, meta, out_dir), tasks))
    per_task = {str(t.id): r for t, r in zip(tasks, rows, strict=True)}

    log.fail_point("summary")
    _, traces, summary = baseline.finish_run(run_args, out_dir, sims, setup, extra=meta)
    search = {
        "actor": args.agent_model,
        "agent_api_base": args.agent_api_base,
        "simulations_per_task": args.simulations,
        "children": args.children,
        "c_uct": args.c_uct,
        "max_depth": args.max_depth,
        "temperature": args.temperature,
        "tasks": len(tasks),
        "rollouts": len(traces),
        "passed_rollouts": summary.get("num_passed"),
        "tasks_with_pairs": sum(1 for r in rows if r["possible_pairs_alpha_1"]),
        "possible_pairs_alpha_1": sum(r["possible_pairs_alpha_1"] for r in rows),
        "cost_estimate_usd": summary.get("cost_estimate_usd"),
        "per_task": per_task,
    }
    (out_dir / "search_summary.json").write_text(json.dumps(search, indent=2))
    print(json.dumps({k: v for k, v in search.items() if k != "per_task"}, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="agent_r_search"))

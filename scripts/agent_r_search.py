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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Which tasks to search.
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--from-run", type=Path, default=None, help="run dir with failures.jsonl (weight_fix ids)")
    parser.add_argument("--all-train", action="store_true")

    # Which models play the actor and the user.
    parser.add_argument("--agent-model", default=baseline.DEFAULT_AGENT, help="the actor (default Tiny Aya Global)")
    parser.add_argument(
        "--agent-api-base",
        default=None,
        help="OpenAI-compatible endpoint for the actor, e.g. vLLM serving the last round's LoRA",
    )
    parser.add_argument(
        "--user-model",
        required=True,
        help="user simulator and judge (no default; team choice vertex/openai/gpt-oss-120b-maas)",
    )

    # The search itself.
    parser.add_argument("--simulations", type=int, default=24, help="rollouts per task (paper: 200-300)")
    parser.add_argument("--children", type=int, default=agent_r.CHILDREN, help="rollouts per expansion")
    parser.add_argument("--c-uct", type=float, default=agent_r.C_UCT)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=agent_r.MAX_DEPTH,
        help="agent steps; deeper nodes are not expanded",
    )
    parser.add_argument("--temperature", type=float, default=agent_r.ROLLOUT_TEMPERATURE)
    parser.add_argument("--max-steps", type=int, default=100, help="tau2 max steps per rollout")
    parser.add_argument("--task-concurrency", type=int, default=1, help="tasks searched at once")
    parser.add_argument("--tool-patches", type=Path, default=None, help="ledger; its applied patches are active")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)

    # Dry run.
    parser.add_argument("--dry-run", action="store_true", help="noisy-oracle fake actor, no network")
    parser.add_argument(
        "--dry-run-oracle-p",
        type=float,
        default=0.75,
        help="dry run: chance that the fake actor takes the task's next gold action",
    )
    return parser.parse_args(argv)


def resolve_task_ids(search_options: argparse.Namespace) -> list[str] | None:
    if search_options.all_train:
        return None

    task_ids = [str(task_id) for task_id in (search_options.task_ids or [])]
    if search_options.from_run:
        from teacher_runs import weight_fix_task_ids

        for task_id in weight_fix_task_ids(search_options.from_run):
            if task_id not in task_ids:
                task_ids.append(task_id)
    if not task_ids:
        raise SystemExit("No task ids: pass --task-ids, --from-run or --all-train.")
    return task_ids


# ---- dry run: a noisy oracle actor, so offline trees have passing and failing leaves ----

_current = threading.local()


def make_oracle_completion(agent_model: str, gold_action_probability: float, seed: int):
    """Fake LLM for --dry-run. The actor takes the task's next gold action with chance
    gold_action_probability, else a plausible wrong one; it says it is done after the gold
    actions. The user asks for help, then stops once the agent says it is done."""
    from shipwright.llm.dry_run import _response
    from shipwright.llm.prompted_tools import format_tool_call

    random_generator = random.Random(seed)
    generator_lock = threading.Lock()
    wrong_actions = [
        ("transfer_to_human_agents", {"summary": "dry run"}),
        ("get_user_details", {"user_id": "sara_doe_496"}),
        ("cancel_pending_order", {"order_id": "#W0000000", "reason": "no longer needed"}),
    ]

    def fake_completion(model: str, messages: list[dict], **completion_options) -> object:
        first_message_text = ""
        if messages:
            first_message_text = str(messages[0].get("content") or "")
        if "metExpectation" in first_message_text:
            verdict = {"results": [{"expectedOutcome": "x", "reasoning": "dry run", "metExpectation": True}]}
            return _response(model, "```json\n" + json.dumps(verdict) + "\n```", None)

        if model == agent_model:
            gold_actions = getattr(_current, "gold", [])
            actions_done = 0
            for message in messages:
                if message.get("role") == "assistant" and "```json" in (message.get("content") or ""):
                    actions_done += 1
            if actions_done >= len(gold_actions):
                return _response(model, "All done. Your request is complete.", None)

            with generator_lock:
                takes_gold_action = random_generator.random() < gold_action_probability
                wrong_action = random_generator.choice(wrong_actions)
            if takes_gold_action:
                tool_name, tool_arguments = gold_actions[actions_done]
            else:
                tool_name, tool_arguments = wrong_action
            return _response(model, format_tool_call(tool_name, tool_arguments), None)

        # The user simulator: the last user-role message is what the agent said.
        last_user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user_text = message.get("content") or ""
                break
        if "All done" in last_user_text:
            return _response(model, "Thanks. ###STOP###", None)
        return _response(model, "Hi, I need help with my order.", None)

    return fake_completion


# ---- rollouts from a prefix ----


def gold_assistant_actions(task) -> list[tuple[str, dict]]:
    """The (tool name, arguments) the task expects from the assistant, in order."""
    if not task.evaluation_criteria:
        return []
    expected_actions = task.evaluation_criteria.actions or []
    return [(action.name, action.arguments) for action in expected_actions if action.requestor == "assistant"]


def make_rollout(config, task, seeds, simulations_out: list, simulations_lock: threading.Lock, use_oracle: bool):
    """rollout(prefix, count) for agent_r.run_search: ``count`` tau2 runs of ``task`` from a prefix."""
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.tasks import InitialState
    from tau2.runner.batch import run_single_task

    message_class_by_role = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    gold_actions = gold_assistant_actions(task)

    def run_one_rollout(prefix: list[dict]) -> dict | None:
        task_copy = task.model_copy(deep=True)
        if prefix:
            initial_state = task.initial_state
            task_copy.initial_state = InitialState(
                initialization_data=initial_state.initialization_data if initial_state else None,
                initialization_actions=initial_state.initialization_actions if initial_state else None,
                message_history=[message_class_by_role[m["role"]].model_validate(m) for m in prefix],
            )
        with simulations_lock:
            seed = next(seeds)
        if use_oracle:
            _current.gold = gold_actions
        try:
            simulation = run_single_task(config, task_copy, seed=seed)
        except Exception as rollout_error:  # noqa: BLE001 - one failed rollout must not stop the search
            logger.warning(f"rollout failed: {rollout_error}", task_id=task.id, prefix_messages=len(prefix))
            return None

        simulation_dict = simulation.model_dump(mode="json")
        reward = (simulation_dict.get("reward_info") or {}).get("reward")
        with simulations_lock:
            simulations_out.append(simulation_dict)
        if reward is None or simulation_dict.get("termination_reason") == "infrastructure_error":
            return None
        return {
            "messages": simulation_dict["messages"],
            "reward": float(reward),
            "simulation_id": simulation_dict["id"],
            "ended": True,
        }

    def rollout(prefix: list[dict], count: int) -> list[dict]:
        with ThreadPoolExecutor(max_workers=count) as executor:
            rollout_results = executor.map(run_one_rollout, [prefix] * count)
            return [result for result in rollout_results if result]

    return rollout


def search_task(
    task,
    search_options,
    config,
    seeds,
    simulations_lock,
    simulations_out: list,
    mcts_metadata: dict,
    run_directory: Path,
) -> dict:
    tree = agent_r.Tree(
        task.id,
        c_uct=search_options.c_uct,
        children=search_options.children,
        max_depth=search_options.max_depth,
    )
    rollout = make_rollout(config, task, seeds, simulations_out, simulations_lock, search_options.dry_run)

    def record_rollout(rollout_result: dict, path: list[int]) -> None:
        with simulations_lock:
            mcts_metadata[rollout_result["simulation_id"]] = {"mcts": {"task_id": str(task.id), "path": path}}

    counts = agent_r.run_search(tree, rollout, search_options.simulations, on_rollout=record_rollout)

    (run_directory / "trees").mkdir(exist_ok=True)
    (run_directory / "trees" / f"{task.id}.json").write_text(json.dumps(tree.to_json(), indent=1))

    leaf_rewards = [leaf["reward"] for leaf in tree.leaves]
    possible_pairs = agent_r.select_pairs(tree.leaves, alpha=1.0)
    if leaf_rewards:
        mean_reward = round(sum(leaf_rewards) / len(leaf_rewards), 3)
    else:
        mean_reward = None
    task_row = {
        **counts,
        "passed_leaves": sum(reward >= 1.0 for reward in leaf_rewards),
        "mean_reward": mean_reward,
        "possible_pairs_alpha_1": len(possible_pairs),
        "root_q": round(tree.root.q, 3),
    }
    logger.info("task searched", task_id=task.id, **task_row)
    return task_row


def build_baseline_argv(search_options: argparse.Namespace, task_ids: list[str] | None) -> list[str]:
    """The baseline.py command line the search reuses for its tau2 setup (always the train split)."""
    baseline_argv = [
        "--task-split", "train",
        "--agent-model", search_options.agent_model,
        "--user-model", search_options.user_model,
        "--agent-temperature", str(search_options.temperature),
        "--max-steps", str(search_options.max_steps),
        "--max-concurrency", str(search_options.children),
        "--seed", str(search_options.seed),
        "--num-tasks", "0",
    ]
    if task_ids:
        baseline_argv += ["--task-ids", *task_ids]
    if search_options.output_dir:
        baseline_argv += ["--output-dir", str(search_options.output_dir)]
    if search_options.agent_api_base:
        baseline_argv += ["--agent-api-base", search_options.agent_api_base]
    if search_options.tool_patches:
        baseline_argv += ["--tool-patches", str(search_options.tool_patches)]
    if search_options.dry_run:
        baseline_argv.append("--dry-run")
    return baseline_argv


def load_train_tasks(domain: str, task_ids: list[str] | None) -> list:
    from tau2.registry import registry

    train_tasks = registry.get_tasks_loader(domain)(task_split_name="train")
    if not task_ids:
        return train_tasks

    task_by_id = {str(task.id): task for task in train_tasks}
    missing_ids = [task_id for task_id in task_ids if task_id not in task_by_id]
    if missing_ids:
        raise SystemExit(f"Not in the train split: {missing_ids}")
    return [task_by_id[task_id] for task_id in task_ids]


def record_search_options(run_directory: Path, search_options: argparse.Namespace) -> None:
    """Add the search's own options to run_config.json."""
    run_config_path = run_directory / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["agent_r_search"] = {
        option_name: (str(option_value) if isinstance(option_value, Path) else option_value)
        for option_name, option_value in vars(search_options).items()
    }
    run_config_path.write_text(json.dumps(run_config, indent=2))


def main(argv: list[str] | None = None) -> int:
    search_options = parse_args(argv)
    task_ids = resolve_task_ids(search_options)
    baseline_options = baseline.parse_args(build_baseline_argv(search_options, task_ids))

    run_directory = baseline.make_run_dir(baseline_options, "agent-r-search")
    if search_options.dry_run:
        fake_completion = make_oracle_completion(
            search_options.agent_model, search_options.dry_run_oracle_p, search_options.seed
        )
    else:
        fake_completion = None
    run_setup = baseline.prepare(baseline_options, run_directory, fake_completion=fake_completion)
    if run_setup is None:
        return 2
    record_search_options(run_directory, search_options)

    tau2_config = baseline.text_run_config(baseline_options, run_directory)
    tasks = load_train_tasks(baseline_options.domain, task_ids)

    seeds = itertools.count(search_options.seed * 1000)
    simulations_lock = threading.Lock()
    simulations: list[dict] = []
    mcts_metadata: dict = {}

    def search_one_task(task) -> dict:
        return search_task(
            task, search_options, tau2_config, seeds, simulations_lock,
            simulations, mcts_metadata, run_directory,
        )

    with ThreadPoolExecutor(max_workers=max(1, search_options.task_concurrency)) as executor:
        task_rows = list(executor.map(search_one_task, tasks))
    row_by_task_id = {str(task.id): row for task, row in zip(tasks, task_rows, strict=True)}

    log.fail_point("summary")
    result = baseline.finish_run(
        baseline_options, run_directory, simulations, run_setup, extra_trace_fields=mcts_metadata
    )

    search_summary = {
        "actor": search_options.agent_model,
        "agent_api_base": search_options.agent_api_base,
        "simulations_per_task": search_options.simulations,
        "children": search_options.children,
        "c_uct": search_options.c_uct,
        "max_depth": search_options.max_depth,
        "temperature": search_options.temperature,
        "tasks": len(tasks),
        "rollouts": len(result.traces),
        "passed_rollouts": result.summary.get("num_passed"),
        "tasks_with_pairs": sum(1 for row in task_rows if row["possible_pairs_alpha_1"]),
        "possible_pairs_alpha_1": sum(row["possible_pairs_alpha_1"] for row in task_rows),
        "cost_estimate_usd": result.summary.get("cost_estimate_usd"),
        "per_task": row_by_task_id,
    }
    (run_directory / "search_summary.json").write_text(json.dumps(search_summary, indent=2))
    printable_summary = {key: value for key, value in search_summary.items() if key != "per_task"}
    print(json.dumps(printable_summary, indent=2))
    print(f"\nWrote {run_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="agent_r_search"))

"""Teacher runs: a strong model plays the agent on tau2 retail, in the student's format.

The teacher and the user simulator are required arguments (--teacher-model, --user-model); see
docs/pipeline/training.md for the team's current choice (Vertex AI).

The teacher sees exactly the prompt Tiny Aya sees: the compact tool list, the ```json
block protocol, no native tools, the same 8192-token prompt budget (see
shipwright.llm.prompted_tools and shipwright.llm.context). Only its reply budget is larger,
because it reasons before it answers. So its trajectories are already in the student
format, and scripts/build_sft.py turns the successful ones into training data.

Task ids (train split only):
  --from-run RUN_DIR   the task ids of the failures in RUN_DIR/failures.jsonl whose route is in
                       --routes (default: tool_fix and weight_fix; the round has no tool patches,
                       so a task routed to tool_fix is only fixed by teaching the weights).
                       Default when no other source is given: the baseline run
  --skip-unsolved-from TEACHER_RUN_DIR ...
                       leave out the tasks that those earlier teacher runs failed in every
                       decided trial (at least --min-failed-trials of them, no pass)
  --only-solved-from TEACHER_RUN_DIR ...
                       the harvest mode: keep only the tasks those earlier teacher runs solved, and play them
                       again (use --trials 5). A task that has passed once passes again about 48% of the time
                       (all teacher runs so far: 34%), so each trial is worth more and the passes add new
                       conversations on tasks the SFT data already covers
  --skip-solved-from TEACHER_RUN_DIR ...
                       leave out the tasks that those earlier teacher runs already solved: their
                       passing traces go into the SFT data (build_sft.py --teacher), so replaying
                       them again only pays for the same examples
  --task-ids 1 2 3     explicit ids
  --all-train          every task of the train split

Writes the normal baseline outputs (traces.jsonl, summary.json, prompts.json, ...) plus
  kept.jsonl           the traces with reward == 1 (the only ones used for training)
  teacher_summary.json kept count, per-task kept count
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import baseline  # noqa: E402

DEFAULT_FROM_RUN = ROOT / "runs" / "baseline-20261108-214337"
# The teacher's reasoning counts against max_tokens; 700 (the student's budget) is too small.
TEACHER_REPLY_TOKENS = 4096


DEFAULT_ROUTES = ("tool_fix", "weight_fix")
# A task the teacher failed this many decided (not infra error) trials in a row, with no pass, is
# left out of the next round: round 2 spent 3 trials each on 11 of 24 tasks it never solved.
MIN_FAILED_TRIALS = 3


def failed_task_ids(run_directory: Path, routes: tuple[str, ...] = DEFAULT_ROUTES) -> list[str]:
    """Task ids of the failed runs that classify.py routed to one of ``routes``, in file order."""
    task_ids: list[str] = []
    with open(run_directory / "failures.jsonl") as failures_file:
        for line in failures_file:
            failure = json.loads(line)
            task_id = str(failure["task_id"])
            if failure.get("route") in routes and task_id not in task_ids:
                task_ids.append(task_id)
    return task_ids


def solved_task_ids(teacher_run_directories: list[Path]) -> set[str]:
    """Tasks with at least one passing trace in the earlier teacher runs."""
    solved: set[str] = set()
    for run_directory in teacher_run_directories:
        with open(run_directory / "traces.jsonl") as traces_file:
            for line in traces_file:
                trace = json.loads(line)
                if trace.get("passed"):
                    solved.add(str(trace["task_id"]))
    return solved


def unsolved_task_ids(teacher_run_directories: list[Path], min_failed_trials: int = MIN_FAILED_TRIALS) -> set[str]:
    """Tasks the earlier teacher runs failed in at least ``min_failed_trials`` decided trials and never passed."""
    failed_trials: dict[str, int] = {}
    solved: set[str] = set()
    for run_directory in teacher_run_directories:
        with open(run_directory / "traces.jsonl") as traces_file:
            for line in traces_file:
                trace = json.loads(line)
                task_id = str(trace["task_id"])
                if trace.get("passed"):
                    solved.add(task_id)
                elif trace.get("termination_reason") != "infrastructure_error":
                    failed_trials[task_id] = failed_trials.get(task_id, 0) + 1
    return {task_id for task_id, count in failed_trials.items() if count >= min_failed_trials and task_id not in solved}


def keep_successful(traces: list[dict]) -> list[dict]:
    """Only runs with reward == 1 (build_trace sets ``passed`` for those)."""
    return [trace for trace in traces if trace.get("passed")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Which tasks the teacher plays.
    parser.add_argument("--from-run", type=Path, default=None, help="run dir with failures.jsonl")
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--all-train", action="store_true", help="run every train task")
    parser.add_argument(
        "--routes", default=",".join(DEFAULT_ROUTES), help="with --from-run: the failure routes to replay"
    )
    parser.add_argument("--skip-unsolved-from", type=Path, nargs="*", default=[], help="earlier teacher run dirs")
    parser.add_argument("--only-solved-from", type=Path, nargs="*", default=[], help="earlier teacher run dirs")
    parser.add_argument("--skip-solved-from", type=Path, nargs="*", default=[], help="earlier teacher run dirs")
    parser.add_argument("--min-failed-trials", type=int, default=MIN_FAILED_TRIALS)
    parser.add_argument("--trials", type=int, default=3)

    # Which models play the teacher and the user.
    parser.add_argument(
        "--teacher-model",
        required=True,
        help="vertex/<publisher>/<model> or an HF router litellm id (no default; the team's current"
        " choice is vertex/deepseek-ai/deepseek-v4.1-flash-maas; it must allow training on its outputs)",
    )
    parser.add_argument(
        "--user-model",
        required=True,
        help="user simulator and judge (no default; the team's current choice is"
        " vertex/openai/gpt-oss-120b-maas)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="teacher temperature (variety across trials)")
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="low/medium/high for reasoning teachers such as gpt-oss; empty: not sent",
    )
    parser.add_argument("--reply-tokens", type=int, default=TEACHER_REPLY_TOKENS)

    # How the run executes.
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=None, help="stop the run once its metered cost passes this")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="fake LLM, no network")
    return parser.parse_args(argv)


def resolve_task_ids(teacher_options: argparse.Namespace) -> list[str] | None:
    """Explicit ids plus the failed ids of the routes (minus the unsolved ones); None means the whole train split."""
    if teacher_options.all_train:
        return None

    task_ids = list(teacher_options.task_ids or [])

    if teacher_options.from_run:
        from_run = teacher_options.from_run
    elif task_ids:
        from_run = None  # explicit ids given: do not also read the default baseline run
    else:
        from_run = DEFAULT_FROM_RUN

    if from_run:
        routes = tuple(route.strip() for route in teacher_options.routes.split(",") if route.strip())
        for task_id in failed_task_ids(from_run, routes):
            if task_id not in task_ids:
                task_ids.append(task_id)
    if teacher_options.skip_unsolved_from:
        unsolved = unsolved_task_ids(teacher_options.skip_unsolved_from, teacher_options.min_failed_trials)
        skipped = [task_id for task_id in task_ids if task_id in unsolved]
        task_ids = [task_id for task_id in task_ids if task_id not in unsolved]
        print(f"Skipping {len(skipped)} tasks the teacher never solved before: {' '.join(skipped)}")
    if teacher_options.only_solved_from:
        solved = solved_task_ids(teacher_options.only_solved_from)
        task_ids = [task_id for task_id in task_ids if task_id in solved]
        print(f"Harvest: {len(task_ids)} tasks the teacher has solved before")
    if teacher_options.skip_solved_from:
        solved = solved_task_ids(teacher_options.skip_solved_from)
        skipped = [task_id for task_id in task_ids if task_id in solved]
        task_ids = [task_id for task_id in task_ids if task_id not in solved]
        print(f"Skipping {len(skipped)} tasks the teacher already solved: {' '.join(skipped)}")
    if not task_ids:
        raise SystemExit("No task ids: pass --task-ids, --from-run or --all-train.")
    return task_ids


def build_baseline_argv(teacher_options: argparse.Namespace, task_ids: list[str] | None) -> list[str]:
    """The baseline.py command line for a teacher run (always the train split)."""
    baseline_argv = [
        "--task-split", "train",  # never the held-out or hidden split
        "--trials", str(teacher_options.trials),
        "--agent-model", teacher_options.teacher_model,
        "--user-model", teacher_options.user_model,
        "--agent-temperature", str(teacher_options.temperature),
        "--max-steps", str(teacher_options.max_steps),
        "--max-concurrency", str(teacher_options.max_concurrency),
        "--seed", str(teacher_options.seed),
    ]
    if task_ids:
        baseline_argv += ["--task-ids", *task_ids]
    else:
        baseline_argv += ["--num-tasks", "0"]
    if teacher_options.output_dir:
        baseline_argv += ["--output-dir", str(teacher_options.output_dir)]
    if teacher_options.max_cost_usd is not None:
        baseline_argv += ["--max-cost-usd", str(teacher_options.max_cost_usd)]
    if teacher_options.dry_run:
        baseline_argv.append("--dry-run")
    return baseline_argv


def count_kept_per_task(kept_traces: list[dict]) -> dict[str, int]:
    kept_per_task: dict[str, int] = {}
    for trace in kept_traces:
        task_id = str(trace["task_id"])
        kept_per_task[task_id] = kept_per_task.get(task_id, 0) + 1
    return kept_per_task


def main(argv: list[str] | None = None) -> int:
    teacher_options = parse_args(argv)
    task_ids = resolve_task_ids(teacher_options)
    baseline_options = baseline.parse_args(build_baseline_argv(teacher_options, task_ids))

    if teacher_options.reasoning_effort and not teacher_options.dry_run:
        teacher_llm_args = {"reasoning_effort": teacher_options.reasoning_effort}
    else:
        teacher_llm_args = {}

    result = baseline.run(
        baseline_options,
        prefix="teacher",
        prompted_models={teacher_options.teacher_model: teacher_options.reply_tokens},
        agent_llm_args=teacher_llm_args,
        role_names={"agent": "teacher"},
    )
    if result is None:
        return 2

    from shipwright.heal.traces import write_jsonl

    kept_traces = keep_successful(result.traces)
    write_jsonl(result.run_directory / "kept.jsonl", kept_traces)

    kept_per_task = count_kept_per_task(kept_traces)
    task_ids_run = {str(trace["task_id"]) for trace in result.traces}
    cache_hit_rate_by_role = result.summary.get("cache_hit_rate") or {}
    teacher_summary = {
        "teacher_model": teacher_options.teacher_model,
        "task_ids": task_ids or "all train",
        "trials": teacher_options.trials,
        "runs": len(result.traces),
        "kept": len(kept_traces),
        "kept_per_task": kept_per_task,
        "tasks_without_success": sorted(task_ids_run - set(kept_per_task)),
        "pass_rate": result.summary.get("pass_rate"),
        "cache_hit_rate": cache_hit_rate_by_role.get("agent"),
        "cost_estimate_usd": result.summary.get("cost_estimate_usd"),
    }
    (result.run_directory / "teacher_summary.json").write_text(json.dumps(teacher_summary, indent=2))
    print(json.dumps(teacher_summary, indent=2))
    print(f"\nWrote {result.run_directory}")
    return 0


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="teacher"))

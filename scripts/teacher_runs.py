"""Teacher runs: a strong model plays the agent on tau2 retail, in the student's format.

The teacher and the user simulator are required arguments (--teacher-model, --user-model); see
docs/training.md for the team's current choice (Vertex AI).

The teacher sees exactly the prompt Tiny Aya sees: the compact tool list, the ```json
block protocol, no native tools, the same 8192-token prompt budget (see
shipwright.prompted_tools and shipwright.context). Only its reply budget is larger,
because it reasons before it answers. So its trajectories are already in the student
format, and scripts/build_sft.py turns the successful ones into training data.

Task ids (train split only):
  --from-run RUN_DIR   the task ids routed to weight_fix in RUN_DIR/failures.jsonl (default
                       when no other source is given: the baseline run)
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


def weight_fix_task_ids(run_dir: Path) -> list[str]:
    """Task ids of the failed runs that classify.py routed to weight_fix, in file order."""
    ids: list[str] = []
    with open(run_dir / "failures.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row.get("route") == "weight_fix" and str(row["task_id"]) not in ids:
                ids.append(str(row["task_id"]))
    return ids


def keep_successful(traces: list[dict]) -> list[dict]:
    """Only runs with reward == 1 (build_trace sets ``passed`` for those)."""
    return [t for t in traces if t.get("passed")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-run", type=Path, default=None, help="run dir with failures.jsonl")
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--all-train", action="store_true", help="run every train task")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--teacher-model", required=True,
                   help="vertex/<publisher>/<model> or an HF router litellm id (no default; the team's current"
                        " choice is vertex/deepseek-ai/deepseek-v4.1-flash-maas; it must allow training on its outputs)")
    p.add_argument("--user-model", required=True,
                   help="user simulator and judge (no default; the team's current choice is"
                        " vertex/openai/gpt-oss-120b-maas)")
    p.add_argument("--temperature", type=float, default=0.7, help="teacher temperature (variety across trials)")
    p.add_argument("--reasoning-effort", default="", help="low/medium/high for reasoning teachers such as gpt-oss; empty: not sent")
    p.add_argument("--reply-tokens", type=int, default=TEACHER_REPLY_TOKENS)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="fake LLM, no network")
    return p.parse_args(argv)


def resolve_task_ids(args: argparse.Namespace) -> list[str] | None:
    """Explicit ids plus weight_fix ids; None means the whole train split."""
    if args.all_train:
        return None
    ids = list(args.task_ids or [])
    from_run = args.from_run or (None if ids else DEFAULT_FROM_RUN)
    if from_run:
        ids += [i for i in weight_fix_task_ids(from_run) if i not in ids]
    if not ids:
        raise SystemExit("No task ids: pass --task-ids, --from-run or --all-train.")
    return ids


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_ids = resolve_task_ids(args)
    run_argv = [
        "--task-split", "train",  # never the held-out or hidden split
        "--trials", str(args.trials),
        "--agent-model", args.teacher_model,
        "--user-model", args.user_model,
        "--agent-temperature", str(args.temperature),
        "--max-steps", str(args.max_steps),
        "--max-concurrency", str(args.max_concurrency),
        "--seed", str(args.seed),
    ]
    run_argv += ["--task-ids", *task_ids] if task_ids else ["--num-tasks", "0"]
    if args.output_dir:
        run_argv += ["--output-dir", str(args.output_dir)]
    if args.dry_run:
        run_argv.append("--dry-run")
    run_args = baseline.parse_args(run_argv)

    llm_args = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort and not args.dry_run else {}
    result = baseline.run(
        run_args,
        prefix="teacher",
        prompted_models={args.teacher_model: args.reply_tokens},
        agent_llm_args=llm_args,
        role_names={"agent": "teacher"},
    )
    if result is None:
        return 2
    out_dir, traces, summary = result

    from shipwright.traces import write_jsonl

    kept = keep_successful(traces)
    write_jsonl(out_dir / "kept.jsonl", kept)
    per_task: dict[str, int] = {}
    for t in kept:
        per_task[str(t["task_id"])] = per_task.get(str(t["task_id"]), 0) + 1
    teacher_summary = {
        "teacher_model": args.teacher_model,
        "task_ids": task_ids or "all train",
        "trials": args.trials,
        "runs": len(traces),
        "kept": len(kept),
        "kept_per_task": per_task,
        "tasks_without_success": sorted({str(t["task_id"]) for t in traces} - set(per_task)),
        "pass_rate": summary.get("pass_rate"),
        "cache_hit_rate": (summary.get("cache_hit_rate") or {}).get("agent"),
        "cost_estimate_usd": summary.get("cost_estimate_usd"),
    }
    (out_dir / "teacher_summary.json").write_text(json.dumps(teacher_summary, indent=2))
    print(json.dumps(teacher_summary, indent=2))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

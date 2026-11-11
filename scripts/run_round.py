"""Run one round of the loop, one stage at a time, with a cost estimate and a hard cap per stage.

  .venv/bin/python scripts/run_round.py plan --round 2 --user-model U --teacher-model T   # free
  .venv/bin/python scripts/run_round.py baseline --round 2 ...                            # spends
  .venv/bin/python scripts/run_round.py classify --round 2 ...
  .venv/bin/python scripts/run_round.py teacher --round 2 ...
  .venv/bin/python scripts/run_round.py data --round 2 ...                                # free
  .venv/bin/python scripts/run_round.py train --round 2 ... [--launch]                    # HF GPU job
  .venv/bin/python scripts/run_round.py gate --round 2 ... [--launch]                     # HF GPU job

Stages run in that order. Each one:
  - checks that the Cloud Run job runs the image built from this commit (cloud stages),
  - refuses to start if its cap would take the round past --round-budget-usd,
  - passes its cap to the script as --max-cost-usd (or, for GPU jobs, price x timeout),
  - writes the result to runs/round-N.json: run dir, estimate, actual cost, status.

The cloud stages (baseline, classify, teacher) run on the owner's Cloud Run job and upload to
gs://lexsi-hackathon-runs/<owner>/. train and gate are plans unless you pass --launch.
Run dirs are named r<round>-baseline and r<round>-teacher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.control_room import gcs_runs  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("run_round")

REGION = "asia-south1"
IMAGE = "asia-south1-docker.pkg.dev/lexsi-hackathon/shipwright/harness"
STAGE_ORDER = ["baseline", "classify", "teacher", "data", "train", "gate"]


@dataclass
class StageBudget:
    estimate_usd: float  # what we expect it to cost
    cap_usd: float       # the most it may cost: --max-cost-usd for API stages, price x timeout for GPU jobs


# The team has $10 in total (2026-11-12), so every cap is small and the caps add up to well under it.
# Estimates come from the round 1 runs (runs/r1-baseline-train: $1.49 for 74 runs, runs/r1-teacher-t1:
# $0.53 for 54 runs, so 3 trials of about 54 tasks is about $1.60). GPU caps are price x timeout:
# train L4 $0.80/h x 1h, gate A10G large $1.50/h x 1h plus about $0.60 of user simulator calls.
GPU_TIMEOUT = "1h"  # the gate's Vertex token lasts about 1 hour (AGENTS.md traps), so the gate must fit in it
TRAIN_FLAVOR = "l4x1"
GATE_FLAVOR = "a10g-large"
BUDGETS = {
    "baseline": StageBudget(estimate_usd=1.5, cap_usd=2.0),
    "classify": StageBudget(estimate_usd=0.05, cap_usd=0.30),
    "teacher": StageBudget(estimate_usd=1.6, cap_usd=2.0),
    "data": StageBudget(estimate_usd=0.0, cap_usd=0.0),
    "train": StageBudget(estimate_usd=0.4, cap_usd=0.8),
    "gate": StageBudget(estimate_usd=1.8, cap_usd=2.1),
}
DESCRIPTIONS = {
    "baseline": "Tiny Aya (no adapter, no tool patches) on the 74 train tasks, 1 trial",
    "classify": "label and route every failure (AgentTune detector and classifier, our rules)",
    "teacher": "the teacher replays the weight-fix failures, 3 trials each; passes are kept",
    "data": "build the SFT dataset from the teacher passes and push it to the HF dataset repo",
    "train": "LoRA on Tiny Aya Global with AgentTune (HF Job, L4, 1h limit)",
    "gate": "base vs the new adapter on the 20 held-out tasks, k=3, 8 checks (HF Job, A10G, 1h limit)",
}


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["plan", *STAGE_ORDER])
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--user-model", required=True, help="user simulator and judge, e.g. vertex/openai/gpt-oss-120b-maas")
    parser.add_argument("--teacher-model", required=True, help="e.g. vertex/deepseek-ai/deepseek-v4.1-flash-maas")
    parser.add_argument("--owner", default="ishank", help="whose Cloud Run job runs the cloud stages")
    parser.add_argument("--round-budget-usd", type=float, default=8.0, help="no stage starts if its cap passes this total")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="cloud stages with the fake LLM: tests the plumbing, free")
    parser.add_argument("--launch", action="store_true", help="train and gate: submit the HF Job (spends credits)")
    return parser.parse_args()


# ---------------------------------------------------------------- the manifest

def manifest_path(round_number: int) -> Path:
    return ROOT / "runs" / f"round-{round_number}.json"


def load_manifest(options: argparse.Namespace) -> dict:
    path = manifest_path(options.round)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "round": options.round,
        "owner": options.owner,
        "user_model": options.user_model,
        "teacher_model": options.teacher_model,
        "round_budget_usd": options.round_budget_usd,
        "stages": {},
    }


def save_manifest(manifest: dict, round_number: int) -> None:
    """Spent so far: the metered cost of a stage, or its estimate when the stage is not metered."""
    spent = 0.0
    for stage_entry in manifest["stages"].values():
        if stage_entry.get("actual_usd") is not None:
            spent += stage_entry["actual_usd"]
        else:
            spent += stage_entry.get("estimate_usd") or 0.0
    manifest["spent_usd"] = round(spent, 4)
    manifest_path(round_number).write_text(json.dumps(manifest, indent=2))


def baseline_dir_name(round_number: int) -> str:
    return f"r{round_number}-baseline"


def teacher_dir_name(round_number: int) -> str:
    return f"r{round_number}-teacher"


def sft_dir_name(round_number: int) -> str:
    return f"r{round_number}-sft"


# ---------------------------------------------------------------- checks before spending

def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


def gcloud_command(*arguments: str) -> list[str]:
    binary = gcs_runs.gcloud_binary()
    if binary is None:
        raise SystemExit("gcloud not found: install it or set SHIPWRIGHT_GCLOUD")
    return [binary, f"--configuration={gcs_runs.gcloud_config(binary)}", f"--project={gcs_runs.PROJECT}", *arguments]


def check_cloud_job_runs_this_commit(owner: str) -> None:
    """The job must run the image built from HEAD, or the round would run old code."""
    head_sha = run_command(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).stdout.strip()
    dirty = run_command(["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise SystemExit("uncommitted changes: commit and push first, so the image matches what you ran")

    job = run_command(gcloud_command(
        "run", "jobs", "describe", f"shipwright-{owner}", f"--region={REGION}", "--format=json"))
    job_image = json.loads(job.stdout)["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]["image"]
    job_digest = job_image.split("@")[-1]

    tag = run_command(gcloud_command(
        "artifacts", "docker", "images", "describe", f"{IMAGE}:{head_sha}", "--format=value(image_summary.digest)"))
    head_digest = tag.stdout.strip()
    if not head_digest:
        raise SystemExit(f"no image for commit {head_sha} yet: push it and wait for the Cloud Build")
    if head_digest != job_digest:
        raise SystemExit(f"shipwright-{owner} runs {job_digest[:19]}, not the image of {head_sha}")
    logger.info("cloud job runs this commit", commit=head_sha, digest=job_digest[:19])


def check_budget(manifest: dict, stage: str) -> None:
    spent = manifest.get("spent_usd") or 0.0
    cap = BUDGETS[stage].cap_usd
    if spent + cap > manifest["round_budget_usd"]:
        raise SystemExit(
            f"{stage}: cap ${cap:.2f} on top of ${spent:.2f} spent passes the round budget "
            f"${manifest['round_budget_usd']:.2f}"
        )


# ---------------------------------------------------------------- cloud stages

def run_cloud_job(owner: str, script_arguments: list[str], fetch_run_dirs: list[str]) -> int:
    """Run `scripts/<script> args` on the owner's Cloud Run job and wait. Returns gcloud's exit code."""
    job_arguments = []
    for run_dir in fetch_run_dirs:
        job_arguments += ["--fetch", f"{owner}/{run_dir}"]
    job_arguments += script_arguments
    delimiter_and_arguments = "^|^" + "|".join(job_arguments)
    command = gcloud_command(
        "run", "jobs", "execute", f"shipwright-{owner}", f"--region={REGION}", "--wait",
        f"--args={delimiter_and_arguments}",
    )
    logger.info("cloud job", job=f"shipwright-{owner}", arguments=job_arguments)
    return subprocess.run(command).returncode


def pull_stage_output(owner: str, run_dir_name: str) -> Path:
    """Copy the run dir the cloud job uploaded to runs/, replacing an older local copy."""
    gcs_runs.pull_runs([owner], ROOT / "runs", [run_dir_name], replace_existing=True)
    return ROOT / "runs" / run_dir_name


def read_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def cloud_stage_arguments(stage: str, options: argparse.Namespace) -> tuple[list[str], list[str], str]:
    """(script arguments, run dirs to fetch first, run dir this stage writes)."""
    cap = str(BUDGETS[stage].cap_usd)
    baseline_name = baseline_dir_name(options.round)
    if stage == "baseline":
        arguments = [
            "baseline.py", "--task-split", "train", "--num-tasks", "0", "--trials", "1",
            "--user-model", options.user_model, "--max-concurrency", str(options.max_concurrency),
            "--max-cost-usd", cap, "--output-dir", f"runs/{baseline_name}",
        ]
        if options.dry_run:
            arguments += ["--dry-run"]
        return arguments, [], baseline_name
    if stage == "classify":
        arguments = ["classify.py", f"runs/{baseline_name}"]
        if options.dry_run:
            arguments.append("--no-llm")
        return arguments, [baseline_name], baseline_name
    teacher_name = teacher_dir_name(options.round)
    arguments = [
        "teacher_runs.py", "--teacher-model", options.teacher_model, "--user-model", options.user_model,
        "--from-run", f"runs/{baseline_name}", "--trials=3", "--max-concurrency", str(options.max_concurrency),
        "--max-cost-usd", cap, "--output-dir", f"runs/{teacher_name}",
    ]
    if options.dry_run:
        arguments.append("--dry-run")
    return arguments, [baseline_name], teacher_name


def run_cloud_stage(stage: str, options: argparse.Namespace) -> dict:
    check_cloud_job_runs_this_commit(options.owner)
    script_arguments, fetch_run_dirs, output_dir_name = cloud_stage_arguments(stage, options)
    exit_code = run_cloud_job(options.owner, script_arguments, fetch_run_dirs)
    output_dir = pull_stage_output(options.owner, output_dir_name)
    summary = read_summary(output_dir)
    cost = (summary.get("cost_estimate_usd") or {}).get("total")
    if stage == "classify":
        cost = None  # classify.py does not meter its calls, and summary.json here is the baseline's
    entry = {
        "run_dir": str(output_dir.relative_to(ROOT)),
        "exit_code": exit_code,
        "actual_usd": cost,
        "num_runs": summary.get("num_runs"),
        "num_passed": summary.get("num_passed"),
        "num_infra_errors": summary.get("num_infra_errors"),
        "cost_cap_hit": summary.get("cost_cap_hit"),
    }
    if stage == "classify":
        failure_summary_path = output_dir / "failure_summary.json"
        if failure_summary_path.exists():
            failure_summary = json.loads(failure_summary_path.read_text())
            entry["num_failed"] = failure_summary.get("num_failed")
            entry["per_route"] = failure_summary.get("per_route")
            entry["per_label"] = failure_summary.get("per_label")
    if stage == "teacher":
        kept_path = output_dir / "kept.jsonl"
        if kept_path.exists():
            entry["kept_traces"] = len([line for line in kept_path.read_text().splitlines() if line.strip()])
    return entry


# ---------------------------------------------------------------- local and GPU stages

def run_data_stage(options: argparse.Namespace) -> dict:
    teacher_dir = ROOT / "runs" / teacher_dir_name(options.round)
    output_dir = ROOT / "runs" / sft_dir_name(options.round)
    command = [
        sys.executable, str(ROOT / "scripts" / "build_sft.py"), "--teacher", str(teacher_dir),
        "--output-dir", str(output_dir), "--push", "--revision", f"round-{options.round}",
    ]
    exit_code = subprocess.run(command).returncode
    stats_path = output_dir / "sft_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    return {
        "run_dir": str(output_dir.relative_to(ROOT)),
        "exit_code": exit_code,
        "actual_usd": 0.0,
        "examples": stats.get("examples"),
        "tasks": len(stats.get("tasks") or []),
    }


def run_gpu_stage(stage: str, options: argparse.Namespace) -> dict:
    revision = f"round-{options.round}"
    if stage == "train":
        command = [sys.executable, str(ROOT / "scripts" / "train_job.py"),
                   "--dataset-revision", revision, "--revision", revision,
                   "--flavor", TRAIN_FLAVOR, "--timeout", GPU_TIMEOUT]
    else:
        command = [sys.executable, str(ROOT / "scripts" / "gate_job.py"), "--round", str(options.round),
                   "--current", "configs/systems/base.yaml",
                   "--candidate", f"configs/systems/r{options.round}-weight.yaml",
                   "--user-model", options.user_model,
                   "--flavor", GATE_FLAVOR, "--timeout", GPU_TIMEOUT]
    if options.launch:
        command.append("--launch")
    exit_code = subprocess.run(command).returncode
    status = "launched" if options.launch else "planned"
    # A GPU job's cost is billed by Hugging Face while it runs: actual_usd stays empty until you read it there.
    return {"exit_code": exit_code, "status": status, "actual_usd": None}


# ---------------------------------------------------------------- main

def print_plan(options: argparse.Namespace, manifest: dict) -> None:
    print(f"Round {options.round}, owner {options.owner}, budget ${options.round_budget_usd:.2f}\n")
    total_estimate = 0.0
    total_cap = 0.0
    for stage in STAGE_ORDER:
        budget = BUDGETS[stage]
        total_estimate += budget.estimate_usd
        total_cap += budget.cap_usd
        done = manifest["stages"].get(stage)
        mark = "done" if done and done.get("exit_code") == 0 else "todo"
        print(f"  {mark}  {stage:9} est ${budget.estimate_usd:5.2f}  cap ${budget.cap_usd:5.2f}  {DESCRIPTIONS[stage]}")
    print(f"\n  total   est ${total_estimate:.2f}   cap ${total_cap:.2f}   spent so far ${manifest.get('spent_usd') or 0:.2f}")
    print("\n  Tiny Aya's price on the HF router is an assumption ($0.50 in, $1.50 out per 1M tokens):")
    print("  read the real spend in your HF billing page after the baseline stage.")


def main() -> int:
    options = parse_options()
    manifest = load_manifest(options)
    if options.stage == "plan":
        print_plan(options, manifest)
        return 0

    stage = options.stage
    previous_stage = STAGE_ORDER[STAGE_ORDER.index(stage) - 1] if stage != STAGE_ORDER[0] else None
    if previous_stage and manifest["stages"].get(previous_stage, {}).get("exit_code") != 0:
        raise SystemExit(f"run {previous_stage} first: it has no finished entry in {manifest_path(options.round).name}")
    check_budget(manifest, stage)

    if stage in ("baseline", "classify", "teacher"):
        entry = run_cloud_stage(stage, options)
    elif stage == "data":
        entry = run_data_stage(options)
    else:
        entry = run_gpu_stage(stage, options)

    budget = BUDGETS[stage]
    entry.update(estimate_usd=budget.estimate_usd, cap_usd=budget.cap_usd,
                 finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    manifest["stages"][stage] = entry
    save_manifest(manifest, options.round)
    print(json.dumps({stage: entry}, indent=2))

    actual = entry.get("actual_usd")
    if actual is not None and actual > budget.estimate_usd * 1.5:
        print(f"\nWARNING: {stage} cost ${actual:.2f}, over 1.5x its ${budget.estimate_usd:.2f} estimate.")
    return entry["exit_code"]


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="run_round"))

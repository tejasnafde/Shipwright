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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.control_room import gcs_runs  # noqa: E402
from shipwright.llm.vertex import ROUTER_USER_MODEL  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("run_round")

REGION = "asia-south1"
IMAGE = "asia-south1-docker.pkg.dev/lexsi-hackathon/shipwright/harness"
STAGE_ORDER = ["baseline", "classify", "teacher", "data", "train", "gate"]
# Runs on its own, any time: the teacher on the held-out tasks, so the gate result can be read against it.
SIDE_STAGES = ["ceiling", "smoke"]
# A stage that finished with more infra errors than this share of its runs, or a dataset of fewer
# examples than this, stops the round before the next stage (override with --force).
MAX_INFRA_ERROR_RATE = 0.25
MIN_SFT_EXAMPLES = 40
IMAGE_POLL_SECONDS = 30


@dataclass
class StageBudget:
    estimate_usd: float  # what we expect it to cost
    cap_usd: float       # the most it may cost: --max-cost-usd for API stages, price x timeout for GPU jobs


# The team has $10 in total (2026-11-12), so every cap is small and the caps add up to well under it.
# Estimates come from the round 1 runs (runs/r1-baseline-train: $1.49 for 74 runs, runs/r1-teacher-t1:
# $0.53 for 54 runs, so 3 trials of about 54 tasks is about $1.60). GPU caps are price x timeout:
# train L40S $1.80/h x 1h, gate A10G large $1.50/h x 1h plus about $0.60 of user simulator calls.
TRAIN_TIMEOUT = "1h"
GATE_TIMEOUT = "90m"
# 185 examples x 3 epochs is about 555 sequences of 5.5k tokens: an L4 (24 GB, $0.80/h) risks the 1 h limit and the
# 262k-token vocabulary logits (docs/pipeline/training.md); an L40S (48 GB, $1.80/h) is faster and has the memory.
TRAIN_FLAVOR = "l40sx1"
GATE_FLAVOR = "a10g-large"
# The gate's user simulator is an HF router model: a Vertex token minted at launch lasts about 1 hour
# with no refresh (AGENTS.md traps), and the gate (120 tau2 runs, BFCL, model load) can run longer, which
# would turn runs into infra errors and reject for "too few valid runs". Both systems use the same simulator.
GATE_USER_MODEL = ROUTER_USER_MODEL
BUDGETS = {
    "baseline": StageBudget(estimate_usd=1.5, cap_usd=2.0),
    "classify": StageBudget(estimate_usd=0.05, cap_usd=0.30),
    "teacher": StageBudget(estimate_usd=1.6, cap_usd=2.0),
    "data": StageBudget(estimate_usd=0.0, cap_usd=0.0),
    "train": StageBudget(estimate_usd=1.0, cap_usd=1.8),
    "ceiling": StageBudget(estimate_usd=0.5, cap_usd=1.0),
    "smoke": StageBudget(estimate_usd=0.1, cap_usd=0.3),
    "gate": StageBudget(estimate_usd=2.3, cap_usd=3.0),  # A10G $1.50/h x 1.5h = $2.25 cap, plus about $0.60 of simulator
}
DESCRIPTIONS = {
    "baseline": "Tiny Aya (no adapter, no tool patches) on the 74 train tasks, 1 trial",
    "classify": "label and route every failure (AgentTune detector and classifier, our rules)",
    "teacher": "the teacher replays the tool-fix and weight-fix failures (minus tasks it never solved), 3 trials each; passes are kept",
    "data": "build the SFT dataset from the teacher passes and push it to the HF dataset repo",
    "train": "LoRA on Tiny Aya Global with AgentTune (HF Job, L40S, 1h limit)",
    "ceiling": "the teacher on the 20 held-out tasks, 3 trials: the pass rate the student is compared with (evaluation only)",
    "smoke": "5 train tasks, 1 trial, on the options you pass: run it after any change to the user model or provider",
    "gate": "base vs the new adapter on the 20 held-out tasks, k=3, 8 checks (HF Job, A10G, 90 min limit)",
}


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["plan", *STAGE_ORDER, *SIDE_STAGES])
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--user-model", default=ROUTER_USER_MODEL,
                        help="user simulator and judge (default: gpt-oss-120b on the HF router via DeepInfra, the same one the"
                        " gate uses; vertex/openai/gpt-oss-120b-maas is the Vertex alternative)")
    parser.add_argument("--teacher-model", required=True, help="e.g. vertex/deepseek-ai/deepseek-v4.1-flash-maas")
    parser.add_argument("--owner", default="ishank", help="whose Cloud Run job runs the cloud stages")
    parser.add_argument("--round-budget-usd", type=float, default=8.0, help="no stage starts if its cap passes this total")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--skip-unsolved-from", nargs="*", default=[], metavar="RUN_DIR_NAME",
                        help="teacher: earlier teacher run dirs (e.g. r2-teacher); leave out the tasks they never solved")
    parser.add_argument("--extra-teacher-runs", nargs="*", default=[], metavar="RUN_DIR_NAME",
                        help="data: earlier teacher run dirs of the same teacher model (e.g. r1-teacher-t1 r1-teacher-all-t2)"
                        " whose passing traces join this round's in the SFT data")
    parser.add_argument("--skip-solved-from", nargs="*", default=[], metavar="RUN_DIR_NAME",
                        help="teacher: earlier teacher run dirs; leave out the tasks they already solved (use the same"
                        " list as --extra-teacher-runs)")
    parser.add_argument("--only-solved-from", nargs="*", default=[], metavar="RUN_DIR_NAME",
                        help="teacher (harvest): a bare name is one of --owner's run dirs, owner/dir another owner's; play only the tasks these earlier teacher runs solved, more trials each")
    parser.add_argument("--min-teacher-rate", type=float, default=0.0,
                        help="teacher (harvest): only tasks whose pooled teacher pass rate is at least this; 0.5 = the gap tasks")
    parser.add_argument("--teacher-trials", type=int, default=3, help="teacher: trials per task (5 for a harvest)")
    parser.add_argument("--wait-for-image-minutes", type=int, default=20,
                        help="cloud stages: how long to wait for the Cloud Build to finish and the job to be repointed")
    parser.add_argument("--force", action="store_true", help="skip the health checks on the previous stage")
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


# Which AgentTune and AuditKIT parts each stage runs (docs/pipeline/baseline.md has the full table). Written into
# the manifest with the installed versions, so a reader of runs/round-N.json can see where the libraries are used.
LIBRARY_USE_BY_STAGE = {
    "classify": ["agenttune FailureDetector", "agenttune FailureClassifier"],
    "train": ["agenttune RetrainingTrigger (the pre-train decision)", "agenttune create_distill_trainer (in the HF Job)"],
    "gate": ["auditkit AgentEvalRunner (pass^1, pass^k, diagnostics)", "auditkit BFCL", "agenttune DeploymentGate"],
}


def library_versions() -> dict:
    from importlib import metadata

    versions = {}
    for library in ("agenttune", "auditkit"):
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            versions[library] = "not installed here (the cloud image and HF jobs have it)"
    return versions


def save_manifest(manifest: dict, round_number: int) -> None:
    """Write the manifest with its spend split in two: metered (the stage measured its own cost) and
    estimated (a GPU job or an unmetered stage: its estimate, until you read the real bill)."""
    metered = 0.0
    estimated = 0.0
    for stage_entry in manifest["stages"].values():
        if stage_entry.get("actual_usd") is not None:
            stage_entry["cost_basis"] = "metered"
            metered += stage_entry["actual_usd"]
        else:
            stage_entry["cost_basis"] = "estimated"
            estimated += stage_entry.get("estimate_usd") or 0.0
    # Spend that belongs to no stage: failed attempts, smoke tests. Add one by hand with what it was and where the number
    # came from (for an HF Job: runtime from `hf jobs inspect` x the flavor's price).
    for other_cost in manifest.get("other_costs", []):
        metered += other_cost["usd"]
    manifest["spent_metered_usd"] = round(metered, 4)
    manifest["spent_estimated_usd"] = round(estimated, 4)
    manifest["spent_usd"] = round(metered + estimated, 4)
    manifest["libraries"] = {"versions": library_versions(), "used_by_stage": LIBRARY_USE_BY_STAGE}
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


def job_image_digest(owner: str) -> str:
    job = run_command(gcloud_command(
        "run", "jobs", "describe", f"shipwright-{owner}", f"--region={REGION}", "--format=json"))
    job_image = json.loads(job.stdout)["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]["image"]
    return job_image.split("@")[-1]


def commit_image_digest(commit: str) -> str:
    """The digest of the image tagged with the commit, or "" while the Cloud Build has not pushed it."""
    tag = run_command(gcloud_command(
        "artifacts", "docker", "images", "describe", f"{IMAGE}:{commit}", "--format=value(image_summary.digest)"))
    return tag.stdout.strip()


def check_cloud_job_runs_this_commit(owner: str, wait_minutes: int) -> None:
    """The job must run the image built from HEAD, or the round would run old code.

    The Cloud Build tags the image first and repoints the job in its last step, so right after a push the
    two disagree for a few minutes: wait for them (up to ``wait_minutes``) instead of failing."""
    head_sha = run_command(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).stdout.strip()
    if run_command(["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.strip():
        raise SystemExit("uncommitted changes: commit and push first, so the image matches what you ran")
    unpushed = run_command(["git", "-C", str(ROOT), "rev-list", "--count", "@{upstream}..HEAD"])
    if unpushed.returncode != 0 or unpushed.stdout.strip() != "0":
        raise SystemExit(f"commit {head_sha} is not pushed: push it so Cloud Build builds its image")

    deadline = time.monotonic() + wait_minutes * 60
    while True:
        head_digest = commit_image_digest(head_sha)
        job_digest = job_image_digest(owner)
        if head_digest and head_digest == job_digest:
            logger.info("cloud job runs this commit", commit=head_sha, digest=job_digest[:19])
            return
        if head_digest:
            reason = f"shipwright-{owner} runs {job_digest[:19]}, not the image of {head_sha}"
        else:
            reason = f"no image for commit {head_sha} yet"
        if time.monotonic() >= deadline:
            raise SystemExit(f"{reason} after waiting {wait_minutes} minutes: check the Cloud Build")
        logger.info("waiting for the Cloud Build", reason=reason)
        time.sleep(IMAGE_POLL_SECONDS)


def check_previous_stage_is_healthy(manifest: dict, stage: str, previous_stage: str) -> None:
    """Stop the round when the stage before this one finished but its output is not worth building on."""
    previous_entry = manifest["stages"][previous_stage]
    runs = previous_entry.get("num_runs") or 0
    infra_errors = previous_entry.get("num_infra_errors") or 0
    if runs and infra_errors / runs > MAX_INFRA_ERROR_RATE:
        raise SystemExit(
            f"{previous_stage}: {infra_errors} of {runs} runs were infra errors "
            f"(over {MAX_INFRA_ERROR_RATE:.0%}): fix the cause, or pass --force"
        )
    examples = previous_entry.get("examples")
    if previous_stage == "data" and examples is not None and examples < MIN_SFT_EXAMPLES:
        raise SystemExit(
            f"the SFT data has {examples} examples (under {MIN_SFT_EXAMPLES}): too little to train on. "
            f"Add earlier teacher runs with --extra-teacher-runs at the data stage, or pass --force"
        )


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
        job_arguments.append(f"--fetch={run_dir}")  # gcloud --args rejects a repeated "--fetch"
    job_arguments += script_arguments
    repeated_values = sorted({value for value in job_arguments if job_arguments.count(value) > 1})
    if repeated_values:
        raise SystemExit(f"gcloud --args rejects a repeated value {repeated_values}: write it as --flag=value, or pass it once")
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


def qualified_run_name(owner: str, run_name: str) -> str:
    """"owner/dir" for a run dir in the bucket: a bare name belongs to ``owner``, "tejas/dir" names another owner."""
    if "/" in run_name:
        return run_name
    return f"{owner}/{run_name}"


def local_run_name(run_name: str) -> str:
    """The directory name under runs/ (a cloud job fetches "owner/dir" into runs/dir)."""
    return run_name.rsplit("/", 1)[-1]


def cloud_stage_arguments(stage: str, options: argparse.Namespace) -> tuple[list[str], list[str], str]:
    """(script arguments, run dirs to fetch first as "owner/dir", run dir this stage writes)."""
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
        return arguments, [qualified_run_name(options.owner, baseline_name)], baseline_name
    if stage == "smoke":
        smoke_name = f"r{options.round}-smoke"
        arguments = [
            "baseline.py", "--task-split", "train", "--num-tasks", "5", "--trials", "1",
            "--user-model", options.user_model, "--max-concurrency", str(options.max_concurrency),
            "--max-cost-usd", cap, "--output-dir", f"runs/{smoke_name}",
        ]
        if options.dry_run:
            arguments.append("--dry-run")
        return arguments, [], smoke_name
    if stage == "ceiling":
        ceiling_name = f"r{options.round}-ceiling"
        arguments = [
            "teacher_runs.py", "--ceiling", "--teacher-model", options.teacher_model, "--user-model", options.user_model,
            "--trials=3", "--max-concurrency", str(options.max_concurrency),
            "--max-cost-usd", cap, "--output-dir", f"runs/{ceiling_name}",
        ]
        if options.dry_run:
            arguments.append("--dry-run")
        return arguments, [], ceiling_name
    teacher_name = teacher_dir_name(options.round)
    skipped_run_names = list(options.skip_unsolved_from)
    arguments = [
        "teacher_runs.py", "--teacher-model", options.teacher_model, "--user-model", options.user_model,
        "--from-run", f"runs/{baseline_name}", f"--trials={options.teacher_trials}", "--max-concurrency", str(options.max_concurrency),
        "--max-cost-usd", cap, "--output-dir", f"runs/{teacher_name}",
    ]
    if skipped_run_names:
        arguments += ["--skip-unsolved-from", *[f"runs/{local_run_name(name)}" for name in skipped_run_names]]
    harvest_run_names = list(options.only_solved_from)
    if harvest_run_names:
        arguments += ["--only-solved-from", *[f"runs/{local_run_name(name)}" for name in harvest_run_names]]
        arguments += ["--min-teacher-rate", str(options.min_teacher_rate)]
    solved_run_names = list(options.skip_solved_from)
    if solved_run_names:
        arguments += ["--skip-solved-from", *[f"runs/{local_run_name(name)}" for name in solved_run_names]]
    if options.dry_run:
        arguments.append("--dry-run")
    fetch_run_dirs = list(dict.fromkeys(
        qualified_run_name(options.owner, name)
        for name in [baseline_name, *skipped_run_names, *harvest_run_names, *solved_run_names]
    ))
    return arguments, fetch_run_dirs, teacher_name


def run_cloud_stage(stage: str, options: argparse.Namespace) -> dict:
    check_cloud_job_runs_this_commit(options.owner, options.wait_for_image_minutes)
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
    if stage == "ceiling":
        entry["teacher_pass_rate"] = summary.get("pass_rate_excluding_infra_errors")
    if entry["num_runs"]:
        entry["infra_error_rate"] = round((entry["num_infra_errors"] or 0) / entry["num_runs"], 3)
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
    command = [sys.executable, str(ROOT / "scripts" / "build_sft.py")]
    if teacher_dir.exists():  # a round that only pools earlier teacher runs has no teacher run of its own
        command += ["--teacher", str(teacher_dir)]
    command += [
        "--output-dir", str(output_dir), "--push", "--revision", f"round-{options.round}",
        "--only-model", options.teacher_model,
    ]
    for extra_name in options.extra_teacher_runs:
        command += ["--teacher", str(ROOT / "runs" / local_run_name(extra_name))]
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


def current_system_path() -> str:
    """The deployed system (scripts/promote.py writes it after a promotion), else the base model."""
    if (ROOT / "configs" / "systems" / "current.yaml").exists():
        return "configs/systems/current.yaml"
    return "configs/systems/base.yaml"


def check_retraining_trigger(options: argparse.Namespace) -> dict:
    """AgentTune's RetrainingTrigger (scripts/round_status.py, offline) decides whether this round trains.

    The weight_fix failures of the baseline go through its buffer; a failure has a correction when the SFT data holds
    a teacher example for the same task. A "no" stops the round before any GPU spend (``--force`` overrides)."""
    baseline_directory = ROOT / "runs" / baseline_dir_name(options.round)
    command = [sys.executable, str(ROOT / "scripts" / "round_status.py"), str(baseline_directory),
               "--sft", str(ROOT / "runs" / sft_dir_name(options.round) / "sft.jsonl")]
    earlier_baseline = ROOT / "runs" / baseline_dir_name(options.round - 1)
    if earlier_baseline.exists():
        command += ["--reward-baseline", str(earlier_baseline)]
    subprocess.run(command, capture_output=True, text=True)
    status_path = baseline_directory / "round_status.json"
    if not status_path.exists():
        raise SystemExit("the AgentTune retraining trigger wrote no round_status.json: run scripts/round_status.py by hand")
    round_status = json.loads(status_path.read_text())
    trigger = {key: round_status.get(key) for key in
               ("should_train", "reason", "num_weight_fix_failures", "num_with_correction")}
    if not trigger["should_train"] and not options.force:
        raise SystemExit(f"AgentTune's RetrainingTrigger says do not train yet ({trigger['reason']}): pass --force to train anyway")
    return trigger


def run_gpu_stage(stage: str, options: argparse.Namespace) -> dict:
    revision = f"round-{options.round}"
    retrain_trigger = None
    if stage == "train":
        retrain_trigger = check_retraining_trigger(options)
        command = [sys.executable, str(ROOT / "scripts" / "train_job.py"),
                   "--dataset-revision", revision, "--revision", revision,
                   "--flavor", TRAIN_FLAVOR, "--timeout", TRAIN_TIMEOUT]
    else:
        command = [sys.executable, str(ROOT / "scripts" / "gate_job.py"), "--round", str(options.round),
                   "--current", current_system_path(),
                   "--candidate", f"configs/systems/r{options.round}-weight.yaml",
                   "--user-model", GATE_USER_MODEL,
                   "--flavor", GATE_FLAVOR, "--timeout", GATE_TIMEOUT]
    if options.launch:
        command.append("--launch")
    exit_code = subprocess.run(command).returncode
    status = "launched" if options.launch else "planned"
    # A GPU job's cost is billed by Hugging Face while it runs: actual_usd stays empty until you read it there.
    entry = {"exit_code": exit_code, "status": status, "actual_usd": None}
    if retrain_trigger is not None:
        entry["retrain_trigger"] = retrain_trigger
    return entry


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
    for side_stage in SIDE_STAGES:
        budget = BUDGETS[side_stage]
        done = manifest["stages"].get(side_stage)
        mark = "done" if done and done.get("exit_code") == 0 else "todo"
        print(f"  {mark}  {side_stage:9} est ${budget.estimate_usd:5.2f}  cap ${budget.cap_usd:5.2f}  (any time) {DESCRIPTIONS[side_stage]}")
    print(f"\n  total   est ${total_estimate:.2f}   cap ${total_cap:.2f}   spent so far ${manifest.get('spent_usd') or 0:.2f}"
          f" (metered ${manifest.get('spent_metered_usd') or 0:.2f}, estimated ${manifest.get('spent_estimated_usd') or 0:.2f})")
    print("\n  Tiny Aya's price on the HF router is an assumption ($0.50 in, $1.50 out per 1M tokens):")
    print("  read the real spend in your HF billing page after the baseline stage.")


def main() -> int:
    options = parse_options()
    manifest = load_manifest(options)
    if options.stage == "plan":
        print_plan(options, manifest)
        return 0

    stage = options.stage
    if stage in SIDE_STAGES:
        previous_stage = None
    else:
        previous_stage = STAGE_ORDER[STAGE_ORDER.index(stage) - 1] if stage != STAGE_ORDER[0] else None
    pooled_data_only = stage == "data" and options.extra_teacher_runs and previous_stage == "teacher"
    if pooled_data_only and "teacher" not in manifest["stages"]:
        previous_stage = "classify"  # no teacher stage this round: the earlier teacher runs are the data
    if previous_stage and manifest["stages"].get(previous_stage, {}).get("exit_code") != 0:
        raise SystemExit(f"run {previous_stage} first: it has no finished entry in {manifest_path(options.round).name}")
    if previous_stage and not options.force:
        check_previous_stage_is_healthy(manifest, stage, previous_stage)
    check_budget(manifest, stage)

    if stage in ("baseline", "classify", "teacher", "ceiling", "smoke"):
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
    sys.exit(log.run_script(main, step="run_round", meta=True))

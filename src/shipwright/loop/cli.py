"""shipwright: the one command that runs the self-healing loop (docs/pipeline/loop.md).

    shipwright doctor [--cloud]                     is this machine and account ready? (free; --cloud about $0.1)
    shipwright plan   --config configs/rounds/X.yaml     what a run would do and may spend (free)
    shipwright run    --config X --dry-run          the whole loop here, offline: fake models, fake HF jobs (free, CI)
    shipwright run    --config X --launch           ONE Cloud Run execution runs the loop, unattended (spends)
    shipwright run    --config X --here             what that execution runs (refuses to run on a laptop)
    shipwright stage  ceiling|smoke --config X [--dry-run|--launch|--here]   a side stage, by hand
    shipwright status [--round N]                   the round manifests: stages, spend, decision
    shipwright show | rollback | promote <gate_decision.json>   the deployed system (ledger on the Hub, with rollback)

``run`` goes round after round: baseline -> classify -> teacher -> data -> train -> gate, then promote, or read why the
gate refused and run the next remedy (shipwright.loop.recipe), until a promotion, a failed stage, the spend cap or no
remedy left. Every stage is written to runs/rN-round/round.json as it finishes; the same command resumes.
Inside Cloud Run the image runs ``scripts/loop.py`` (this module) through scripts/job_entry.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shipwright import cloud
from shipwright.loop import config as config_module
from shipwright.loop import controller
from shipwright.loop import recipe as recipe_module
from shipwright.loop.stages import Stages, hub_file
from shipwright.loop.workspace import Workspace
from shipwright.observability import log

ROOT = cloud.ROOT
logger = log.get_logger("loop")
IMAGE_POLL_SECONDS = 30
IMAGE_WAIT_MINUTES = 20


# ---------------------------------------------------------------- the cloud image must be this commit

def run_quietly(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


def check_cloud_job_runs_this_commit(owner: str, wait_minutes: int = IMAGE_WAIT_MINUTES) -> None:
    """The job must run the image built from HEAD, or the loop would run old code. Right after a push the Cloud Build
    tags the image first and repoints the job last, so wait for the two to agree instead of failing."""
    head_sha = run_quietly(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).stdout.strip()
    if run_quietly(["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.strip():
        raise SystemExit("uncommitted changes: commit and push first, so the image matches what you run")
    unpushed = run_quietly(["git", "-C", str(ROOT), "rev-list", "--count", "@{upstream}..HEAD"])
    if unpushed.returncode != 0 or unpushed.stdout.strip() != "0":
        raise SystemExit(f"commit {head_sha} is not pushed: push it so Cloud Build builds its image")
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        head_digest = run_quietly(cloud.gcloud_command(
            "artifacts", "docker", "images", "describe", f"{cloud.IMAGE}:{head_sha}",
            "--format=value(image_summary.digest)")).stdout.strip()
        job = run_quietly(cloud.gcloud_command("run", "jobs", "describe", f"shipwright-{owner}",
                                               f"--region={cloud.REGION}", "--format=json"))
        job_image = json.loads(job.stdout)["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]["image"]
        job_digest = job_image.split("@")[-1]
        if head_digest and head_digest == job_digest:
            logger.info("cloud job runs this commit", commit=head_sha)
            return
        if head_digest:
            reason = f"shipwright-{owner} runs {job_digest[:19]}, not the image of {head_sha}"
        else:
            reason = f"no image for commit {head_sha} yet"
        if time.monotonic() >= deadline:
            raise SystemExit(f"{reason} after waiting {wait_minutes} minutes: check the Cloud Build")
        logger.info("waiting for the Cloud Build", reason=reason)
        time.sleep(IMAGE_POLL_SECONDS)


def launch_in_the_cloud(config: dict, job_arguments: list[str]) -> int:
    """One execution of shipwright-<owner> running ``scripts/loop.py <job_arguments> --here``. It does not wait."""
    check_cloud_job_runs_this_commit(config["owner"])
    command = cloud.gcloud_command("run", "jobs", "execute", f"shipwright-{config['owner']}", f"--region={cloud.REGION}",
                                   f"--task-timeout={(config.get('cloud') or {}).get('task_timeout', '8h')}",
                                   "--args=^|^" + "|".join(["loop.py", *job_arguments, "--here"]))
    print("starting:", " ".join(job_arguments))
    exit_code = subprocess.run(command).returncode
    print(f"\nFollow it: scripts/why.py <execution name>; results in gs://{cloud.bucket()}/{config['owner']}/ "
          f"(r{config['round']}-round/, chain-r{config['round']}/) and `shipwright status` after `scripts/pull_runs.py`.")
    return exit_code


def check_where(here: bool, dry_run: bool, config: dict) -> None:
    if dry_run:
        return
    if not cloud.in_the_image():
        raise SystemExit("a real run happens in the cloud, not on a laptop: use --launch (or --dry-run here)")
    pinned = config.get("git_sha")
    running = os.environ.get("SHIPWRIGHT_GIT_SHA", "")
    if pinned and not (running.startswith(pinned) or pinned.startswith(running or "-")):
        raise SystemExit(f"the config pins commit {pinned}, this image is {running or 'unknown'}")


# ---------------------------------------------------------------- the loop

def resolved_config(config: dict, config_path: Path, rounds: int, dry_run: bool) -> dict:
    """The config as run, plus what is not in the file: the code, the image and the frozen files' hashes."""
    if cloud.in_the_image():
        code_sha = os.environ.get("SHIPWRIGHT_GIT_SHA", "")[:12]
    else:
        code_sha = run_quietly(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"]).stdout.strip()
    from importlib import metadata

    versions = {}
    for library in ("agenttune", "auditkit"):
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            versions[library] = "not installed here"
    return {**config, "rounds": rounds, "dry_run": dry_run, "config_file": str(config_path),
            "config_sha": config_module.file_sha(config_path),
            "resolved": {"code_sha": code_sha, "libraries": versions,
                         "image": os.environ.get("SHIPWRIGHT_IMAGE_DIGEST") or os.environ.get("SHIPWRIGHT_IMAGE"),
                         "frozen_sha": {name: config_module.file_sha(ROOT / path)
                                        for name, path in config["frozen"].items()}}}


def start_point(config: dict, dry_run: bool) -> tuple[dict | None, int | None]:
    """(the gate decision the chain continues from, the round whose adapter that candidate serves), or (None, None)."""
    start = config.get("start")
    if not start:
        return None, None
    candidate = yaml.safe_load((ROOT / start["candidate"]).read_text())
    revision = str((candidate.get("adapter") or {}).get("revision") or "")
    adapter_round = int(revision.removeprefix("round-")) if revision.removeprefix("round-").isdigit() else None
    if start.get("regate"):
        return None, adapter_round  # the old decision is not used: the first round measures again
    if dry_run:
        decision = {"decision": "reject", "checks": [{"name": "pass_k", "passed": False}],
                    "metrics_current": {"label_rates": {}}, "metrics_candidate": {"label_rates": {}}}
        return decision, adapter_round
    decision_file = hub_file(f"gate/round-{start['from_round']}/{candidate['name']}", "gate_decision.json")
    if decision_file is None:
        raise SystemExit(f"start: no gate decision for round {start['from_round']} ({candidate['name']}) on the Hub")
    return json.loads(decision_file.read_text()), adapter_round


def run_loop(config: dict, config_path: Path, rounds: int, dry_run: bool, runs_directory: Path | None = None,
             redo: list[str] | None = None) -> int:
    if dry_run:
        config = config_module.dry_run(config)
    workspace = Workspace(config, runs_directory, dry_run)
    resolved = resolved_config(config, config_path, rounds, dry_run)

    def promote(round_number: int, decision: dict) -> None:
        decision_file = workspace.path(round_number, "gate") / "gate_decision.json"
        workspace.promote(decision, str(decision_file))
        round_directory = workspace.path(round_number, "round")
        shutil.copy(workspace.ledger_path(), round_directory / "ledger.jsonl")
        workspace.upload(round_directory)

    def write_next_round(round_number: int, decision: dict, chosen: recipe_module.Recipe) -> None:
        directory = workspace.path(round_number, "round")
        directory.mkdir(parents=True, exist_ok=True)
        next_round = {"rejected_round": round_number, "failed_checks": sorted(recipe_module.failed_check_names(decision)),
                      "checks": decision.get("checks"), "candidate": f"r{round_number}-weight",
                      "next_recipe": recipe_module.recipe_to_dict(chosen)}
        (directory / "next_round.json").write_text(json.dumps(next_round, indent=2))

    def load_manifest(round_number: int) -> dict | None:
        """A resumed round skips its finished stages, except the ones --redo names (first round only)."""
        manifest = workspace.load_manifest(round_number)
        if manifest is not None and round_number == config["round"]:
            for stage in redo or []:
                manifest["stages"].pop(stage, None)
        return manifest

    hooks = controller.Hooks(
        stage_functions=lambda round_number, state: Stages(config, workspace, round_number, state, dry_run).functions(),
        save_manifest=workspace.save_manifest, load_manifest=load_manifest, promote=promote,
        write_next_round=write_next_round, log=lambda message: logger.info(message))
    start_decision, start_adapter_round = start_point(config, dry_run)
    result = controller.run_chain(config, rounds, hooks, resolved, start_decision, start_adapter_round)

    chain_directory = workspace.runs / f"{config.get('name_prefix', '')}chain-r{config['round']}"
    chain_directory.mkdir(parents=True, exist_ok=True)
    report = {"status": result.status, "reason": result.reason, "spent_usd": result.spent_usd,
              "total_cap_usd": config["spend"]["total"], "config": resolved,
              "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rounds": [{"round": entry.round_number, "status": entry.status, "reason": entry.reason,
                          "spent_usd": round(entry.spent_usd, 4),
                          "manifest": f"runs/{workspace.path(entry.round_number, 'round').name}/round.json"}
                         for entry in result.rounds]}
    (chain_directory / "chain.json").write_text(json.dumps(report, indent=2, default=str))
    workspace.upload(chain_directory)
    print(f"\nLoop: {result.status.upper()} ({result.reason}). Spent ${result.spent_usd:.2f} of "
          f"${config['spend']['total']:.2f}. Summary: {chain_directory}/chain.json")
    if result.status == controller.STOPPED:
        return 1
    return 0


def run_side_stage(config: dict, stage: str, dry_run: bool) -> int:
    if dry_run:
        config = config_module.dry_run(config)
    workspace = Workspace(config, None, dry_run)
    state = controller.ChainState(recipe=controller.starting_recipe(config))
    entry = Stages(config, workspace, config["round"], state, dry_run).functions()[stage]()
    print(json.dumps(entry, indent=2, default=str))
    output = entry.get("run_dir")
    if output:
        workspace.upload(ROOT / output)
    return entry["exit_code"]


def run_doctor(cloud_check: bool = False) -> int:
    """scripts/doctor.py: every integration that broke a real run, checked before any spend."""
    command = [sys.executable, str(ROOT / "scripts" / "doctor.py")]
    if cloud_check:
        command.append("--cloud")
    return subprocess.run(command).returncode


# ---------------------------------------------------------------- read-only commands

def print_plan(config: dict, rounds: int, launching: bool = False) -> None:
    caps = config["spend"]
    last = f" to {config['round'] + rounds - 1}" if rounds > 1 else ""
    print(f"Round {config['round']}{last}, owner {config['owner']}, total cap ${caps['total']:.2f}")
    if config.get("start"):
        print(f"  continues from the gate of round {config['start']['from_round']} ({config['start']['candidate']})")
    else:
        print("  first round: " + ", ".join(stage for stage in controller.FRESH_STAGES
                                            if stage != "bfcl_screen" or config["gate"].get("screen_bfcl")))
    print(f"  starting recipe: {recipe_module.recipe_to_dict(controller.starting_recipe(config))}")
    for stage in config_module.STAGES:
        print(f"  {stage:16} cap ${caps[stage]:.2f}")
    stop = config["stop"]
    print(f"  stop rules: over {stop['max_infra_error_rate']:.0%} infra errors, under {stop['min_sft_examples']} SFT "
          f"examples, {stop['retries']} retry per stage, at most {recipe_module.MAX_CORRECTION_ROUNDS} correction rounds")
    if not launching:
        print("\nPlan only. `run --launch` starts one Cloud Run execution; `run --dry-run` walks the loop here, offline.")


def print_status(round_number: int | None) -> int:
    runs = ROOT / "runs"
    paths = sorted(runs.glob("round-*.json"), key=lambda path: int(path.stem.removeprefix("round-")))
    if round_number is not None:
        paths = [path for path in paths if path.stem == f"round-{round_number}"]
    for path in paths:
        manifest = json.loads(path.read_text())
        decision = ((manifest.get("stages") or {}).get("gate") or {}).get("decision") or {}
        print(f"round {manifest.get('round')}: {manifest.get('status', 'unfinished')}"
              f" ({manifest.get('reason') or decision.get('decision') or ''}), spent ${manifest.get('spent_usd') or 0:.2f}")
        for stage, entry in (manifest.get("stages") or {}).items():
            mark = "ok" if entry.get("ok", entry.get("exit_code") == 0) else "FAIL"
            print(f"    {mark:4} {stage:16} ${entry.get('actual_usd') or 0:.2f}  {entry.get('problem') or entry.get('note') or ''}")
    if not paths:
        print("no round manifests in runs/ (scripts/pull_runs.py fetches cloud runs)")
    return 0


def deployed_command(command: str, decision_path: Path | None) -> int:
    workspace = Workspace({"owner": "ishank"})
    if command == "promote":
        workspace.promote(json.loads(decision_path.read_text()), str(decision_path))
    elif command == "rollback":
        workspace.rollback()
    system = yaml.safe_load(workspace.deployed_system_path().read_text())
    adapter = system.get("adapter")
    detail = f" ({adapter['repo']}@{adapter['revision']})" if adapter else " (the base model)"
    print(f"Deployed: {system['name']}{detail}")
    return 0


# ---------------------------------------------------------------- main

def parse_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="shipwright", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="preflight checks")
    doctor.add_argument("--cloud", action="store_true")
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--owner", default=None, help="run on this person's Cloud Run job (default: the config's)")
        command.add_argument("--rounds", type=int, default=None, help="rounds back to back (default: the config's)")
        if name == "run":
            where = command.add_mutually_exclusive_group(required=True)
            where.add_argument("--dry-run", action="store_true")
            where.add_argument("--launch", action="store_true")
            where.add_argument("--here", action="store_true")
            command.add_argument("--skip-doctor", action="store_true", help="with --launch: skip the free preflight")
            command.add_argument("--redo", nargs="*", default=[], choices=config_module.STAGES,
                                 help="run these finished stages of the first round again")
    stage = commands.add_parser("stage", help="a side stage by hand")
    stage.add_argument("name", choices=config_module.SIDE_STAGES)
    stage.add_argument("--config", type=Path, required=True)
    stage.add_argument("--owner", default=None, help="run on this person's Cloud Run job (default: the config's)")
    where = stage.add_mutually_exclusive_group(required=True)
    where.add_argument("--dry-run", action="store_true")
    where.add_argument("--launch", action="store_true")
    where.add_argument("--here", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--round", type=int, default=None)
    commands.add_parser("show")
    commands.add_parser("rollback")
    promote = commands.add_parser("promote")
    promote.add_argument("gate_decision", type=Path)
    return parser.parse_args(argv)


def dispatch(argv: list[str]) -> int:
    options = parse_options(argv)
    if options.command == "doctor":
        return run_doctor(options.cloud)
    if options.command == "status":
        return print_status(options.round)
    if options.command in ("show", "rollback", "promote"):
        return deployed_command(options.command, getattr(options, "gate_decision", None))

    config = config_module.load(options.config)
    config_path = options.config.resolve()
    owner_arguments = []
    if options.owner:
        if options.owner not in cloud.OWNERS:
            raise SystemExit(f"--owner {options.owner}: not one of {list(cloud.OWNERS)}")
        config["owner"] = options.owner
        owner_arguments = ["--owner", options.owner]
    if options.command == "plan":
        print_plan(config, options.rounds or config.get("rounds", 1))
        return 0
    relative_config = str(config_path.relative_to(ROOT))
    if options.command == "stage":
        if options.launch:
            return launch_in_the_cloud(config, ["stage", options.name, "--config", relative_config, *owner_arguments])
        check_where(options.here, options.dry_run, config)
        return run_side_stage(config, options.name, options.dry_run)
    rounds = options.rounds or config.get("rounds", 1)
    if options.launch:
        print_plan(config, rounds, launching=True)
        if not options.skip_doctor and run_doctor() != 0:
            print("\nThe preflight found problems: nothing was started and nothing was spent. Fix them (or --skip-doctor).")
            return 2
        job_arguments = ["run", "--config", relative_config, f"--rounds={rounds}", *owner_arguments]
        if options.redo:
            job_arguments += ["--redo", *options.redo]
        return launch_in_the_cloud(config, job_arguments)
    check_where(options.here, options.dry_run, config)
    return run_loop(config, config_path, rounds, options.dry_run, redo=options.redo)


def main() -> int:
    return log.run_script(lambda: dispatch(sys.argv[1:]), step="loop", meta=True)

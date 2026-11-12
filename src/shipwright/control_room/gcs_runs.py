"""Copy run dirs from the team's GCS bucket into the local runs/ dir, so control-room can show them.

Cloud Run jobs upload every run dir to gs://lexsi-hackathon-runs/<owner>/<run dir>/. Nothing here
spends money: it only reads from the bucket. A run dir already in runs/ is never overwritten.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

BUCKET = "gs://lexsi-hackathon-runs"
PROJECT = "lexsi-hackathon"
OWNERS = ("tejas", "ishank", "mitushi", "main")
HOMEBREW_GCLOUD = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
NOT_RUN_DIRS = ("jobs",)  # <owner>/jobs/ holds execution logs, not a run dir


@dataclass
class PullResult:
    owner: str
    run_name: str
    status: str  # "pulled", "already local" or "failed"
    detail: str = ""


def gcloud_binary() -> str | None:
    explicit = os.environ.get("SHIPWRIGHT_GCLOUD")
    if explicit:
        return explicit
    found = shutil.which("gcloud")
    if found:
        return found
    if Path(HOMEBREW_GCLOUD).exists():
        return HOMEBREW_GCLOUD
    return None


def gcloud_config(binary: str) -> str:
    """$SHIPWRIGHT_GCLOUD_CONFIG, else `lexsi` if that config exists on this machine, else `personal`."""
    from_env = os.environ.get("SHIPWRIGHT_GCLOUD_CONFIG")
    if from_env:
        return from_env
    listed = subprocess.run([binary, "config", "configurations", "list", "--format=value(name)"],
                            capture_output=True, text=True, timeout=60)
    if "lexsi" in listed.stdout.split():
        return "lexsi"
    return "personal"


def _gcloud_command(binary: str, *storage_arguments: str) -> list[str]:
    return [binary, f"--configuration={gcloud_config(binary)}", f"--project={PROJECT}", "storage", *storage_arguments]


def list_remote_runs(owner: str) -> list[str]:
    """Run dir names under gs://lexsi-hackathon-runs/<owner>/, without the jobs/ dir."""
    binary = gcloud_binary()
    if binary is None:
        raise RuntimeError("gcloud not found: install it or set SHIPWRIGHT_GCLOUD")
    completed = subprocess.run(_gcloud_command(binary, "ls", f"{BUCKET}/{owner}/"),
                               capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 and "matched no objects" in completed.stderr:
        return []  # this owner has uploaded nothing yet
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip()[-400:] or "gcloud storage ls failed")
    run_names = []
    for line in completed.stdout.splitlines():
        line = line.strip().rstrip("/")
        if not line:
            continue
        name = line.rsplit("/", 1)[-1]
        if name not in NOT_RUN_DIRS:
            run_names.append(name)
    return run_names


# Logs of scripts that launch or move other runs (they call log.run_script(meta=True) on a laptop). Pulled into
# runs/_meta/, so they do not clutter the run list.
ORCHESTRATOR_PREFIXES = ("run_round-", "train_job-", "gate_job-", "pull_runs-", "promote-")


def push_run(owner: str, run_directory: Path) -> None:
    """Copy a local run dir to gs://<bucket>/<owner>/<name>, so a Cloud Run job can --fetch it."""
    binary = gcloud_binary()
    if binary is None:
        raise RuntimeError("gcloud not found: install it or set SHIPWRIGHT_GCLOUD")
    copy = subprocess.run(
        _gcloud_command(binary, "cp", "-r", str(run_directory), f"{BUCKET}/{owner}/"),
        capture_output=True, text=True, timeout=1800)
    if copy.returncode != 0:
        raise RuntimeError(f"gcloud cp of {run_directory} failed: {copy.stderr.strip()[-300:]}")


def pull_runs(
    owners: list[str],
    runs_directory: Path,
    only_run_names: list[str] | None = None,
    replace_existing: bool = False,
) -> list[PullResult]:
    """Copy every remote run dir of these owners that is not in runs_directory yet.

    ``replace_existing`` deletes a local copy first: for a run dir that a later step changed in the bucket.
    """
    binary = gcloud_binary()
    if binary is None:
        raise RuntimeError("gcloud not found: install it or set SHIPWRIGHT_GCLOUD")
    runs_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for owner in owners:
        for run_name in list_remote_runs(owner):
            if only_run_names and run_name not in only_run_names:
                continue
            if run_name.startswith(ORCHESTRATOR_PREFIXES):
                destination_directory = runs_directory / "_meta"
            else:
                destination_directory = runs_directory
            destination_directory.mkdir(parents=True, exist_ok=True)
            if replace_existing and (destination_directory / run_name).exists():
                shutil.rmtree(destination_directory / run_name)
            if (destination_directory / run_name).exists():
                results.append(PullResult(owner, run_name, "already local"))
                continue
            copy = subprocess.run(
                _gcloud_command(binary, "cp", "-r", f"{BUCKET}/{owner}/{run_name}", str(destination_directory) + "/"),
                capture_output=True, text=True, timeout=1800)
            if copy.returncode == 0:
                results.append(PullResult(owner, run_name, "pulled"))
            else:
                results.append(PullResult(owner, run_name, "failed", copy.stderr.strip()[-300:]))
    return results

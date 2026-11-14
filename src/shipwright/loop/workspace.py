"""Where a run of the loop keeps its state: run dirs, round manifests and the promotion ledger.

Run dirs live in ``runs/``. Inside the cloud image every finished stage's dir is uploaded to
gs://<bucket>/<owner>/<dir>/ at once, and a dir an earlier execution wrote is fetched from there, so the same command
resumes a dead execution and nothing is lost when the container goes away. On a laptop only a dry run writes here.

The promotion ledger (which adapter is deployed, with rollback) is kept in the public runs dataset on the Hub,
``promotions/ledger.jsonl``, and mirrored to ``promotions/ledger.jsonl`` and ``configs/systems/current.yaml`` in the
repo. Before 2026-11-14 a promotion made in the cloud was written into the image and lost unless someone committed it by
hand. A dry run keeps its own ledger in its round dir.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shipwright import cloud
from shipwright.heal import promotion
from shipwright.loop.config import run_name

ROOT = cloud.ROOT
LEDGER_IN_HUB = "promotions/ledger.jsonl"
LEDGER_IN_REPO = ROOT / "promotions" / "ledger.jsonl"
CURRENT_SYSTEM_IN_REPO = ROOT / "configs" / "systems" / "current.yaml"


def read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def relative(path: Path) -> str:
    """runs/... for a manifest (the tests write under a tmp dir)."""
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


class Workspace:
    def __init__(self, config: dict, runs_directory: Path | None = None, dry_run: bool = False):
        self.config = config
        self.runs = runs_directory or ROOT / "runs"
        self.dry_run = dry_run
        self.owner = config["owner"]

    # ------------------------------------------------------------ run dirs

    def path(self, round_number: int, kind: str) -> Path:
        return self.runs / run_name(self.config, round_number, kind)

    def fetch(self, qualified_name: str) -> bool:
        """Make runs/<dir> exist: "owner/dir" or a bare dir name of this config's owner. False when it is nowhere."""
        if "/" not in qualified_name:
            qualified_name = f"{self.owner}/{qualified_name}"
        local = self.runs / qualified_name.rsplit("/", 1)[-1]
        if local.is_dir():
            return True
        if not cloud.in_the_image():
            return False
        import httpx

        try:
            with httpx.Client(timeout=300) as http_client:
                cloud.fetch_dir(http_client, cloud.bucket(), qualified_name, self.runs)
        except FileNotFoundError:
            return False
        return True

    def upload(self, directory: Path) -> None:
        """In the cloud: copy a finished run dir to the bucket now, so a dead execution loses nothing."""
        if cloud.in_the_image() and directory.is_dir():
            cloud.upload_dir(directory, self.owner)

    # ------------------------------------------------------------ manifests

    def load_manifest(self, round_number: int) -> dict | None:
        """The round's manifest from an earlier execution (resume), with the run dirs of its finished stages."""
        round_directory = self.path(round_number, "round")
        self.fetch(round_directory.name)
        manifest_path = round_directory / "round.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["stages"].values():
            if entry.get("ok") and entry.get("run_dir"):
                self.fetch(Path(entry["run_dir"]).name)
        return manifest

    def save_manifest(self, round_number: int, manifest: dict) -> None:
        round_directory = self.path(round_number, "round")
        round_directory.mkdir(parents=True, exist_ok=True)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        text = json.dumps(manifest, indent=2, default=str)
        (round_directory / "round.json").write_text(text)
        if not self.config.get("name_prefix"):
            (self.runs / f"round-{round_number}.json").write_text(text)  # the control room reads runs/round-N.json
        for entry in manifest["stages"].values():
            if entry.get("ok") and entry.get("run_dir") and not entry.get("uploaded"):
                self.upload(ROOT / entry["run_dir"])
                entry["uploaded"] = cloud.in_the_image()
        self.upload(round_directory)

    # ------------------------------------------------------------ the promotion ledger

    def ledger_path(self) -> Path:
        if self.dry_run:
            return self.runs / f"{self.config.get('name_prefix', '')}promotions" / "ledger.jsonl"
        return LEDGER_IN_REPO

    def pull_ledger(self) -> Path:
        """The Hub's ledger into the repo mirror (it is the record; the repo copy follows it). Dry run: its own."""
        local = self.ledger_path()
        if self.dry_run:
            return local
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        try:
            downloaded = hf_hub_download(cloud.RUNS_REPO, LEDGER_IN_HUB, repo_type="dataset", force_download=True,
                                         token=cloud.hf_token())
        except (EntryNotFoundError, RepositoryNotFoundError):
            return local  # no promotion was ever recorded on the Hub: the repo copy is all there is
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded, local)
        return local

    def push_ledger(self, local: Path, message: str) -> None:
        if self.dry_run:
            return
        from huggingface_hub import HfApi

        HfApi(token=cloud.hf_token()).upload_file(path_or_fileobj=str(local), path_in_repo=LEDGER_IN_HUB,
                                                  repo_id=cloud.RUNS_REPO, repo_type="dataset", commit_message=message)

    def deployed_system_path(self) -> Path:
        """A system config of what is deployed now, for the gate's --current."""
        system = promotion.deployed_system(promotion.read_ledger(self.pull_ledger()))
        if self.dry_run:
            path = self.ledger_path().parent / "current.yaml"
        else:
            path = CURRENT_SYSTEM_IN_REPO
        write_system(path, system, "the deployed system, from the promotion ledger. Do not edit.")
        return path

    def promote(self, decision: dict, decision_file: str) -> dict:
        ledger = self.pull_ledger()
        row = promotion.promote(ledger, decision, decision_file)
        self.push_ledger(ledger, f"promote round {decision['round']}")
        self.deployed_system_path()
        return row

    def rollback(self) -> dict:
        ledger = self.pull_ledger()
        row = promotion.rollback(ledger)
        self.push_ledger(ledger, "rollback")
        self.deployed_system_path()
        return row


def write_system(path: Path, system: dict, why: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Written by shipwright.loop: {why}\n" + yaml.safe_dump(system, sort_keys=False))

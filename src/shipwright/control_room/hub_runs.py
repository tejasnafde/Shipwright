"""Get a round's gate results from the public runs dataset on the Hugging Face Hub into runs/gate-rN.

The gate runs as an HF Job and pushes its results to the Hub (`gate/round-N/<candidate>/`: the decision, both systems'
traces, the metrics, the vLLM log). The control room reads them from `runs/gate-rN`, so it needs them on disk. The
dataset is public: no token is needed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from shipwright.cloud import RUNS_REPO


def pull_gate_results(round_number: int, runs_directory: Path, candidate: str | None = None) -> Path:
    """Download gate/round-N/<candidate>/ into runs/gate-rN (replacing an older copy) and return that dir."""
    from huggingface_hub import snapshot_download

    candidate = candidate or f"r{round_number}-weight"
    source_prefix = f"gate/round-{round_number}/{candidate}"
    with tempfile.TemporaryDirectory() as scratch:
        snapshot_download(RUNS_REPO, repo_type="dataset", allow_patterns=[f"{source_prefix}/*"], local_dir=scratch)
        downloaded = Path(scratch) / source_prefix
        if not (downloaded / "gate_decision.json").exists():
            raise FileNotFoundError(f"no gate results for round {round_number} under {source_prefix} in {RUNS_REPO}")
        destination = runs_directory / f"gate-r{round_number}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(downloaded, destination)
    return destination

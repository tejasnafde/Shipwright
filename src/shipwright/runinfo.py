"""Find a run and read its small files, locally or on GCS (scripts/why.py, scripts/rerun.py).

A run reference is one of:
  runs/<dir>                       a local run dir
  gs://<bucket>/<owner>/<dir>      a run dir uploaded by scripts/job_entry.py
  gs://<bucket>/<owner>/jobs/<run id>.json   a job index (lists the run dirs of one execution)
  <run id>                         a Cloud Run execution name (shipwright-tejas-abc12) or a local
                                   run id (local-...): looked up in runs/*/repro.json, then in
                                   gs://<bucket>/*/jobs/<run id>.json

GCS reads use ``gcloud storage`` with the named config (``$SHIPWRIGHT_GCLOUD_CONFIG``, default
personal; never ADC), and only small files: anything over ``max_bytes`` is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKET = "lexsi-hackathon-runs"
SMALL = 5_000_000
Runner = Callable[[list[str]], subprocess.CompletedProcess]


def bucket() -> str:
    return os.environ.get("SHIPWRIGHT_RUNS_BUCKET") or DEFAULT_BUCKET


def gcloud(*args: str) -> list[str]:
    from shipwright import vertex

    config = os.environ.get("SHIPWRIGHT_GCLOUD_CONFIG") or vertex.DEFAULT_GCLOUD_CONFIG
    return [vertex.gcloud_binary(), f"--configuration={config}", f"--project={vertex.project()}", *args]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


class Source:
    """One run dir: local path or gs:// prefix."""

    def __init__(self, location: str, runner: Runner | None = None):
        self.location = location.rstrip("/")
        self.runner = runner or _run
        self.is_gcs = self.location.startswith("gs://")
        self._sizes: dict[str, int] | None = None

    def __repr__(self) -> str:
        return f"Source({self.location})"

    def _gcs_sizes(self) -> dict[str, int]:
        if self._sizes is None:
            out = self.runner(gcloud("storage", "ls", "-l", self.location + "/"))
            sizes = {}
            for line in (out.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit() and parts[-1].startswith("gs://"):
                    sizes[parts[-1].rsplit("/", 1)[-1]] = int(parts[0])
            self._sizes = sizes
        return self._sizes

    def read_text(self, name: str, max_bytes: int = SMALL) -> str | None:
        if not self.is_gcs:
            path = Path(self.location) / name
            if not path.is_file() or path.stat().st_size > max_bytes:
                return None
            return path.read_text(errors="replace")
        size = self._gcs_sizes().get(name)
        if size is None or size > max_bytes:
            return None
        out = self.runner(gcloud("storage", "cat", f"{self.location}/{name}"))
        return out.stdout if out.returncode == 0 else None

    def exists(self) -> bool:
        if not self.is_gcs:
            return Path(self.location).is_dir()
        return bool(self._gcs_sizes())

    def json(self, name: str) -> dict | None:
        text = self.read_text(name)
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None

    def jsonl(self, name: str, max_bytes: int = SMALL) -> list[dict] | None:
        text = self.read_text(name, max_bytes)
        if text is None:
            return None
        out = []
        for line in text.splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out


def _local_by_run_id(run_id: str, runs: Path) -> list[str]:
    hits = []
    if runs.is_dir():
        for p in sorted(runs.glob("*/repro.json")):
            try:
                rec = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            ids = {rec.get("run_id")} | {x.get("run_id") for x in rec.get("later_steps") or []}
            if run_id in ids:
                hits.append(str(p.parent))
    return hits


def resolve(ref: str, runner: Runner | None = None, runs: Path | None = None) -> tuple[list[Source], dict | None]:
    """(run dirs, job index or None) for a reference. Raises LookupError if nothing is found."""
    runner = runner or _run
    if not ref.startswith("gs://") and Path(ref).is_dir():
        return [Source(str(Path(ref)), runner)], None
    if ref.startswith("gs://"):
        if ref.endswith(".json"):
            out = runner(gcloud("storage", "cat", ref))
            if out.returncode != 0:
                raise LookupError(f"cannot read {ref}: {(out.stderr or '').strip()[:300]}")
            index = json.loads(out.stdout)
            return [Source(d, runner) for d in index.get("run_dirs") or []], index
        return [Source(ref, runner)], None
    local = _local_by_run_id(ref, runs or ROOT / "runs")
    if local:
        return [Source(d, runner) for d in local], None
    pattern = f"gs://{bucket()}/*/jobs/{ref}.json"
    out = runner(gcloud("storage", "cat", pattern))
    if out.returncode != 0 or not (out.stdout or "").strip():
        raise LookupError(f"no run dir or run id {ref!r} (looked in runs/ and {pattern})")
    index = json.loads(out.stdout)
    return [Source(d, runner) for d in index.get("run_dirs") or []], index


def first_dir(ref: str, runner: Runner | None = None) -> tuple[Source, dict | None]:
    sources, index = resolve(ref, runner)
    if not sources:
        raise LookupError(f"{ref}: the job wrote no run dir" + (f" (log: {index.get('log')})" if index else ""))
    return sources[0], index


def describe(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."

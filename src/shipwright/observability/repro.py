"""``repro.json``: everything needed to run a script again exactly (docs/ops/debugging.md).

Written into every run dir by ``shipwright.observability.log.run_script``. ``scripts/rerun.py`` reads it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = ("tau2", "litellm", "agenttune", "auditkit", "httpx", "huggingface_hub", "opentelemetry-sdk",
            "pyyaml", "shipwright")
_cache: dict[str, Any] = {}


def git_info(root: Path = ROOT) -> dict[str, Any]:
    """(sha, dirty, branch). The image has no .git: the sha comes from SHIPWRIGHT_GIT_SHA."""
    if "git" in _cache:
        return _cache["git"]
    sha = os.environ.get("SHIPWRIGHT_GIT_SHA") or None
    dirty = branch = None
    if (root / ".git").exists():
        def git(*args: str) -> str | None:
            try:
                out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
                return out.stdout.strip() if out.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                return None
        sha = sha or git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=no")
        dirty = None if status is None else bool(status)
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
    _cache["git"] = {"sha": sha, "dirty": dirty, "branch": branch}
    return _cache["git"]


def tau2_commit(root: Path = ROOT) -> str | None:
    try:
        text = (root / "scripts" / "setup_tau2.sh").read_text()
    except OSError:
        return None
    m = re.search(r'TAU2_COMMIT="([0-9a-f]+)"', text)
    return m.group(1) if m else None


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def image_info() -> dict[str, str | None]:
    return {
        "image": os.environ.get("SHIPWRIGHT_IMAGE") or None,
        "image_digest": os.environ.get("SHIPWRIGHT_IMAGE_DIGEST") or None,
        "cloud_run_job": os.environ.get("CLOUD_RUN_JOB") or None,
        "cloud_run_execution": os.environ.get("CLOUD_RUN_EXECUTION") or None,
    }


def collect(*, run_id: str, script: str, argv: list[str], owner: str | None, extra: dict | None = None) -> dict:
    """The repro record. ``extra`` holds what the script adds: models, seeds, task ids, patches."""
    git = git_info()
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "argv": list(argv),
        "owner": owner,
        "git_sha": git["sha"],
        "git_dirty": git["dirty"],
        "git_branch": git["branch"],
        **image_info(),
        "tau2_commit": tau2_commit(),
        "python": platform.python_version(),
        "packages": package_versions(),
        **(extra or {}),
    }


def models_from_args(args: Any) -> dict[str, Any]:
    """Every ``*_model`` argument, plus seed, task ids and tool patch ids, from an argparse namespace."""
    out: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key.endswith("_model") or key.endswith("_models"):
            out.setdefault("models", {})[key] = value
    for key in ("seed", "task_ids", "task_split", "trials", "num_tasks", "tool_patch_ids", "domain"):
        if hasattr(args, key):
            value = getattr(args, key)
            out[key] = str(value) if isinstance(value, Path) else value
    return out


def script_name() -> str:
    return Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "python"

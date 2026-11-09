"""Print (and with --execute, run) the exact command that reruns a run, from its repro.json.

Usage:
  .venv/bin/python scripts/rerun.py <ref> [--task <id>] [--where local-dry-run|cloud] [--offline] [--execute]

<ref> is a run dir, a gs:// run dir, a job index or a run id (as in scripts/why.py).

--where local-dry-run (default)  the same script and flags on this machine, offline: baseline and
                                 teacher runs get --dry-run, classify and tool_fix get --no-llm,
                                 gate_eval gets --local-dry-run, --launch and --push are dropped.
--where cloud                    gcloud run jobs execute shipwright-<owner> with the same flags.
                                 The job runs the latest image of its branch: the image of the
                                 original run is printed so you can compare. --offline makes it a
                                 dry run there too.
--task <id>                      only that task (scripts with --task-ids: baseline, teacher_runs).

Without --execute nothing runs. A command that calls model APIs (HF credits, Vertex billing)
also needs --allow-spend, and a human on the team must have asked for that run (AGENTS.md).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipwright.runinfo import first_dir, gcloud  # noqa: E402

REGION = "asia-south1"
# script -> (flag to add, flags to drop) for an offline rerun
OFFLINE: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "baseline.py": ("--dry-run", ()),
    "teacher_runs.py": ("--dry-run", ()),
    "classify.py": ("--no-llm", ("--judge", "--judge-all")),
    "tool_fix.py": ("--no-llm", ()),
    "gate_eval.py": ("--local-dry-run", ("--in-job",)),
    "build_sft.py": (None, ("--push",)),
    "train_job.py": (None, ("--launch",)),
    "gate_job.py": (None, ("--launch",)),
}
TASK_SCRIPTS = ("baseline.py", "teacher_runs.py")
# Flags that take values until the next flag (nargs="*" / "+").
MULTI = ("--task-ids", "--tool-patch-ids", "--target-label", "--apply")
OFFLINE_SAFE = {"baseline.py": "--dry-run", "teacher_runs.py": "--dry-run", "classify.py": "--no-llm",
                "tool_fix.py": "--no-llm", "gate_eval.py": "--local-dry-run"}


def drop_flag(argv: list[str], flag: str, takes: int | str = 0) -> list[str]:
    """Remove ``flag`` and its values. ``takes``: 0, 1, or "*" (values until the next --flag)."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == flag:
            i += 1
            if takes == "*":
                while i < len(argv) and not argv[i].startswith("--"):
                    i += 1
            else:
                i += int(takes)
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _localize(argv: list[str]) -> list[str]:
    """Paths of a Cloud Run run (/app/...) become repo-relative."""
    return [a[len("/app/"):] if a.startswith("/app/") else a for a in argv]


def offline(script: str, argv: list[str]) -> list[str]:
    add, drops = OFFLINE.get(script, (None, ()))
    for flag in drops:
        argv = drop_flag(argv, flag)
    if add and add not in argv:
        argv = argv + [add]
    return argv


def is_offline(script: str, argv: list[str]) -> bool:
    if script in ("train_job.py", "gate_job.py"):
        return "--launch" not in argv
    if script == "build_sft.py":
        return "--push" not in argv
    if script in ("why.py", "rerun.py", "round_status.py", "otel_to_traces.py"):
        return True
    flag = OFFLINE_SAFE.get(script)
    return bool(flag) and flag in argv


def build(rep: dict, *, where: str, task: str | None = None, offline_cloud: bool = False,
          owner: str | None = None, stamp: str | None = None) -> tuple[list[str], list[str]]:
    """(command, notes) for a repro record."""
    script = rep.get("script") or ""
    argv = _localize(list(rep.get("argv") or []))
    notes: list[str] = []
    if not script.endswith(".py"):
        raise ValueError(f"repro.json has no script: {script!r}")
    if task is not None:
        if script not in TASK_SCRIPTS:
            raise ValueError(f"--task needs a script with --task-ids ({', '.join(TASK_SCRIPTS)}), not {script}")
        for flag, takes in (("--task-ids", "*"), ("--num-tasks", 1), ("--all-train", 0), ("--from-run", 1)):
            argv = drop_flag(argv, flag, takes)
        argv += ["--task-ids", str(task)]
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    if "--output-dir" in argv or any(a.startswith("--output-dir=") for a in argv):
        argv = drop_flag(argv, "--output-dir", 1) + ["--output-dir", f"runs/rerun-{rep.get('run_id') or 'x'}-{stamp}"]
    if where == "local-dry-run":
        argv = offline(script, argv)
        for a in argv:
            if a.startswith("runs/") and "rerun-" not in a:
                notes.append(f"needs {a} on this machine (gcloud storage cp -r gs://<bucket>/<owner>/{a[5:]} runs/)")
        return [".venv/bin/python", f"scripts/{script}", *argv], notes
    if where != "cloud":
        raise ValueError(f"--where must be local-dry-run or cloud, not {where!r}")
    if offline_cloud:
        argv = offline(script, argv)
    owner = owner or rep.get("owner") or os.environ.get("OWNER")
    if not owner:
        raise ValueError("no owner in repro.json (a local run): pass --owner tejas|ishank|mitushi|main")
    parts = [script, *argv]
    delim = next(d for d in ("|", "@", "#", "%", "~") if not any(d in p for p in parts))
    cmd = gcloud("run", "jobs", "execute", f"shipwright-{owner}", f"--region={REGION}", "--wait",
                 f"--args=^{delim}^{delim.join(parts)}")
    if rep.get("image"):
        notes.append(f"the original run used {rep['image']} (git {(rep.get('git_sha') or '?')[:12]});"
                     f" the job runs the latest build of branch {owner}")
    return cmd, notes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref")
    p.add_argument("--task", default=None)
    p.add_argument("--where", choices=("local-dry-run", "cloud"), default="local-dry-run")
    p.add_argument("--offline", action="store_true", help="with --where cloud: make it a dry run")
    p.add_argument("--owner", default=None, help="job owner for --where cloud (default: the run's owner)")
    p.add_argument("--execute", action="store_true", help="run the command (default: print only)")
    p.add_argument("--allow-spend", action="store_true", help="with --execute: allow a command that calls model APIs")
    args = p.parse_args(argv)

    try:
        src, _ = first_dir(args.ref)
    except LookupError as e:
        print(str(e), file=sys.stderr)
        return 2
    rep = src.json("repro.json")
    if not rep:
        print(f"no repro.json in {src.location}", file=sys.stderr)
        return 2
    try:
        cmd, notes = build(rep, where=args.where, task=args.task, offline_cloud=args.offline, owner=args.owner)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    script = rep["script"]
    run_argv = cmd[2:] if args.where == "local-dry-run" else (offline(script, list(rep.get("argv") or []))
                                                             if args.offline else list(rep.get("argv") or []))
    safe = is_offline(script, run_argv)
    print(shlex.join(cmd))
    for n in notes:
        print(f"# note: {n}")
    if not safe:
        print("# WARNING: this command calls model APIs (HF credits or Vertex billing). Run it only when a human"
              " on the team asked for it.")
    if not args.execute:
        return 0
    if not safe and not args.allow_spend:
        print("Not run: add --allow-spend to run a command that spends credits.", file=sys.stderr)
        return 2
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

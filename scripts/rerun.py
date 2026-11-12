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

from shipwright.observability.runinfo import first_dir, gcloud  # noqa: E402

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
    kept_arguments = []
    position = 0
    while position < len(argv):
        argument = argv[position]
        if argument == flag:
            position += 1
            if takes == "*":
                while position < len(argv) and not argv[position].startswith("--"):
                    position += 1
            else:
                position += int(takes)
            continue
        if argument.startswith(flag + "="):
            position += 1
            continue
        kept_arguments.append(argument)
        position += 1
    return kept_arguments


def _localize(argv: list[str]) -> list[str]:
    """Paths of a Cloud Run run (/app/...) become repo-relative."""
    localized_arguments = []
    for argument in argv:
        if argument.startswith("/app/"):
            argument = argument[len("/app/"):]
        localized_arguments.append(argument)
    return localized_arguments


def offline(script: str, argv: list[str]) -> list[str]:
    """The arguments of an offline version of the run: add its dry-run flag, drop flags that spend."""
    flag_to_add, flags_to_drop = OFFLINE.get(script, (None, ()))
    for flag in flags_to_drop:
        argv = drop_flag(argv, flag)
    if flag_to_add and flag_to_add not in argv:
        argv = argv + [flag_to_add]
    return argv


def is_offline(script: str, argv: list[str]) -> bool:
    """True if running this script with these arguments makes no paid model or job call."""
    if script in ("train_job.py", "gate_job.py"):
        return "--launch" not in argv
    if script == "build_sft.py":
        return "--push" not in argv
    if script in ("why.py", "rerun.py", "round_status.py", "otel_to_traces.py"):
        return True
    offline_flag = OFFLINE_SAFE.get(script)
    return bool(offline_flag) and offline_flag in argv


def build_local_command(repro_record: dict, script: str, argv: list[str]) -> tuple[list[str], list[str]]:
    """(command, notes) for a local offline rerun."""
    argv = offline(script, argv)
    notes = []
    for argument in argv:
        if argument.startswith("runs/") and "rerun-" not in argument:
            notes.append(
                f"needs {argument} on this machine"
                f" (gcloud storage cp -r gs://<bucket>/<owner>/{argument[5:]} runs/)"
            )
    return [".venv/bin/python", f"scripts/{script}", *argv], notes


def build_cloud_command(
    repro_record: dict,
    script: str,
    argv: list[str],
    *,
    offline_cloud: bool,
    owner: str | None,
) -> tuple[list[str], list[str]]:
    """(command, notes) for a rerun as a Cloud Run job."""
    if offline_cloud:
        argv = offline(script, argv)
    owner = owner or repro_record.get("owner") or os.environ.get("OWNER")
    if not owner:
        raise ValueError("no owner in repro.json (a local run): pass --owner tejas|ishank|mitushi|main")

    job_arguments = [script, *argv]
    # gcloud splits --args on commas: pick a delimiter that appears in none of the arguments.
    delimiter = next(
        candidate for candidate in ("|", "@", "#", "%", "~")
        if not any(candidate in argument for argument in job_arguments)
    )
    command = gcloud(
        "run", "jobs", "execute", f"shipwright-{owner}",
        f"--region={REGION}", "--wait",
        f"--args=^{delimiter}^{delimiter.join(job_arguments)}",
    )
    notes = []
    if repro_record.get("image"):
        notes.append(
            f"the original run used {repro_record['image']} (git {(repro_record.get('git_sha') or '?')[:12]});"
            f" the job runs the latest build of branch {owner}"
        )
    return command, notes


def build(
    repro_record: dict,
    *,
    where: str,
    task: str | None = None,
    offline_cloud: bool = False,
    owner: str | None = None,
    stamp: str | None = None,
) -> tuple[list[str], list[str]]:
    """(command, notes) for a repro record."""
    script = repro_record.get("script") or ""
    argv = _localize(list(repro_record.get("argv") or []))
    if not script.endswith(".py"):
        raise ValueError(f"repro.json has no script: {script!r}")

    if task is not None:
        if script not in TASK_SCRIPTS:
            raise ValueError(f"--task needs a script with --task-ids ({', '.join(TASK_SCRIPTS)}), not {script}")
        for flag, value_count in (("--task-ids", "*"), ("--num-tasks", 1), ("--all-train", 0), ("--from-run", 1)):
            argv = drop_flag(argv, flag, value_count)
        argv += ["--task-ids", str(task)]

    if not stamp:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    has_output_dir = "--output-dir" in argv or any(argument.startswith("--output-dir=") for argument in argv)
    if has_output_dir:
        rerun_directory = f"runs/rerun-{repro_record.get('run_id') or 'x'}-{stamp}"
        argv = drop_flag(argv, "--output-dir", 1) + ["--output-dir", rerun_directory]

    if where == "local-dry-run":
        return build_local_command(repro_record, script, argv)
    if where == "cloud":
        return build_cloud_command(repro_record, script, argv, offline_cloud=offline_cloud, owner=owner)
    raise ValueError(f"--where must be local-dry-run or cloud, not {where!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ref")
    parser.add_argument("--task", default=None)
    parser.add_argument("--where", choices=("local-dry-run", "cloud"), default="local-dry-run")
    parser.add_argument("--offline", action="store_true", help="with --where cloud: make it a dry run")
    parser.add_argument("--owner", default=None, help="job owner for --where cloud (default: the run's owner)")
    parser.add_argument("--execute", action="store_true", help="run the command (default: print only)")
    parser.add_argument(
        "--allow-spend",
        action="store_true",
        help="with --execute: allow a command that calls model APIs",
    )
    options = parser.parse_args(argv)

    try:
        run_source, _ = first_dir(options.ref)
    except LookupError as lookup_error:
        print(str(lookup_error), file=sys.stderr)
        return 2
    repro_record = run_source.json("repro.json")
    if not repro_record:
        print(f"no repro.json in {run_source.location}", file=sys.stderr)
        return 2
    try:
        command, notes = build(
            repro_record,
            where=options.where,
            task=options.task,
            offline_cloud=options.offline,
            owner=options.owner,
        )
    except ValueError as build_error:
        print(str(build_error), file=sys.stderr)
        return 2

    script = repro_record["script"]
    if options.where == "local-dry-run":
        script_arguments = command[2:]
    elif options.offline:
        script_arguments = offline(script, list(repro_record.get("argv") or []))
    else:
        script_arguments = list(repro_record.get("argv") or [])
    makes_no_paid_calls = is_offline(script, script_arguments)

    print(shlex.join(command))
    for note in notes:
        print(f"# note: {note}")
    if not makes_no_paid_calls:
        print(
            "# WARNING: this command calls model APIs (HF credits or Vertex billing). Run it only when a human"
            " on the team asked for it."
        )
    if not options.execute:
        return 0
    if not makes_no_paid_calls and not options.allow_spend:
        print("Not run: add --allow-spend to run a command that spends credits.", file=sys.stderr)
        return 2
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

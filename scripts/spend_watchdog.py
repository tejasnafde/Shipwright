"""A second pair of eyes on a running loop: stop it, and cancel its GPU jobs, the moment its real spend passes a cap.

    .venv/bin/python scripts/spend_watchdog.py --cap-usd 3.88 --since 2026-11-13T21:05:00+00:00 --rounds 6 7

It does not trust the loop's own estimates. Every minute it prices the HF jobs created since ``--since`` by what they have
really run (running seconds times the flavor's price, a job that is still running included) and adds the spend the loop
recorded for its non-GPU steps (the teacher's corrections). Past the cap it cancels every running job and stops the loop, then
exits. It also stops waiting after ``--max-hours``. When the loop has ended and no job is running, it prints the final spend
and exits: nothing keeps running or billing after a promotion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_PRICES = {"a10g-large": 1.50, "l40sx1": 1.80, "a100-large": 2.50, "a10g-small": 1.00, "l4x1": 0.80}
UNKNOWN_FLAVOR_PRICE = 2.75  # the dearest we offer: an unknown GPU is priced high, not free
HF_JOB_STEPS = ("candidate_train", "train", "bfcl_screen", "gate")  # the manifest steps that are HF jobs (counted from the jobs)


def job_cost(job, now: datetime) -> float:
    """What one HF job has cost so far, from the seconds it has actually run."""
    price = HF_PRICES.get(str(job.flavor), UNKNOWN_FLAVOR_PRICE)
    if job.durations and job.durations.running_secs:
        seconds = job.durations.running_secs
    elif job.started_at is not None:
        end = job.finished_at or now
        seconds = max(0.0, (end - job.started_at).total_seconds())
    else:
        seconds = 0.0  # still waiting for a GPU: nothing is billed
    return seconds / 3600 * price


def manifest_other_costs(rounds: list[int], runs_directory: Path) -> float:
    """What the loop recorded for steps that are not HF jobs (the teacher's corrections, the classify estimate)."""
    total = 0.0
    for round_number in rounds:
        path = runs_directory / f"round-{round_number}.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        for step, entry in (manifest.get("stages") or {}).items():
            if step not in HF_JOB_STEPS:
                total += entry.get("actual_usd") or 0.0
        total += sum(other["usd"] for other in manifest.get("other_costs", []))
    return total


def spend_since(jobs: list, since: datetime, rounds: list[int], runs_directory: Path, now: datetime) -> float:
    jobs_cost = sum(job_cost(job, now) for job in jobs if job.created_at >= since)
    return jobs_cost + manifest_other_costs(rounds, runs_directory)


def loop_is_running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def stop_everything(hf_api, jobs: list, since: datetime, pattern: str) -> list[str]:
    """Cancel every job of this run that has not finished, and stop the loop. Returns the job ids cancelled."""
    cancelled = []
    for job in jobs:
        if job.created_at >= since and str(job.status.stage) in ("RUNNING", "SCHEDULING"):
            hf_api.cancel_job(job_id=job.id)
            cancelled.append(job.id)
    subprocess.run(["pkill", "-f", pattern])
    return cancelled


def write_log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with open(path, "a") as log_file:
        log_file.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-usd", type=float, required=True, help="stop the run when its real spend reaches this")
    parser.add_argument("--since", required=True, help="ISO time the run started: only jobs created after it count")
    parser.add_argument("--rounds", type=int, nargs="+", required=True, help="the loop's round numbers (their manifests are read)")
    parser.add_argument("--loop-pattern", default="run_loop.py")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--log", type=Path, default=ROOT / "runs" / "_meta" / "spend-watchdog.log")
    options = parser.parse_args()

    from huggingface_hub import HfApi

    hf_api = HfApi()
    since = datetime.fromisoformat(options.since)
    if since > datetime.now(timezone.utc):
        raise SystemExit(f"--since {options.since} is in the future: no job would be counted and the cap would never trigger")
    started = time.monotonic()
    options.log.parent.mkdir(parents=True, exist_ok=True)
    write_log(options.log, f"watching: cap ${options.cap_usd:.2f}, jobs created after {options.since}, rounds {options.rounds}")
    last_reported = -1.0
    while True:
        now = datetime.now(timezone.utc)
        jobs = list(hf_api.list_jobs())
        spent = spend_since(jobs, since, options.rounds, ROOT / "runs", now)
        running = [job for job in jobs if job.created_at >= since and str(job.status.stage) == "RUNNING"]
        if abs(spent - last_reported) >= 0.05:
            write_log(options.log, f"spent ${spent:.2f} of ${options.cap_usd:.2f}; running jobs: {len(running)}")
            last_reported = spent
        if spent >= options.cap_usd:
            cancelled = stop_everything(hf_api, jobs, since, options.loop_pattern)
            write_log(options.log, f"CAP REACHED at ${spent:.2f}: cancelled {cancelled or 'no job'}, stopped the loop")
            return 3
        if (time.monotonic() - started) / 3600 > options.max_hours:
            stop_everything(hf_api, jobs, since, options.loop_pattern)
            write_log(options.log, f"MAX TIME ({options.max_hours} h) reached at ${spent:.2f}: stopped")
            return 4
        waiting = [job for job in jobs if job.created_at >= since and str(job.status.stage) == "SCHEDULING"]
        if not loop_is_running(options.loop_pattern) and not running and not waiting:
            write_log(options.log, f"the loop has ended and nothing is running or queued: final spend ${spent:.2f}. Nothing keeps billing.")
            return 0
        time.sleep(options.interval)


if __name__ == "__main__":
    sys.exit(main())

"""The spend watchdog: what it counts, and that it stops at the cap."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import spend_watchdog as watchdog  # noqa: E402

NOW = datetime(2026, 11, 13, 22, 0, tzinfo=timezone.utc)


def fake_job(flavor, created_minutes_ago, running_secs=None, started_minutes_ago=None, finished=False, stage="RUNNING"):
    durations = argparse.Namespace(running_secs=running_secs) if running_secs is not None else None
    started = NOW - timedelta(minutes=started_minutes_ago) if started_minutes_ago is not None else None
    return argparse.Namespace(
        id=f"job-{flavor}-{created_minutes_ago}", flavor=flavor, created_at=NOW - timedelta(minutes=created_minutes_ago),
        started_at=started, finished_at=NOW if finished else None, durations=durations, status=argparse.Namespace(stage=stage))


def test_a_finished_job_costs_its_running_seconds_and_a_queued_one_nothing():
    assert abs(watchdog.job_cost(fake_job("l40sx1", 90, running_secs=1800), NOW) - 0.90) < 1e-9
    assert watchdog.job_cost(fake_job("a10g-large", 5, stage="SCHEDULING"), NOW) == 0.0


def test_a_running_job_is_priced_from_when_it_started():
    cost = watchdog.job_cost(fake_job("a10g-large", 40, started_minutes_ago=30), NOW)
    assert abs(cost - 0.75) < 1e-9  # 30 minutes at $1.50 an hour
    assert watchdog.job_cost(fake_job("mystery-gpu", 40, started_minutes_ago=60), NOW) == watchdog.UNKNOWN_FLAVOR_PRICE


def test_only_jobs_created_after_the_start_and_the_recorded_non_gpu_steps_count(tmp_path):
    (tmp_path / "round-6.json").write_text(json.dumps({"stages": {
        "corrections": {"actual_usd": 0.31}, "gate": {"actual_usd": 9.99},  # a gate is counted from its job, not the manifest
        "classify": {"actual_usd": 0.05}}, "other_costs": [{"usd": 0.02}]}))
    since = NOW - timedelta(minutes=60)
    jobs = [fake_job("l40sx1", 120, running_secs=3600),          # before the start: not this run
            fake_job("a10g-large", 30, running_secs=1800)]        # this run: $0.75
    spent = watchdog.spend_since(jobs, since, [6, 7], tmp_path, NOW)
    assert abs(spent - (0.75 + 0.31 + 0.05 + 0.02)) < 1e-9


def test_past_the_cap_running_and_queued_jobs_are_cancelled_and_the_loop_is_stopped(monkeypatch):
    cancelled, killed = [], []

    class Api:
        def cancel_job(self, job_id):
            cancelled.append(job_id)

    monkeypatch.setattr(watchdog.subprocess, "run", lambda command, **keyword_arguments: killed.append(command))
    since = NOW - timedelta(minutes=60)
    jobs = [fake_job("a10g-large", 30, started_minutes_ago=20, stage="RUNNING"),
            fake_job("l40sx1", 10, stage="SCHEDULING"),
            fake_job("a10g-large", 200, stage="RUNNING"),         # not this run: left alone
            fake_job("a10g-large", 20, running_secs=60, finished=True, stage="COMPLETED")]
    result = watchdog.stop_everything(Api(), jobs, since, "run_loop.py")
    assert len(result) == 2 and len(cancelled) == 2 and killed == [["pkill", "-f", "run_loop.py"]]

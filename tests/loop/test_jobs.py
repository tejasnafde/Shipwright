"""HF Jobs (shipwright.loop.jobs): queue fallback, pricing by running seconds, the gate's command and code bundle.
Fakes only: nothing is launched."""

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from shipwright import cloud
from shipwright.loop import jobs

ROOT = Path(__file__).resolve().parents[2]


class FakeHub:
    def __init__(self, stages, flavor="a10g-large", running_secs=0):
        self.stages = list(stages)
        self.flavor = flavor
        self.running_secs = running_secs
        self.cancelled = []

    def inspect_job(self, job_id):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return SimpleNamespace(status=SimpleNamespace(stage=stage), flavor=self.flavor,
                               durations=SimpleNamespace(running_secs=self.running_secs))

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)


def test_a_job_stuck_in_the_queue_is_cancelled_and_costs_nothing():
    hub = FakeHub(["SCHEDULING"])
    clock = iter(range(0, 10_000, 100))
    stage, cost = jobs.wait_for_job("j1", queue_limit_seconds=300, hf_api=hub, sleep=lambda s: None,
                                    clock=lambda: next(clock))
    assert stage == jobs.QUEUE_TIMEOUT and cost == 0.0 and hub.cancelled == ["j1"]


def test_a_job_that_runs_is_priced_by_its_running_seconds():
    hub = FakeHub(["SCHEDULING", "RUNNING", "COMPLETED"], flavor="l40sx1", running_secs=1800)
    stage, cost = jobs.wait_for_job("j1", hf_api=hub, sleep=lambda s: None)
    assert stage == "COMPLETED" and cost == pytest.approx(0.9)


def test_the_next_gpu_type_is_tried_when_the_first_waits_too_long_and_the_last_waits_as_long_as_it_takes():
    launched, limits = [], []

    def wait(job_id, queue_limit_seconds):
        limits.append(queue_limit_seconds)
        return (jobs.QUEUE_TIMEOUT, 0.0) if job_id == "job-a10g-large" else ("ERROR", 0.3)

    result = jobs.run_on_first_free_gpu(lambda flavor: launched.append(flavor) or f"job-{flavor}",
                                        ["a10g-large", "l40sx1"], wait=wait)
    assert launched == ["a10g-large", "l40sx1"] and limits == [jobs.QUEUE_LIMIT_MINUTES * 60, None]
    assert result.stage == "ERROR" and not result.completed and result.flavor == "l40sx1"  # a failure is not retried here
    assert [attempt["stage"] for attempt in result.tried] == [jobs.QUEUE_TIMEOUT, "ERROR"]


def test_the_gate_command_names_the_bundled_system_configs(tmp_path):
    candidate = tmp_path / "r12-weight.yaml"
    job = jobs.GateJob(round_number=12, current=ROOT / "configs" / "systems" / "base.yaml", candidate=candidate,
                       user_model="u", mode="train_eval", task_ids=["1", "2"])
    arguments = jobs.gate_script_arguments(job, "gate/round-12/code/x.tar.gz")
    assert arguments[arguments.index("--candidate") + 1] == "configs/systems/generated/r12-weight.json"
    assert arguments[arguments.index("--current") + 1] == "configs/systems/generated/base.json"
    assert arguments[arguments.index("--config") + 1] == "configs/systems/generated/gate.json"
    assert arguments[-5:] == ["--train-eval", "--trials", "1", "--task-ids", "1", "2"][-5:]
    assert job.results_prefix("r12-weight") == "gate/round-12/r12-weight-train"


def test_the_code_bundle_carries_a_system_config_that_is_not_committed(tmp_path):
    candidate = tmp_path / "r99-weight.yaml"
    candidate.write_text("name: r99-weight\nmodel: CohereLabs/tiny-aya-global\n")
    archive = tmp_path / "code.tar.gz"
    jobs.pack_bundle(archive, [ROOT / "configs" / "systems" / "base.yaml", candidate])
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        bundled = json.loads(bundle.extractfile("configs/systems/generated/r99-weight.json").read())
    assert bundled["name"] == "r99-weight" and "scripts/gate_eval.py" in names
    assert "configs/gate.yaml" in names


def test_every_gpu_flavor_has_one_price():
    assert cloud.worst_case_gpu_usd("a10g-small", "90m") == 1.5
    assert cloud.worst_case_gpu_usd("h200", "1h", 5.0) == 5.0
    with pytest.raises(SystemExit):
        cloud.gpu_price("h200")


@pytest.mark.parametrize("mode", ["gate", "bfcl_screen", "train_eval"])
def test_no_job_argument_trips_huggingface_hubs_local_file_rule(mode, tmp_path, monkeypatch):
    """run_uv_job refuses an argument ending in .py/.sh/.yaml/.yml/.toml that is not a local file (round 12's first launch
    failed so), and uploads and renames one that is. The arguments must name no such file, wherever the launch runs."""
    monkeypatch.chdir(tmp_path)  # not the repo root: nothing relative resolves by luck
    job = jobs.GateJob(round_number=12, current=ROOT / "configs" / "systems" / "base.yaml",
                       candidate=ROOT / "configs" / "systems" / "r11-r7-fixes.yaml", user_model="u", mode=mode,
                       task_ids=["1"], runs_only=True)
    arguments = jobs.gate_script_arguments(job, "gate/round-12/code/x.tar.gz")
    flagged = [argument for argument in arguments if Path(argument).suffix in jobs.HF_LOCAL_FILE_SUFFIXES]
    assert flagged == []
    train = jobs.TrainJob(dataset_revision="round-12", revision="round-12", training_arguments=["--epochs", "2"])
    assert [a for a in train.script_arguments() if Path(a).suffix in jobs.HF_LOCAL_FILE_SUFFIXES] == []

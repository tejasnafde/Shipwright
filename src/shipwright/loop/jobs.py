"""HF Jobs: launch the training job and the gate job, wait for them, price them, fall back to another GPU.

Both payloads are uv scripts the HF Job runs: ``scripts/train_lora.py`` (training) and ``scripts/gate_eval.py --in-job``
(the gate, a BFCL screen, or the candidate on train tasks). The GitHub repo is private, so the gate job cannot clone it:
``launch_gate`` uploads a tar of the code to the runs dataset first (``git archive HEAD``, or the image's tree inside
Cloud Run). The two system configs the gate compares are added to that tar, so a candidate config written by the loop
never has to be committed (before 2026-11-14 the loop committed and pushed it in the middle of a run).

``run_on_first_free_gpu`` tries the flavors in order: a job that waits in the GPU queue longer than the queue limit is
cancelled (a queued job is not billed) and the next flavor is tried; the last one waits as long as it takes. A failed job
is not retried here: the stage reports it, and the controller decides about a retry.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from shipwright import cloud

ROOT = cloud.ROOT
TRAIN_SCRIPT = ROOT / "scripts" / "train_lora.py"
GATE_SCRIPT = ROOT / "scripts" / "gate_eval.py"
BUNDLE_PATHS = ["src", "scripts", "configs", "patches", "pyproject.toml"]
GENERATED_SYSTEMS_DIR = "configs/systems/generated"  # where a bundle carries the system configs it was given
QUEUE_LIMIT_MINUTES = 12
JOB_POLL_SECONDS = 45
FINAL_STAGES = ("COMPLETED", "ERROR", "CANCELED", "DELETED")
QUEUE_TIMEOUT = "QUEUE_TIMEOUT"


@dataclass
class TrainJob:
    dataset_revision: str
    revision: str                       # the adapter's branch in output_repo
    flavor: str = "l40sx1"
    timeout: str = "45m"
    dataset: str = cloud.SFT_REPO
    output_repo: str = cloud.LORA_REPO
    training_arguments: list[str] = field(default_factory=list)  # more train_lora.py flags: --epochs 2 --lr 1e-4 ...
    namespace: str | None = None

    def script_arguments(self) -> list[str]:
        return ["--dataset", self.dataset, "--dataset-revision", self.dataset_revision,
                "--output-repo", self.output_repo, "--revision", self.revision, *self.training_arguments]


@dataclass
class GateJob:
    round_number: int
    current: Path                       # system configs (configs/systems/*.yaml, or one the loop wrote anywhere)
    candidate: Path
    user_model: str
    flavor: str = "a10g-large"
    timeout: str = "60m"                # never shorter: a gate needs about 40 minutes of running time (AGENTS.md, gate 8)
    mode: str = "gate"                  # gate | bfcl_screen | train_eval
    task_ids: list[str] = field(default_factory=list)  # train_eval only
    trials: int = 1                     # train_eval only
    runs_repo: str = cloud.RUNS_REPO
    config: Path = ROOT / "configs" / "gate.yaml"
    target_labels: list[str] | None = None
    serve_mode: str | None = None
    namespace: str | None = None

    def results_prefix(self, candidate_name: str) -> str:
        """Where gate_eval.py pushes the results in the runs dataset."""
        suffix = {"gate": "", "bfcl_screen": "-bfcl", "train_eval": "-train"}[self.mode]
        return f"gate/round-{self.round_number}/{candidate_name}{suffix}"


# ---------------------------------------------------------------- the code bundle (gate only)

def code_sha() -> str:
    """Short sha of the code: git HEAD, or the image's SHIPWRIGHT_GIT_SHA."""
    if not cloud.in_the_image():
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    image_sha = os.environ.get("SHIPWRIGHT_GIT_SHA", "").strip()
    return image_sha[:7] if image_sha else "nogit"


def bundled_system_path(system_path: Path) -> str:
    """The path a system config has inside the bundle (and so on the job's command line)."""
    return f"{GENERATED_SYSTEMS_DIR}/{system_path.name}"


def pack_bundle(archive_path: Path, system_paths: list[Path]) -> None:
    """git archive HEAD of BUNDLE_PATHS (inside the image: the same paths from its tree), plus the system configs."""
    def skip_python_caches(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in tar_info.name or tar_info.name.endswith(".pyc"):
            return None
        return tar_info

    with tempfile.TemporaryDirectory() as scratch:
        plain_tar = Path(scratch) / "code.tar"
        if cloud.in_the_image():
            with tarfile.open(plain_tar, "w") as bundle:
                for bundle_path in BUNDLE_PATHS:
                    if (ROOT / bundle_path).exists():
                        bundle.add(ROOT / bundle_path, arcname=bundle_path, filter=skip_python_caches)
        else:
            subprocess.check_call(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(plain_tar), "HEAD",
                                   "--", *BUNDLE_PATHS])
        with tarfile.open(plain_tar, "a") as bundle:
            for system_path in system_paths:
                bundle.add(system_path, arcname=bundled_system_path(system_path))
        with tarfile.open(plain_tar) as source, tarfile.open(archive_path, "w:gz") as compressed:
            for member in source.getmembers():
                compressed.addfile(member, source.extractfile(member) if member.isfile() else None)


def gate_script_arguments(job: GateJob, code_bundle_path: str) -> list[str]:
    """The gate_eval.py command line the HF Job runs."""
    arguments = ["--in-job", "--round", str(job.round_number),
                 "--config", str(job.config.resolve().relative_to(ROOT)),
                 "--current", bundled_system_path(job.current), "--candidate", bundled_system_path(job.candidate),
                 "--user-model", job.user_model, "--runs-repo", job.runs_repo, "--code-bundle", code_bundle_path]
    if job.target_labels is not None:
        arguments += ["--target-label", *job.target_labels]
    if job.serve_mode:
        arguments += ["--serve-mode", job.serve_mode]
    if job.mode == "train_eval":
        arguments += ["--train-eval", "--trials", str(job.trials), "--task-ids", *job.task_ids]
    if job.mode == "bfcl_screen":
        arguments.append("--bfcl-screen")
    return arguments


def code_bundle_path(job: GateJob) -> str:
    return f"gate/round-{job.round_number}/code/shipwright-{code_sha()}-{job.candidate.stem}-{job.mode}.tar.gz"


# ---------------------------------------------------------------- launching

def launch_train(job: TrainJob) -> str:
    """Submit train_lora.py. Returns the HF job id."""
    from huggingface_hub import run_uv_job

    from shipwright.observability import instrument

    token = require_hf_token()
    with instrument.external("hf_job_launch", f"{job.flavor} {TRAIN_SCRIPT.name}", timeout=job.timeout) as call_record:
        launched = run_uv_job(str(TRAIN_SCRIPT), script_args=job.script_arguments(), flavor=job.flavor,
                              timeout=job.timeout, secrets={"HF_TOKEN": token},
                              labels={"app": "lexsi-hackathon", "round": job.revision, "step": "train"},
                              namespace=job.namespace, token=token)
        call_record["job_id"] = launched.id
    return launched.id


def launch_gate(job: GateJob, needs_vertex_token: bool) -> str:
    """Upload the code bundle, then submit gate_eval.py --in-job. Returns the HF job id.

    ``needs_vertex_token``: the user simulator or the classifier is on Vertex. An HF Job cannot run gcloud, so a token is
    minted now and passed as a secret; it lasts about 1 hour with no refresh (AGENTS.md, Known traps)."""
    from huggingface_hub import HfApi, run_uv_job

    from shipwright.llm import vertex
    from shipwright.observability import instrument

    token = require_hf_token()
    job_secrets = {"HF_TOKEN": token}
    if needs_vertex_token:
        if vertex.on_gcp():
            job_secrets["SHIPWRIGHT_VERTEX_TOKEN"] = vertex.metadata_token()[0]
        else:
            job_secrets["SHIPWRIGHT_VERTEX_TOKEN"] = vertex.gcloud_token()

    hf_api = HfApi(token=token)
    hf_api.create_repo(job.runs_repo, repo_type="dataset", exist_ok=True, private=False)
    bundle_path = code_bundle_path(job)
    with tempfile.TemporaryDirectory() as scratch:
        archive_path = Path(scratch) / "code.tar.gz"
        pack_bundle(archive_path, [job.current, job.candidate])
        with instrument.external("hf_hub_upload", f"datasets/{job.runs_repo}/{bundle_path}"):
            hf_api.upload_file(path_or_fileobj=str(archive_path), path_in_repo=bundle_path, repo_id=job.runs_repo,
                               repo_type="dataset", commit_message=f"{job.mode} round {job.round_number}: code")

    with instrument.external("hf_job_launch", f"{job.flavor} {GATE_SCRIPT.name}", timeout=job.timeout) as call_record:
        launched = run_uv_job(str(GATE_SCRIPT), script_args=gate_script_arguments(job, bundle_path), flavor=job.flavor,
                              timeout=job.timeout, secrets=job_secrets,
                              labels={"app": "lexsi-hackathon", "round": f"round-{job.round_number}", "step": job.mode},
                              namespace=job.namespace, token=token)
        call_record["job_id"] = launched.id
    return launched.id


def require_hf_token() -> str:
    token = cloud.hf_token()
    if not token:
        raise SystemExit("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.")
    return token


# ---------------------------------------------------------------- waiting and pricing

@dataclass
class JobResult:
    stage: str           # COMPLETED, ERROR, CANCELED, DELETED or QUEUE_TIMEOUT
    cost_usd: float      # running seconds x the flavor's price (a queued job costs nothing)
    job_id: str
    flavor: str
    tried: list[dict] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.stage == "COMPLETED"


def wait_for_job(job_id: str, queue_limit_seconds: float | None = None, hf_api=None,
                 sleep=time.sleep, clock=time.monotonic) -> tuple[str, float]:
    """Block until an HF job ends. Returns (final stage, cost in USD from its running seconds and flavor price).

    A job still waiting for a GPU after ``queue_limit_seconds`` is cancelled and QUEUE_TIMEOUT comes back."""
    from shipwright.observability import log

    logger = log.get_logger("jobs")
    if hf_api is None:
        from huggingface_hub import HfApi

        hf_api = HfApi(token=cloud.hf_token())
    queued_since = clock()
    while True:
        job = hf_api.inspect_job(job_id=job_id)
        stage = str(job.status.stage)
        if stage in FINAL_STAGES:
            running_seconds = job.durations.running_secs if job.durations else 0
            price = cloud.GPU_PRICES.get(str(job.flavor), max(cloud.GPU_PRICES.values()))
            return stage, round(running_seconds / 3600 * price, 4)
        if stage != "SCHEDULING":
            queued_since = clock()  # it runs: the queue limit no longer applies
        elif queue_limit_seconds is not None and clock() - queued_since > queue_limit_seconds:
            hf_api.cancel_job(job_id=job_id)
            logger.warning("cancelled a job that waited too long for a GPU", job=job_id)
            return QUEUE_TIMEOUT, 0.0
        logger.info("waiting for the HF job", job=job_id, stage=stage)
        sleep(JOB_POLL_SECONDS)


def run_on_first_free_gpu(launch: Callable[[str], str], flavors: list[str],
                          queue_limit_minutes: float = QUEUE_LIMIT_MINUTES,
                          wait: Callable[[str, float | None], tuple[str, float]] | None = None) -> JobResult:
    """``launch(flavor)`` submits the job and returns its id. Each flavor but the last gets the queue limit."""
    if wait is None:
        wait = wait_for_job
    if not flavors:
        raise SystemExit("no GPU flavor given")
    tried = []
    for position, flavor in enumerate(flavors):
        is_last = position == len(flavors) - 1
        job_id = launch(flavor)
        if is_last:
            stage, cost = wait(job_id, None)
        else:
            stage, cost = wait(job_id, queue_limit_minutes * 60)
        tried.append({"flavor": flavor, "job": job_id, "stage": stage})
        if stage != QUEUE_TIMEOUT:
            return JobResult(stage, cost, job_id, flavor, tried)
    raise AssertionError("unreachable: the last flavor waits until the job ends")

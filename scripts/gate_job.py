"""Launch the deploy gate (scripts/gate_eval.py --in-job) as an HF Job.

Without --launch this prints the plan and the cost estimate only: nothing is uploaded,
nothing runs, nothing is billed. With --launch it

1. packs the committed code (git archive HEAD: src, scripts, configs, patches, pyproject.toml;
   inside the Cloud Run job, which has no .git, the same paths from the image)
   and uploads it to <runs-repo>/gate/round-N/code/ (the repo is private, so the job
   cannot clone it),
2. submits gate_eval.py with an explicit timeout (HF Jobs stop after 30 minutes by default).

Costs: the GPU (flavor price x run time, at most x timeout) and the user simulator calls on
the HF router (both from the team's HF credits).

  .venv/bin/python scripts/gate_job.py --round 1 --current configs/systems/base.yaml \\
      --candidate configs/systems/r1-tool.yaml --user-model <USER_MODEL>            # plan only
  ... --launch                                                                       # submit
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_job import estimate_cost  # noqa: E402

GATE_SCRIPT = ROOT / "scripts" / "gate_eval.py"
RUNS_REPO = "mitushigoyal5/shipwright-runs"
BUNDLE_PATHS = ["src", "scripts", "configs", "patches", "pyproject.toml"]
# User simulator cost per tau2 run (USD). The 20-task baseline spent about $0.0026 per run with
# gpt-oss-120b on cerebras; doubled for longer runs. Vertex gpt-oss-120b list prices are about
# half of cerebras (0.09/0.36 vs 0.25/0.69 per 1M tokens). Estimates only.
USER_COST_PER_RUN = {"hf_router": 0.005, "vertex": 0.0025}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # What the gate compares.
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--current", type=Path, required=True, help="system config, e.g. configs/systems/base.yaml")
    parser.add_argument("--candidate", type=Path, required=True, help="system config of the candidate")
    parser.add_argument(
        "--user-model",
        required=True,
        help="user simulator: HF router litellm id or vertex/<publisher>/<model> (no default;"
        " team choice: vertex/openai/gpt-oss-120b-maas)",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gate.yaml")
    parser.add_argument("--target-label", nargs="*", default=None, help="override the candidate's targets")
    parser.add_argument("--serve-mode", choices=("lora", "merged"), default=None)

    # The HF Job.
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--timeout", default="2h", help="job time limit, e.g. 2h or 90m (never omit it)")
    parser.add_argument("--price-per-hour", type=float, default=None, help="USD/h for a flavor not in PRICES")
    parser.add_argument("--runs-repo", default=RUNS_REPO)
    parser.add_argument("--namespace", default=None, help="HF namespace that runs (and pays for) the job")
    parser.add_argument("--launch", action="store_true", help="upload the code and submit the job (spends credits)")
    return parser.parse_args(argv)


def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def run_git(*git_arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *git_arguments], text=True).strip()


def has_git() -> bool:
    """False in the harness image (Cloud Run Jobs): it has no .git, only the built tree."""
    return (ROOT / ".git").exists()


def code_sha() -> str:
    """Short sha of the code: git HEAD, or the image's SHIPWRIGHT_GIT_SHA."""
    if has_git():
        return run_git("rev-parse", "--short", "HEAD")
    image_sha = os.environ.get("SHIPWRIGHT_GIT_SHA", "").strip()
    if image_sha:
        return image_sha[:7]
    return "nogit"


def pack_bundle(archive_path: Path) -> None:
    """git archive HEAD of BUNDLE_PATHS; without .git, a tar of the same paths as built into the image."""
    if has_git():
        subprocess.check_call(
            ["git", "-C", str(ROOT), "archive", "--format=tar.gz", "-o", str(archive_path), "HEAD",
             "--", *BUNDLE_PATHS]
        )
        return

    def skip_python_caches(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in tar_info.name or tar_info.name.endswith(".pyc"):
            return None
        return tar_info

    with tarfile.open(archive_path, "w:gz") as bundle:
        for bundle_path in BUNDLE_PATHS:
            if (ROOT / bundle_path).exists():
                bundle.add(ROOT / bundle_path, arcname=bundle_path, filter=skip_python_caches)


def gate_script_arguments(job_options: argparse.Namespace, code_bundle_path: str) -> list[str]:
    """The gate_eval.py command line the HF Job runs."""
    script_arguments = [
        "--in-job",
        "--round", str(job_options.round),
        "--config", relative_to_root(job_options.config),
        "--current", relative_to_root(job_options.current),
        "--candidate", relative_to_root(job_options.candidate),
        "--user-model", job_options.user_model,
        "--runs-repo", job_options.runs_repo,
        "--code-bundle", code_bundle_path,
    ]
    if job_options.target_label is not None:
        script_arguments += ["--target-label", *job_options.target_label]
    if job_options.serve_mode:
        script_arguments += ["--serve-mode", job_options.serve_mode]
    return script_arguments


def describe_system(system: dict, served_name: str) -> str:
    """One line of detail for a system: its adapter and tool patches, if any."""
    details = []
    if system["adapter"]:
        details.append(f"adapter {system['adapter']['repo']}@{system['adapter']['revision']}")
    if system["tool_patches"]:
        patch_ids = system["tool_patches"]["ids"]
        patch_count = patch_ids if isinstance(patch_ids, str) else len(patch_ids)
        details.append(f"patches {patch_count}")
    return f"{system['name']} -> served as {served_name} {', '.join(details)}"


def print_plan(
    job_options: argparse.Namespace,
    gate_config: dict,
    systems: dict[str, dict],
    code_bundle_path: str,
    commit_sha: str,
    user_model_on_vertex: bool,
) -> None:
    """The plan and cost estimate. Prints only; nothing is uploaded or launched."""
    from shipwright.heal.systems import served_name

    held_out = gate_config["heldout"]
    total_runs = 2 * held_out["num_tasks"] * gate_config["k"]
    if user_model_on_vertex:
        user_cost_per_run = USER_COST_PER_RUN["vertex"]
    else:
        user_cost_per_run = USER_COST_PER_RUN["hf_router"]
    user_cost = total_runs * user_cost_per_run
    price_per_hour, worst_case_gpu_cost = estimate_cost(
        job_options.flavor, job_options.timeout, job_options.price_per_hour
    )

    if job_options.target_label is not None:
        targets = job_options.target_label
    else:
        targets = systems["candidate"]["targets"]
    if user_model_on_vertex:
        user_provider = "Vertex AI (GCP billing)"
    else:
        user_provider = "the HF router"

    print(f"gate round {job_options.round}  (config status: {gate_config['status']})")
    for role, system in systems.items():
        print(f"  {role:9}  {describe_system(system, served_name(system))}")
    print(f"targets:   {targets or 'NONE (the gate will reject)'}")
    print(
        f"held-out:  {held_out['domain']} {held_out['task_split']}, {held_out['num_tasks']} tasks"
        f" x k={gate_config['k']} x 2 systems = {total_runs} tau2 runs"
    )
    print(f"user sim:  {job_options.user_model} via {user_provider}, about ${user_cost:.2f} estimate")
    if user_model_on_vertex:
        print(
            "           warning: the job gets a gcloud token minted now, valid about 1 h, with no refresh."
            " A gate that runs longer loses its user simulator; keep the timeout near 1 h or use an HF user model."
        )
    bfcl_config = gate_config.get("bfcl") or {}
    if bfcl_config.get("enabled"):
        print(
            f"BFCL:      {', '.join(bfcl_config['categories'])},"
            f" up to {bfcl_config.get('limit_per_category')} each, on the GPU"
        )
    script_arguments = gate_script_arguments(job_options, code_bundle_path)
    print(f"script:    {relative_to_root(GATE_SCRIPT)} {' '.join(script_arguments)}")
    print(f"flavor:    {job_options.flavor} at ${price_per_hour:.2f}/h")
    print(f"timeout:   {job_options.timeout}")
    print(
        f"max cost:  ${worst_case_gpu_cost:.2f} GPU (price x timeout; billed only for the time the job runs)"
        f" + about ${user_cost:.2f} user simulator"
    )
    print(
        f"results:   https://huggingface.co/datasets/{job_options.runs_repo}/tree/main/"
        f"gate/round-{job_options.round}/{systems['candidate']['name']}"
    )
    print(f"code:      git HEAD {commit_sha} -> {code_bundle_path}")
    print(f"namespace: {job_options.namespace or 'token owner'}")
    uncommitted_changes = ""
    if has_git():
        uncommitted_changes = run_git("status", "--porcelain", "--", *BUNDLE_PATHS)
    if uncommitted_changes:
        print("\nwarning: uncommitted changes in the bundle paths are NOT included (git archive HEAD)")


def upload_code_bundle(hf_api, job_options: argparse.Namespace, code_bundle_path: str, commit_sha: str) -> None:
    """git archive HEAD, uploaded to the runs dataset (the repo is private, so the job cannot clone it)."""
    from shipwright.observability import instrument

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "code.tar.gz"
        pack_bundle(archive_path)
        with instrument.external("hf_hub_upload", f"datasets/{job_options.runs_repo}/{code_bundle_path}"):
            hf_api.upload_file(
                path_or_fileobj=str(archive_path),
                path_in_repo=code_bundle_path,
                repo_id=job_options.runs_repo,
                repo_type="dataset",
                commit_message=f"gate round {job_options.round}: code {commit_sha}",
            )


def launch_job(
    job_options: argparse.Namespace,
    code_bundle_path: str,
    commit_sha: str,
    user_model_on_vertex: bool,
) -> int:
    import baseline
    from huggingface_hub import HfApi, run_uv_job
    from shipwright.llm import vertex
    from shipwright.observability import instrument

    token = baseline.load_token()
    if not token:
        print("No HF token: expected $SWITCHBOARD_SECRETS_DIR/HF_TOKEN or $HF_TOKEN.", file=sys.stderr)
        return 2

    job_secrets = {"HF_TOKEN": token}
    if user_model_on_vertex:
        try:
            if vertex.on_gcp():
                job_secrets["SHIPWRIGHT_VERTEX_TOKEN"] = vertex.metadata_token()[0]
            else:
                job_secrets["SHIPWRIGHT_VERTEX_TOKEN"] = vertex.gcloud_token()
        except RuntimeError as token_error:
            print(f"Vertex token: {token_error}", file=sys.stderr)
            return 2

    hf_api = HfApi(token=token)
    hf_api.create_repo(job_options.runs_repo, repo_type="dataset", exist_ok=True, private=False)
    upload_code_bundle(hf_api, job_options, code_bundle_path, commit_sha)

    external_call_name = f"{job_options.flavor} {GATE_SCRIPT.name}"
    with instrument.external("hf_job_launch", external_call_name, timeout=job_options.timeout) as call_record:
        job = run_uv_job(
            str(GATE_SCRIPT),
            script_args=gate_script_arguments(job_options, code_bundle_path),
            flavor=job_options.flavor,
            timeout=job_options.timeout,
            secrets=job_secrets,
            labels={"app": "lexsi-hackathon", "round": f"round-{job_options.round}", "step": "gate"},
            namespace=job_options.namespace,
            token=token,
        )
        call_record["job_id"] = job.id
    print(f"\nLaunched job {job.id}: {getattr(job, 'url', '')}")
    print(f"Logs: hf jobs logs {job.id}    Stop: hf jobs cancel {job.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    job_options = parse_args(argv)

    from shipwright.heal.gate import load_config
    from shipwright.heal.systems import load_system
    from shipwright.llm import vertex

    gate_config = load_config(job_options.config)
    systems = {
        "current": load_system(job_options.current, ROOT),
        "candidate": load_system(job_options.candidate, ROOT),
    }
    user_model_on_vertex = vertex.is_vertex_model(job_options.user_model)
    commit_sha = code_sha()
    code_bundle_path = f"gate/round-{job_options.round}/code/shipwright-{commit_sha}.tar.gz"

    print_plan(job_options, gate_config, systems, code_bundle_path, commit_sha, user_model_on_vertex)
    if not job_options.launch:
        print("\nPlan only. Nothing was uploaded or launched. Add --launch to submit the job.")
        return 0
    return launch_job(job_options, code_bundle_path, commit_sha, user_model_on_vertex)


if __name__ == "__main__":
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="gate", meta=True))

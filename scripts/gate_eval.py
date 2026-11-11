# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=1.0",
# ]
# ///
"""Deploy gate: evaluate current vs candidate on the held-out split, decide, publish.

Three modes:

  --in-job           Inside an HF Job (launched by scripts/gate_job.py). Downloads the code
                     bundle, installs the harness (scripts/setup_tau2.sh), starts vLLM with
                     the base model and the candidate LoRA, runs tau2 for both systems with
                     k trials (agent: http://localhost:8000/v1, user simulator: HF router or Vertex),
                     runs BFCL, classifies failures, computes metrics, decides and pushes
                     everything to <runs-repo>/gate/round-N/<candidate>/ plus one row to
                     the dataset's metrics.jsonl.
  --local-dry-run    Offline: fake LLM (scripts/baseline.py --dry-run), fixture BFCL data,
                     a fake BFCL model. No vLLM, no network, no token. Exercises the pipeline.
  --current-run DIR --candidate-run DIR
                     Offline: decide from two run dirs that already have failures.jsonl.

Outputs (output dir): <system>/ run dirs (traces.jsonl, failures.jsonl, bfcl.json, ...),
metrics_current.json, metrics_candidate.json, gate_decision.json, gate_inputs.json and
metrics.jsonl (one row per gate run).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS_REPO = "mitushigoyal5/shipwright-runs"
VLLM_PORT = 8000
API_BASE = f"http://localhost:{VLLM_PORT}/v1"
JOB_WORKDIR = Path("/tmp/shipwright")
DRY_BFCL = ROOT / "tests" / "fixtures" / "bfcl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # What the gate compares.
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="loop round number (results go to gate/round-N/<candidate>/)",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gate.yaml")
    parser.add_argument("--current", type=Path, default=None, help="system config of the current system")
    parser.add_argument("--candidate", type=Path, default=None, help="system config of the candidate")
    parser.add_argument(
        "--user-model",
        default=None,
        help="user simulator: HF router litellm id or vertex/<publisher>/<model> (required for"
        " --in-job; no default). In a job, vertex/ uses $SHIPWRIGHT_VERTEX_TOKEN (set by gate_job.py)",
    )
    parser.add_argument("--target-label", nargs="*", default=None, help="override the candidate's targets")

    # How the evaluation runs.
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runs-repo", default=RUNS_REPO)
    parser.add_argument("--no-push", action="store_true", help="in a job: keep results local (debug)")
    parser.add_argument("--serve-mode", choices=("lora", "merged"), default=None, help="override serve.mode")
    parser.add_argument(
        "--no-merge-fallback",
        action="store_true",
        help="do not fall back to a merged adapter if vLLM fails to start with LoRA",
    )
    parser.add_argument("--num-tasks", type=int, default=None, help="override heldout.num_tasks (debug only)")

    # The mode: exactly one of --in-job, --local-dry-run, or --current-run with --candidate-run.
    parser.add_argument("--in-job", action="store_true")
    parser.add_argument("--code-bundle", default=None, help="path in --runs-repo of the code tarball (job mode)")
    parser.add_argument("--bootstrapped", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-dry-run", action="store_true")
    parser.add_argument("--current-run", type=Path, default=None)
    parser.add_argument("--candidate-run", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    gate_options = parser.parse_args(argv)

    decide_only = bool(gate_options.current_run)
    if bool(gate_options.current_run) != bool(gate_options.candidate_run):
        parser.error("--current-run and --candidate-run go together")

    selected_mode_count = sum([gate_options.in_job, gate_options.local_dry_run, decide_only])
    if selected_mode_count != 1:
        parser.error("choose one mode: --in-job, --local-dry-run, or --current-run/--candidate-run")
    if not decide_only and not (gate_options.current and gate_options.candidate):
        parser.error("--current and --candidate system configs are required")
    if gate_options.in_job and not gate_options.user_model:
        parser.error("--user-model is required (no default)")

    # Config paths are repo-relative (the job re-runs this script from the unpacked bundle).
    for option_name in ("config", "current", "candidate"):
        config_path = getattr(gate_options, option_name)
        if config_path is not None and not config_path.is_absolute():
            setattr(gate_options, option_name, ROOT / config_path)
    return gate_options


# ---------------------------------------------------------------- shared steps

def run(
    command: list[str],
    log_path: Path | None = None,
    working_directory: Path | None = None,
    environment: dict | None = None,
) -> int:
    """Run a command and return its exit code. With log_path, stdout and stderr go to that file."""
    print("+ " + " ".join(command), flush=True)
    if log_path is None:
        return subprocess.call(command, cwd=working_directory, env=environment)
    with open(log_path, "w") as log_file:
        return subprocess.call(
            command,
            cwd=working_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def heldout_task_ids(gate_config: dict, num_tasks: int | None = None) -> list[str]:
    """The held-out ids of configs/splits.yaml (never the hidden ones); --num-tasks takes the first N."""
    from shipwright.heal import splits

    task_ids = splits.heldout_ids()
    configured_count = gate_config["heldout"]["num_tasks"]
    if len(task_ids) != configured_count:
        raise SystemExit(
            f"configs/gate.yaml heldout.num_tasks is {configured_count}, "
            f"configs/splits.yaml has {len(task_ids)} held-out ids"
        )
    if num_tasks is not None:
        return task_ids[:num_tasks]
    return task_ids


def baseline_cmd(
    gate_config: dict,
    system: dict,
    patch_ids: list[str],
    run_directory: Path,
    gate_options: argparse.Namespace,
) -> list[str]:
    """The baseline.py command line that runs one system on the held-out tasks."""
    from shipwright.heal.systems import agent_model_id, baseline_args

    held_out = gate_config["heldout"]
    command = [
        sys.executable, str(ROOT / "scripts" / "baseline.py"),
        "--domain", held_out["domain"],
        "--task-split", held_out["task_split"],
        "--task-ids", *heldout_task_ids(gate_config, gate_options.num_tasks),
        "--trials", str(gate_config["k"]),
        "--seed", str(gate_config["seed"]),
        "--agent-model", agent_model_id(system),
        "--agent-temperature", str(gate_config.get("agent_temperature", 0.0)),
        "--max-steps", str(gate_config.get("max_steps", 100)),
        "--max-concurrency", str(gate_config.get("max_concurrency", 4)),
        "--output-dir", str(run_directory),
        *baseline_args(system, patch_ids, ROOT),
    ]
    if gate_options.local_dry_run:
        command.append("--dry-run")
    else:
        command += ["--agent-api-base", API_BASE, "--user-model", gate_options.user_model]
    return command


def classify_cmd(gate_config: dict, run_directory: Path) -> list[str]:
    """The classify.py command line: LLM classifier if the config names one, else rules only."""
    classifier_model = (gate_config.get("classify") or {}).get("classifier_model")
    command = [sys.executable, str(ROOT / "scripts" / "classify.py"), str(run_directory)]
    if classifier_model:
        return command + ["--classifier-model", classifier_model]
    return command + ["--no-llm"]


def run_bfcl(
    gate_config: dict,
    system: dict,
    run_directory: Path,
    gate_options: argparse.Namespace,
) -> dict | None:
    """BFCL tool-calling score of one system (None when BFCL is off). Writes bfcl.json."""
    from shipwright.heal import bfcl_eval
    from shipwright.heal.systems import served_name

    bfcl_config = gate_config.get("bfcl") or {}
    if not bfcl_config.get("enabled"):
        return None

    if gate_options.local_dry_run:
        samples_root = DRY_BFCL
        revision = "fixture"

        def chat(messages: list[dict]) -> str:
            return "Sorry, I cannot help with that."  # fake model: never calls a tool
    else:
        revision = bfcl_config["revision"]
        samples_root = bfcl_eval.download(revision, bfcl_config["categories"], os.environ.get("HF_TOKEN"))
        chat = bfcl_eval.make_chat(API_BASE, served_name(system))

    samples = bfcl_eval.load_samples(
        samples_root, bfcl_config["categories"], revision, bfcl_config.get("limit_per_category")
    )
    result = bfcl_eval.evaluate(samples, chat)
    result["revision"] = revision
    (run_directory / "bfcl.json").write_text(json.dumps(result, indent=1))
    print(f"BFCL {system['name']}: {result['accuracy_points']} points on {result['num_samples']} samples")
    return {key: value for key, value in result.items() if key != "rows"}


def metrics_for(run_directory: Path, gate_config: dict, system_name: str, bfcl_result: dict | None) -> dict | None:
    from shipwright.heal.eval_metrics import compute_metrics, load_run

    try:
        metrics = compute_metrics(load_run(run_directory), gate_config["k"], system=system_name)
    except (FileNotFoundError, json.JSONDecodeError) as metrics_error:
        print(f"metrics for {system_name} failed: {metrics_error}", file=sys.stderr)
        return None

    bfcl_path = run_directory / "bfcl.json"
    if bfcl_result is None and bfcl_path.exists():
        saved_bfcl = json.loads(bfcl_path.read_text())
        bfcl_result = {key: value for key, value in saved_bfcl.items() if key != "rows"}
    metrics["bfcl"] = bfcl_result
    return metrics


def missing_metrics(system_name: str, reason: str) -> dict:
    """Metrics for a system whose run failed: every check on it fails, so the gate rejects."""
    return {
        "system": system_name,
        "error": reason,
        "num_valid_runs": 0,
        "num_tasks_complete": 0,
        "pass_1": None,
        "pass_k": None,
        "task_ids": [],
        "failed_run_labels": [],
    }


def decide_and_write(
    output_directory: Path,
    gate_config: dict,
    round_number: int,
    metrics_by_role: dict[str, dict],
    systems: dict,
    targets: list[str],
) -> dict:
    """Run the gate on the two systems' metrics, write the decision files, print the checks."""
    from shipwright.heal.gate import decide, metrics_row

    for role in ("current", "candidate"):
        metrics_path = output_directory / f"metrics_{role}.json"
        metrics_path.write_text(json.dumps(metrics_by_role[role], indent=1, default=str))

    decision = decide(metrics_by_role["current"], metrics_by_role["candidate"], gate_config, targets)
    decision["round"] = round_number
    decision["systems"] = systems
    (output_directory / "gate_decision.json").write_text(json.dumps(decision, indent=1, default=str))

    metrics_json_row = metrics_row(round_number, decision, systems)
    with open(output_directory / "metrics.jsonl", "a") as metrics_file:
        metrics_file.write(json.dumps(metrics_json_row, default=str) + "\n")

    print(f"\nGATE round {round_number}: {decision['decision'].upper()}")
    for check in decision["checks"]:
        check_label = "ok" if check["passed"] else "FAIL"
        print(f"  [{check_label}] {check['name']}: {check['detail']}")
    for warning in decision["warnings"]:
        print(f"  warning: {warning}")
    return decision


def load_inputs(gate_options: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    """Gate config, the two systems, their resolved patch ids, and the systems as recorded."""
    from shipwright.heal.gate import load_config
    from shipwright.heal.systems import load_system, resolve_patch_ids
    from shipwright.heal.tool_fix import load_ledger

    gate_config = load_config(gate_options.config)
    if gate_options.serve_mode:
        gate_config.setdefault("serve", {})["mode"] = gate_options.serve_mode

    systems = {
        "current": load_system(gate_options.current, ROOT),
        "candidate": load_system(gate_options.candidate, ROOT),
    }
    if systems["current"]["name"] == systems["candidate"]["name"]:
        raise SystemExit("current and candidate need different names")

    patch_ids_by_role = {}
    recorded_systems = {}
    for role, system in systems.items():
        if system["tool_patches"]:
            ledger_rows = load_ledger(ROOT / system["tool_patches"]["ledger"])
        else:
            ledger_rows = []
        patch_ids_by_role[role] = resolve_patch_ids(system, ledger_rows)
        recorded_systems[role] = {**system, "resolved_patch_ids": patch_ids_by_role[role]}
    return gate_config, systems, patch_ids_by_role, recorded_systems


def default_out(round_number: int, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "runs" / f"gate-round-{round_number}-{tag}-{timestamp}"


# ---------------------------------------------------------------- vLLM (job mode)

def download_adapter(adapter: dict, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    downloaded_path = snapshot_download(
        adapter["repo"],
        revision=adapter["revision"],
        local_dir=str(destination),
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(downloaded_path)


MERGE_CODE = """
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged", out)
"""


def merge_adapter(base: str, adapter_directory: Path, output_directory: Path, vllm_spec: str, log_path: Path) -> Path:
    """peft merge_and_unload in vLLM's own env (same torch and transformers)."""
    command = [
        "uv", "run", "--no-project", "--python", "3.12", "--with", vllm_spec, "--with", "peft",
        "python", "-c", MERGE_CODE, base, str(adapter_directory), str(output_directory),
    ]
    if run(command, log_path=log_path) != 0:
        raise RuntimeError(f"adapter merge failed, see {log_path}")
    return output_directory


def vllm_cmd(server: dict, serve_config: dict, lora_paths: dict[str, Path], model_path: str) -> list[str]:
    command = [
        "uv", "run", "--no-project", "--python", "3.12", "--with", serve_config["vllm"],
        "vllm", "serve", model_path,
        "--served-model-name", server["served_base"],
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", str(serve_config.get("max_model_len", 8192)),
        "--gpu-memory-utilization", str(serve_config.get("gpu_memory_utilization", 0.85)),
    ]
    if lora_paths:
        lora_modules = [f"{name}={path}" for name, path in lora_paths.items()]
        command += [
            "--enable-lora",
            "--max-lora-rank", str(serve_config.get("max_lora_rank", 16)),
            "--max-loras", str(len(lora_paths)),
            "--lora-modules", *lora_modules,
        ]
    return command


class VLLMServer:
    """A vLLM server process, started on enter and stopped on exit."""

    def __init__(self, command: list[str], expected_model_names: list[str], log_path: Path, timeout_seconds: int):
        self.command = command
        self.expected_model_names = expected_model_names
        self.log_path = log_path
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "VLLMServer":
        import httpx

        print("+ " + " ".join(self.command), flush=True)
        self._log_file = open(self.log_path, "w")
        self.process = subprocess.Popen(self.command, stdout=self._log_file, stderr=subprocess.STDOUT)

        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(f"vLLM exited with {self.process.returncode}, see {self.log_path}")
            try:
                response = httpx.get(f"{API_BASE}/models", timeout=5)
                served_names = {model["id"] for model in response.json().get("data", [])}
                if response.status_code == 200 and set(self.expected_model_names) <= served_names:
                    print(f"vLLM ready: {sorted(served_names)}", flush=True)
                    return self
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(10)

        self.close()
        raise RuntimeError(f"vLLM not ready after {self.timeout_seconds}s, see {self.log_path}")

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._log_file.close()

    def __exit__(self, *exception_info) -> None:
        self.close()


def run_server_plan(
    plan: list[dict],
    gate_config: dict,
    system_by_name: dict[str, tuple[str, dict]],
    patch_ids_by_role: dict,
    output_directory: Path,
    gate_options: argparse.Namespace,
    status_by_system: dict[str, dict],
) -> None:
    """For each vLLM server of the plan: start it, then run tau2 and BFCL for its systems."""
    from shipwright.heal.systems import served_name

    serve_config = gate_config.get("serve") or {}
    adapters_directory = Path("/tmp/adapters")

    for server_index, server in enumerate(plan):
        lora_paths = {
            lora_name: download_adapter(adapter, adapters_directory / lora_name)
            for lora_name, adapter in server["loras"].items()
        }
        model_path = server["base"]
        if server["merged"]:
            adapter_directory = download_adapter(server["merged"], adapters_directory / server["served_base"])
            merged_directory = Path(f"/tmp/merged/{server['served_base']}")
            model_path = str(merge_adapter(
                server["base"], adapter_directory, merged_directory,
                serve_config["vllm"], output_directory / f"merge-{server_index}.log",
            ))

        expected_model_names = [server["served_base"], *lora_paths]
        startup_timeout = int(serve_config.get("startup_timeout_s", 1200))
        server_command = vllm_cmd(server, serve_config, lora_paths, model_path)
        with VLLMServer(server_command, expected_model_names, output_directory / f"vllm-{server_index}.log",
                        startup_timeout):
            for system_name in server["systems"]:
                role, system = system_by_name[system_name]
                run_directory = output_directory / system_name
                run_directory.mkdir(parents=True, exist_ok=True)
                baseline_exit_code = run(
                    baseline_cmd(gate_config, system, patch_ids_by_role[role], run_directory, gate_options),
                    log_path=run_directory / "baseline.log",
                )
                bfcl_result = None
                try:
                    bfcl_result = run_bfcl(gate_config, system, run_directory, gate_options)
                except Exception as bfcl_error:  # noqa: BLE001 - recorded; the gate rejects without BFCL
                    print(f"BFCL failed for {system_name}: {bfcl_error}", file=sys.stderr)
                status_by_system[system_name] = {
                    "baseline_exit": baseline_exit_code,
                    "bfcl": bfcl_result,
                    "served_as": served_name(system),
                }


def serve_and_run(
    gate_config: dict,
    systems: dict,
    patch_ids_by_role: dict,
    output_directory: Path,
    gate_options: argparse.Namespace,
) -> dict:
    """Start vLLM per the serve plan, run tau2 and BFCL for each system. Returns status per system."""
    from shipwright.heal.systems import serve_plan

    serve_mode = (gate_config.get("serve") or {}).get("mode", "lora")
    system_by_name = {system["name"]: (role, system) for role, system in systems.items()}
    all_systems = list(systems.values())
    status_by_system: dict[str, dict] = {}

    def run_plan(mode: str) -> None:
        run_server_plan(
            serve_plan(all_systems, mode), gate_config, system_by_name, patch_ids_by_role,
            output_directory, gate_options, status_by_system,
        )

    try:
        run_plan(serve_mode)
        status_by_system["_serve_mode"] = {"mode": serve_mode}
    except RuntimeError as serve_error:
        any_system_has_adapter = any(system.get("adapter") for system in all_systems)
        cannot_fall_back = (
            serve_mode != "lora"
            or not any_system_has_adapter
            or gate_options.no_merge_fallback
            or bool(status_by_system)  # some system already ran: do not run it twice
        )
        if cannot_fall_back:
            raise
        print(f"vLLM with LoRA failed ({serve_error}); falling back to merged adapters", file=sys.stderr)
        run_plan("merged")
        status_by_system["_serve_mode"] = {"mode": "merged", "lora_error": str(serve_error)[:500]}
    return status_by_system


# ---------------------------------------------------------------- push

def push(output_directory: Path, gate_options: argparse.Namespace, row_file: Path, candidate_name: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    from shipwright.observability import instrument

    token = os.environ.get("HF_TOKEN")
    hf_api = HfApi(token=token)
    hf_api.create_repo(gate_options.runs_repo, repo_type="dataset", exist_ok=True, private=False)

    path_in_repo = f"gate/round-{gate_options.round}/{candidate_name}"
    with instrument.external("hf_hub_upload", f"datasets/{gate_options.runs_repo}/{path_in_repo}"):
        hf_api.upload_folder(
            folder_path=str(output_directory),
            repo_id=gate_options.runs_repo,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            commit_message=f"gate round {gate_options.round}: results",
            ignore_patterns=["**/tau2/**"],  # tau2's own results.json duplicates traces.jsonl
        )

    # Append this run's row to the dataset-level metrics.jsonl.
    try:
        existing_metrics_path = hf_hub_download(
            gate_options.runs_repo, "metrics.jsonl", repo_type="dataset", token=token
        )
        existing_metrics = Path(existing_metrics_path).read_text()
    except (EntryNotFoundError, RepositoryNotFoundError):
        existing_metrics = ""
    separator = ""
    if existing_metrics and not existing_metrics.endswith("\n"):
        separator = "\n"
    combined_metrics_path = output_directory / "_metrics_all.jsonl"
    combined_metrics_path.write_text(existing_metrics + separator + row_file.read_text())
    with instrument.external("hf_hub_upload", f"datasets/{gate_options.runs_repo}/metrics.jsonl"):
        hf_api.upload_file(
            path_or_fileobj=str(combined_metrics_path),
            path_in_repo="metrics.jsonl",
            repo_id=gate_options.runs_repo,
            repo_type="dataset",
            commit_message=f"metrics.jsonl: gate round {gate_options.round}",
        )
    print(f"pushed to https://huggingface.co/datasets/{gate_options.runs_repo}/tree/main/{path_in_repo}")


# ---------------------------------------------------------------- modes

def bootstrap(argv: list[str], gate_options: argparse.Namespace) -> None:
    """Job mode, first pass: fetch the code bundle, install the harness, re-run in its venv."""
    from huggingface_hub import hf_hub_download

    if not gate_options.code_bundle:
        raise SystemExit("--code-bundle is required in job mode")
    bundle_path = hf_hub_download(
        gate_options.runs_repo, gate_options.code_bundle,
        repo_type="dataset", token=os.environ.get("HF_TOKEN"),
    )
    if JOB_WORKDIR.exists():
        shutil.rmtree(JOB_WORKDIR)
    JOB_WORKDIR.mkdir(parents=True)
    with tarfile.open(bundle_path) as bundle:
        bundle.extractall(JOB_WORKDIR, filter="data")
    if run(["bash", str(JOB_WORKDIR / "scripts" / "setup_tau2.sh")]) != 0:
        raise SystemExit("harness setup failed")
    python_path = str(JOB_WORKDIR / ".venv" / "bin" / "python")
    os.execv(python_path, [python_path, str(JOB_WORKDIR / "scripts" / "gate_eval.py"), *argv, "--bootstrapped"])


def _expected_reject() -> None:
    try:
        from shipwright.observability import log
    except ImportError:
        return
    log.expected_exit(1, "the gate rejected the candidate")


def _attach(output_directory: Path) -> None:
    """log.jsonl, errors.json and repro.json in the gate output dir (when run through log.run_script)."""
    try:
        from shipwright.observability import log
    except ImportError:
        return
    log.attach_run_dir(output_directory)


def decide_from_existing_runs(gate_options: argparse.Namespace) -> int:
    """--current-run and --candidate-run: gate two finished run dirs. Exit 0 = promote, 1 = reject."""
    from shipwright.heal.gate import load_config

    gate_config = load_config(gate_options.config)
    if gate_options.output_dir:
        output_directory = gate_options.output_dir.resolve()
    else:
        output_directory = default_out(gate_options.round, "decide").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    _attach(output_directory)

    run_directories = {"current": gate_options.current_run, "candidate": gate_options.candidate_run}
    metrics_by_role = {}
    for role, run_directory in run_directories.items():
        metrics = metrics_for(run_directory, gate_config, run_directory.name, None)
        metrics_by_role[role] = metrics or missing_metrics(run_directory.name, "no metrics")

    targets = gate_options.target_label or []
    recorded_systems = {role: {"run_dir": str(run_directory)} for role, run_directory in run_directories.items()}
    decision = decide_and_write(
        output_directory, gate_config, gate_options.round, metrics_by_role, recorded_systems, targets
    )
    print(f"\nWrote {output_directory}")
    if decision["decision"] == "promote":
        return 0
    _expected_reject()
    return 1


def collect_metrics(
    output_directory: Path,
    gate_config: dict,
    systems: dict,
    status_by_system: dict,
) -> dict[str, dict]:
    """Classify each system's run and compute its metrics; a failed system gets missing_metrics."""
    metrics_by_role = {}
    for role, system in systems.items():
        system_name = system["name"]
        run_directory = output_directory / system_name
        system_status = status_by_system.get(system_name) or {}
        baseline_exit_code = system_status.get("baseline_exit")

        if baseline_exit_code != 0 or not (run_directory / "traces.jsonl").exists():
            metrics_by_role[role] = missing_metrics(system_name, f"tau2 run failed (exit {baseline_exit_code})")
            continue
        classify_exit_code = run(classify_cmd(gate_config, run_directory), log_path=run_directory / "classify.log")
        if classify_exit_code != 0:
            metrics_by_role[role] = missing_metrics(system_name, "classify failed")
            continue
        metrics = metrics_for(run_directory, gate_config, system_name, system_status.get("bfcl"))
        metrics_by_role[role] = metrics or missing_metrics(system_name, "no metrics")
    return metrics_by_role


def evaluate_systems(gate_options: argparse.Namespace) -> int:
    """--in-job and --local-dry-run: run both systems on the held-out tasks, then gate them."""
    gate_config, systems, patch_ids_by_role, recorded_systems = load_inputs(gate_options)
    if gate_options.target_label is not None:
        targets = gate_options.target_label
    else:
        targets = systems["candidate"]["targets"]

    mode_tag = "dry" if gate_options.local_dry_run else "job"
    if gate_options.output_dir:
        output_directory = gate_options.output_dir.resolve()
    elif gate_options.in_job:
        output_directory = Path("/tmp/gate-out").resolve()
    else:
        output_directory = default_out(gate_options.round, mode_tag).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    _attach(output_directory)

    gate_inputs = {
        "round": gate_options.round,
        "config": gate_config,
        "systems": recorded_systems,
        "targets": targets,
        "user_model": gate_options.user_model,
        "mode": mode_tag,
    }
    (output_directory / "gate_inputs.json").write_text(json.dumps(gate_inputs, indent=1))

    if gate_options.local_dry_run:
        status_by_system = {}
        for role, system in systems.items():
            run_directory = output_directory / system["name"]
            run_directory.mkdir(parents=True, exist_ok=True)
            baseline_exit_code = run(
                baseline_cmd(gate_config, system, patch_ids_by_role[role], run_directory, gate_options),
                log_path=run_directory / "baseline.log",
            )
            status_by_system[system["name"]] = {
                "baseline_exit": baseline_exit_code,
                "bfcl": run_bfcl(gate_config, system, run_directory, gate_options),
            }
    else:
        try:
            status_by_system = serve_and_run(gate_config, systems, patch_ids_by_role, output_directory, gate_options)
        except Exception as serve_error:  # noqa: BLE001 - still decide (reject) and publish the logs
            print(f"serving or running failed: {serve_error}", file=sys.stderr)
            status_by_system = {"_error": str(serve_error)[:1000]}

    metrics_by_role = collect_metrics(output_directory, gate_config, systems, status_by_system)
    (output_directory / "run_status.json").write_text(json.dumps(status_by_system, indent=1, default=str))
    decide_and_write(output_directory, gate_config, gate_options.round, metrics_by_role, recorded_systems, targets)

    if gate_options.in_job and not gate_options.no_push:
        push(output_directory, gate_options, output_directory / "metrics.jsonl", systems["candidate"]["name"])
    print(f"\nWrote {output_directory}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = list(sys.argv[1:])
    else:
        argv = list(argv)
    gate_options = parse_args(argv)

    if gate_options.in_job and not gate_options.bootstrapped:
        bootstrap(argv, gate_options)  # does not return

    if gate_options.current_run:
        return decide_from_existing_runs(gate_options)
    return evaluate_systems(gate_options)


if __name__ == "__main__":
    if "--in-job" in sys.argv and "--bootstrapped" not in sys.argv:
        sys.exit(main())  # first pass in an HF Job: the harness (and shipwright.log) is not installed yet
    from shipwright.observability import log

    sys.exit(log.run_script(main, step="gate"))

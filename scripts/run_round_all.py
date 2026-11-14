"""A whole round with one command, in one Cloud Run Job execution: `scripts/run_round.py all` runs this.

    .venv/bin/python scripts/run_round.py all --config configs/rounds/default.yaml             # plan: free
    .venv/bin/python scripts/run_round.py all --config configs/rounds/default.yaml --launch    # one Cloud Run execution
    .venv/bin/python scripts/run_round.py all --config configs/rounds/default.yaml --dry-run   # offline, fake models and jobs

--launch starts ONE execution of shipwright-<owner> that runs this script with --here. Inside it, every stage runs in
order with no laptop, agent or human watching: baseline -> classify -> teacher -> data -> train (HF Job, polled) ->
gate (HF Job, polled) -> promote (scripts/promote.py) or a reject with next_round.json. The decisions (stop rules, one
retry, spend caps, later rounds) live in src/shipwright/heal/round_chain.py.

Outputs, per round N: runs/rN-round/ (round.json: the manifest with the resolved config; gate_decision.json;
metrics.jsonl; next_round.json on a reject; ledger.jsonl and current.yaml on a promotion), mirrored to
runs/round-N.json for the control room. The chain's summary: runs/chain-rN/chain.json. In the cloud every stage's
run dir is uploaded to gs://lexsi-hackathon-runs/<owner>/ as soon as it finishes, so a rerun of the same command
fetches them and resumes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shipwright.heal import promotion, round_chain  # noqa: E402
from shipwright.heal import recipe as recipe_module  # noqa: E402
from shipwright.heal.round_chain import run_name  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("run_round_all")
RUNS = ROOT / "runs"
CLASSIFY_ESTIMATE_USD = 0.05  # classify.py does not meter its calls (run_round.BUDGETS)


def load_script_module(name: str):
    """run_loop.py and run_round.py are scripts, not packages: load them by path (as tests/test_run_loop.py does)."""
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def in_the_image() -> bool:
    """True inside the harness image (Cloud Run): it has no .git, only the built tree (as gate_job.has_git)."""
    return not (ROOT / ".git").exists()


# ---------------------------------------------------------------- dry-run config

def dry_run_config(config: dict) -> dict:
    """The same chain, offline: fake models (the scripts' own --dry-run), fake HF jobs, run dirs named dry-*, and
    stop rules that the fake models can meet (they never pass a task, so there is no SFT data)."""
    dry = json.loads(json.dumps(config))
    dry["name_prefix"] = "dry-"
    dry["baseline"]["num_tasks"] = 2
    dry["teacher"]["trials"] = 1
    dry["data"]["pool"] = []
    dry["stop"]["min_sft_examples"] = 0
    dry["train"]["require_trigger"] = False
    return dry


# ---------------------------------------------------------------- moving run dirs (cloud only)

def owner_bucket(config: dict) -> tuple[str, str]:
    return config["owner"], os.environ.get("SHIPWRIGHT_RUNS_BUCKET", "lexsi-hackathon-runs")


def upload_run_dir(config: dict, directory: Path) -> None:
    """In the cloud: copy a finished stage's run dir to the bucket now, so a dead execution loses nothing."""
    if not in_the_image() or not directory.is_dir():
        return
    import httpx

    job_entry = load_script_module("job_entry")
    owner, bucket = owner_bucket(config)
    with httpx.Client(timeout=300) as http_client:
        for path in directory.rglob("*"):
            if path.is_file():
                job_entry.upload(http_client, bucket, f"{owner}/{directory.name}/{path.relative_to(directory)}",
                                 path.read_bytes())


def fetch_run_dir(config: dict, qualified_name: str) -> bool:
    """In the cloud: download gs://<bucket>/<owner>/<dir>/ into runs/<dir>/. False when it is not there."""
    local = RUNS / qualified_name.rsplit("/", 1)[-1]
    if local.is_dir():
        return True
    if not in_the_image():
        return False
    import httpx

    job_entry = load_script_module("job_entry")
    _, bucket = owner_bucket(config)
    try:
        with httpx.Client(timeout=300) as http_client:
            job_entry.fetch_dir(http_client, bucket, qualified_name, RUNS)
    except FileNotFoundError:
        return False
    return True


# ---------------------------------------------------------------- manifests and outputs

def round_directory(config: dict, round_number: int) -> Path:
    return RUNS / run_name(config, round_number, "round")


def load_manifest(config: dict, round_number: int) -> dict | None:
    """The round's manifest from an earlier execution (resume), with the run dirs of its finished stages."""
    fetch_run_dir(config, f"{config['owner']}/{round_directory(config, round_number).name}")
    path = round_directory(config, round_number) / "round.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for entry in manifest["stages"].values():
        if entry.get("ok") and entry.get("run_dir"):
            fetch_run_dir(config, f"{config['owner']}/{Path(entry['run_dir']).name}")
    return manifest


def save_manifest(config: dict, round_number: int, manifest: dict) -> None:
    directory = round_directory(config, round_number)
    directory.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = json.dumps(manifest, indent=2, default=str)
    (directory / "round.json").write_text(text)
    if not config.get("name_prefix"):
        (RUNS / f"round-{round_number}.json").write_text(text)  # the control room reads runs/round-N.json
    for entry in manifest["stages"].values():
        if entry.get("ok") and entry.get("run_dir") and not entry.get("uploaded"):
            upload_run_dir(config, ROOT / entry["run_dir"])
            entry["uploaded"] = in_the_image()
    upload_run_dir(config, directory)


def resolved_config(config: dict, config_path: Path, rounds: int, dry_run: bool) -> dict:
    """The config as run, plus what is not in the file: the code, the image and the frozen files' hashes."""
    if in_the_image():
        code_sha = os.environ.get("SHIPWRIGHT_GIT_SHA", "")[:12]
    else:
        code_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    return {**config, "rounds": rounds, "dry_run": dry_run, "config_file": str(config_path),
            "config_sha": round_chain.file_sha(config_path),
            "resolved": {"code_sha": code_sha, "image": os.environ.get("SHIPWRIGHT_IMAGE_DIGEST")
                         or os.environ.get("SHIPWRIGHT_IMAGE"),
                         "frozen_sha": {name: round_chain.file_sha(ROOT / path)
                                        for name, path in config["frozen"].items()}}}


CANDIDATE_EXTRA_KEYS = {"state_reminders"}


def write_candidate_system(config: dict, round_number: int, state: dict) -> Path:
    """configs/systems/rN-weight.yaml from the template: this chain's adapter, the recipe's guards. In the image the gate
    bundle is a tar of the tree, so the file needs no commit. A dry run writes it into its round dir only."""
    from shipwright.heal import systems

    system = yaml.safe_load((ROOT / config["gate"]["candidate_template"]).read_text())
    system["name"] = f"r{round_number}-weight"
    system["adapter"] = {"repo": config["train"]["output_repo"], "revision": f"round-{state['adapter_round']}"}
    system["tool_guards"] = state["recipe"].tool_guards
    # Serving-only switches for the candidate (config gate.candidate_extra), e.g. state_reminders: true.
    for key, value in (config["gate"].get("candidate_extra") or {}).items():
        if key not in CANDIDATE_EXTRA_KEYS:
            raise SystemExit(f"gate.candidate_extra: {key} is not one of {sorted(CANDIDATE_EXTRA_KEYS)}")
        system[key] = value
    systems.validate_system(system)
    text = (f"# Round {round_number} candidate, written by scripts/run_round_all.py from "
            f"{config['gate']['candidate_template']}.\n" + yaml.safe_dump(system, sort_keys=False))
    if config.get("name_prefix"):
        path = round_directory(config, round_number) / f"{system['name']}.yaml"
    else:
        path = ROOT / "configs" / "systems" / f"{system['name']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def promote(config: dict, round_number: int, decision: dict) -> None:
    """scripts/promote.py's ledger and current.yaml; a copy goes into the round dir (in the cloud, commit them from
    there: the image is thrown away). A dry run keeps its ledger in its round dir only."""
    directory = round_directory(config, round_number)
    decision_path = RUNS / run_name(config, round_number, "gate") / "gate_decision.json"
    if config.get("name_prefix"):
        ledger_path = directory / "ledger.jsonl"
        promotion.promote(ledger_path, decision, str(decision_path))
        system = promotion.deployed_system(promotion.read_ledger(ledger_path))
        (directory / "current.yaml").write_text(yaml.safe_dump(system, sort_keys=False))
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "promote.py"), "promote", str(decision_path)], check=True)
    shutil.copy(ROOT / "promotions" / "ledger.jsonl", directory / "ledger.jsonl")
    shutil.copy(ROOT / "configs" / "systems" / "current.yaml", directory / "current.yaml")
    upload_run_dir(config, directory)


def write_next_round(config: dict, round_number: int, decision: dict, next_recipe: recipe_module.Recipe) -> None:
    """What a following round starts from after a refusal: why, the recipe it would use, and the command that gets the
    candidate's own failures on the train tasks (the held-out failures are never training data)."""
    directory = round_directory(config, round_number)
    failed = sorted(recipe_module.failed_check_names(decision))
    next_round = {
        "rejected_round": round_number, "failed_checks": failed,
        "checks": decision.get("checks"),
        "candidate": f"r{round_number}-weight",
        "next_recipe": recipe_module.recipe_to_dict(next_recipe),
        "failures_on_train_tasks": (f".venv/bin/python scripts/run_loop.py --round {round_number + 1} --from-round "
                                    f"{round_number} --budget-usd {config['spend']['total']} --launch"),
        "base_failures_run": f"runs/{run_name(config, config['round'], 'baseline')}",
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "next_round.json").write_text(json.dumps(next_round, indent=2))


# ---------------------------------------------------------------- the stages

def run_script(arguments: list[str]) -> int:
    """scripts/<script> args, in this process's tree (in the cloud: the execution's own container)."""
    logger.info("stage command", command=arguments)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / arguments[0]), *arguments[1:]]).returncode


def relative(path: Path) -> str:
    """runs/... for the manifest (the tests write under a tmp dir)."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def api_stage_entry(exit_code: int, run_directory: Path) -> dict:
    summary = read_json(run_directory / "summary.json")
    return {"exit_code": exit_code, "run_dir": relative(run_directory),
            "actual_usd": (summary.get("cost_estimate_usd") or {}).get("total") or 0.0,
            "num_runs": summary.get("num_runs"), "num_passed": summary.get("num_passed"),
            "num_infra_errors": summary.get("num_infra_errors"), "cost_cap_hit": summary.get("cost_cap_hit")}


class Stages:
    """One round's stage functions. ``dry_run`` swaps the models for the scripts' fake LLM and the HF jobs for fakes."""

    def __init__(self, config: dict, round_number: int, state: dict, dry_run: bool):
        self.config, self.round_number, self.state, self.dry_run = config, round_number, state, dry_run
        self.baseline_directory = RUNS / state["baseline_run"]
        self.teacher_directory = RUNS / state["teacher_run"]

    def functions(self) -> dict:
        return {"baseline": self.baseline, "classify": self.classify, "teacher": self.teacher, "data": self.data,
                "train": self.train, "gate": self.gate}

    def baseline(self) -> dict:
        settings, models = self.config["baseline"], self.config["models"]
        arguments = ["baseline.py", "--task-split", "train", "--num-tasks", str(settings["num_tasks"]),
                     "--trials", str(settings["trials"]), "--agent-model", models["agent"], "--user-model", models["user"],
                     "--max-concurrency", str(settings["max_concurrency"]), "--seed", str(self.config["seed"]),
                     "--max-cost-usd", str(self.config["spend"]["baseline"]),
                     "--output-dir", str(self.baseline_directory)]
        if self.dry_run:
            arguments.append("--dry-run")
        return api_stage_entry(run_script(arguments), self.baseline_directory)

    def classify(self) -> dict:
        arguments = ["classify.py", str(self.baseline_directory), "--classifier-model", self.config["models"]["classifier"]]
        if self.dry_run:
            arguments.append("--no-llm")
        exit_code = run_script(arguments)
        failures = read_json(self.baseline_directory / "failure_summary.json")
        return {"exit_code": exit_code, "run_dir": relative(self.baseline_directory),
                "actual_usd": 0.0 if self.dry_run else CLASSIFY_ESTIMATE_USD,
                "num_failed": failures.get("num_failed"), "per_route": failures.get("per_route")}

    def teacher(self) -> dict:
        settings, models = self.config["teacher"], self.config["models"]
        arguments = ["teacher_runs.py", "--teacher-model", models["teacher"], "--user-model", models["user"],
                     "--from-run", str(self.baseline_directory), f"--trials={settings['trials']}",
                     "--max-concurrency", str(settings["max_concurrency"]), "--seed", str(self.config["seed"]),
                     "--max-cost-usd", str(self.config["spend"]["teacher"]), "--output-dir", str(self.teacher_directory)]
        if self.dry_run:
            arguments.append("--dry-run")
        entry = api_stage_entry(run_script(arguments), self.teacher_directory)
        kept = self.teacher_directory / "kept.jsonl"
        entry["kept_traces"] = len([line for line in kept.read_text().splitlines() if line.strip()]) if kept.exists() else 0
        return entry

    def data(self) -> dict:
        recipe = self.state["recipe"]
        output_directory = RUNS / run_name(self.config, self.round_number, "sft")
        missing = [name for name in self.config["data"]["pool"] if not fetch_run_dir(self.config, name)]
        if missing:
            return {"exit_code": 1, "actual_usd": 0.0, "note": f"pooled teacher runs not found: {missing}"}
        arguments = ["build_sft.py", "--output-dir", str(output_directory), "--only-model", self.config["models"]["teacher"],
                     "--revision", f"round-{self.round_number}", "--repo", self.config["data"]["repo"],
                     "--boost-from", str(self.baseline_directory), "--boost", str(recipe.boost)]
        for teacher_directory in [self.teacher_directory, *[RUNS / name.rsplit("/", 1)[-1] for name in self.config["data"]["pool"]]]:
            arguments += ["--teacher", str(teacher_directory)]
        if recipe.general_replay:
            arguments += ["--general-replay", str(load_script_module("run_loop").build_general_replay(recipe.general_replay))]
        if not self.dry_run:
            arguments.append("--push")
        exit_code = run_script(arguments)
        statistics = read_json(output_directory / "sft_stats.json")
        return {"exit_code": exit_code, "actual_usd": 0.0, "run_dir": relative(output_directory),
                "examples": statistics.get("examples"), "tasks": len(statistics.get("tasks") or []),
                "general_replay": statistics.get("general_replay")}

    def retraining_trigger(self) -> dict:
        """AgentTune's RetrainingTrigger (scripts/round_status.py, offline)."""
        sft_path = RUNS / run_name(self.config, self.round_number, "sft") / "sft.jsonl"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "round_status.py"), str(self.baseline_directory),
                        "--sft", str(sft_path)], capture_output=True, text=True)
        status = read_json(self.baseline_directory / "round_status.json")
        return {key: status.get(key) for key in ("should_train", "reason", "num_weight_fix_failures", "num_with_correction")}

    def train(self) -> dict:
        trigger = self.retraining_trigger()
        if self.config["train"]["require_trigger"] and not trigger.get("should_train"):
            return {"exit_code": 1, "actual_usd": 0.0, "retrain_trigger": trigger, "no_retry": True,
                    "note": f"AgentTune's RetrainingTrigger says do not train ({trigger.get('reason')})"}
        if self.dry_run:
            return {"exit_code": 0, "actual_usd": 0.0, "job": "fake-train", "flavor": "fake", "retrain_trigger": trigger}
        settings, revision = self.config["train"], f"round-{self.round_number}"

        def make_command(flavor: str) -> list[str]:
            return [sys.executable, str(ROOT / "scripts" / "train_job.py"), "--dataset", self.config["data"]["repo"],
                    "--dataset-revision", revision, "--output-repo", settings["output_repo"], "--revision", revision,
                    "--flavor", flavor, "--timeout", settings["timeout"], "--launch", "--",
                    "--epochs", f"{self.state['recipe'].epochs:g}", "--lr", f"{self.state['recipe'].learning_rate:g}",
                    "--seed", str(self.config["seed"]), *settings["extra_args"]]

        result = load_script_module("run_loop").run_hf_job(make_command, settings["flavors"])
        return {"exit_code": 0 if result["stage"] == "COMPLETED" else 1, "actual_usd": result["cost"], "job": result["job"],
                "flavor": result["flavor"], "tried": result["tried"], "retrain_trigger": trigger,
                "note": None if result["stage"] == "COMPLETED" else f"the HF job ended as {result['stage']}"}

    def gate(self) -> dict:
        candidate_path = write_candidate_system(self.config, self.round_number, self.state)
        gate_directory = RUNS / run_name(self.config, self.round_number, "gate")
        gate_directory.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            candidate = yaml.safe_load(candidate_path.read_text())
            decision = fake_gate_decision(self.round_number, self.state, candidate)
            entry = {"exit_code": 0, "actual_usd": 0.0, "job": "fake-gate"}
        else:
            decision, entry = self.real_gate(candidate_path)
            if decision is None:
                return entry
        (gate_directory / "gate_decision.json").write_text(json.dumps(decision, indent=1, default=str))
        round_dir = round_directory(self.config, self.round_number)
        round_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(gate_directory / "gate_decision.json", round_dir / "gate_decision.json")
        if (gate_directory / "metrics.jsonl").exists():
            shutil.copy(gate_directory / "metrics.jsonl", round_dir / "metrics.jsonl")
        entry.update(run_dir=relative(gate_directory), candidate=relative(candidate_path),
                     decision={key: decision.get(key) for key in
                               ("round", "decision", "checks", "metrics_current", "metrics_candidate",
                                            "thresholds", "systems")})
        return entry

    def real_gate(self, candidate_path: Path) -> tuple[dict | None, dict]:
        from huggingface_hub import hf_hub_download

        settings, run_round = self.config["gate"], load_script_module("run_round")
        candidate = f"r{self.round_number}-weight"

        def make_command(flavor: str) -> list[str]:
            return [sys.executable, str(ROOT / "scripts" / "gate_job.py"), "--round", str(self.round_number),
                    "--current", run_round.current_system_path(), "--candidate", relative(candidate_path),
                    "--user-model", self.config["models"]["user"], "--runs-repo", settings["runs_repo"],
                    "--flavor", flavor, "--timeout", settings["timeout"], "--launch"]

        result = load_script_module("run_loop").run_hf_job(make_command, settings["flavors"])
        entry = {"exit_code": 0, "actual_usd": result["cost"] + settings.get("api_usd", 0.0), "job": result["job"],
                 "flavor": result["flavor"], "tried": result["tried"]}
        if result["stage"] != "COMPLETED":
            entry.update(exit_code=1, note=f"the HF job ended as {result['stage']}")
            return None, entry
        gate_directory = RUNS / run_name(self.config, self.round_number, "gate")
        prefix = f"gate/round-{self.round_number}/{candidate}"
        decision_path = hf_hub_download(settings["runs_repo"], f"{prefix}/gate_decision.json", repo_type="dataset",
                                        force_download=True)
        try:
            metrics_path = hf_hub_download(settings["runs_repo"], f"{prefix}/metrics.jsonl", repo_type="dataset",
                                           force_download=True)
            shutil.copy(metrics_path, gate_directory / "metrics.jsonl")
        except Exception:  # noqa: BLE001 - the metrics row is a copy for the summary; the decision is what counts
            logger.warning("no metrics.jsonl next to the gate decision", prefix=prefix)
        return json.loads(Path(decision_path).read_text()), entry


def fake_gate_decision(round_number: int, state: dict, candidate: dict) -> dict:
    """A gate decision with the real file's keys. It refuses on pass_k, unless the guards are on (round 2 of a dry
    chain), then it approves: a two-round dry run walks both the reject and the promote path."""
    approve = state["recipe"].tool_guards
    checks = [{"name": "min_valid_runs", "passed": True}, {"name": "pass_k", "passed": approve},
              {"name": "bfcl", "passed": True}]
    metrics = {"pass_1": 0.1, "pass_k": 0.05, "tool_call_error_rate": 0.5, "label_rates": {}, "bfcl": {"accuracy_points": 70}}
    return {"decision": "promote" if approve else "reject", "checks": checks, "metrics_current": metrics,
            "metrics_candidate": metrics, "thresholds": {}, "round": round_number,
            "systems": {"candidate": candidate}, "dry_run": True}


# ---------------------------------------------------------------- main

def parse_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_round.py all", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/rounds/<name>.yaml")
    parser.add_argument("--rounds", type=int, default=None, help="rounds back to back (default: the config's)")
    parser.add_argument("--dry-run", action="store_true", help="here, offline: fake models, fake HF jobs, dry-* run dirs")
    parser.add_argument("--launch", action="store_true", help="start the Cloud Run execution that runs the chain (spends)")
    parser.add_argument("--here", action="store_true", help="run the chain in this process (what the cloud execution runs)")
    return parser.parse_args(argv)


def print_plan(config: dict, rounds: int) -> None:
    caps = config["spend"]
    print(f"Round {config['round']}" + (f" to {config['round'] + rounds - 1}" if rounds > 1 else "")
          + f", owner {config['owner']}, total cap ${caps['total']:.2f}")
    for stage in round_chain.STAGES:
        print(f"  {stage:9} cap ${caps[stage]:.2f}")
    print(f"  stop rules: over {config['stop']['max_infra_error_rate']:.0%} infra errors, under "
          f"{config['stop']['min_sft_examples']} SFT examples, {config['stop']['retries']} retry per stage")
    print("\nPlan only. --launch starts one Cloud Run execution; --dry-run walks the chain here, offline.")


def launch_in_the_cloud(config: dict, config_path: Path, rounds: int) -> int:
    """One execution of shipwright-<owner> running this chain with --here. It does not wait: the laptop can close."""
    run_round = load_script_module("run_round")
    run_round.check_cloud_job_runs_this_commit(config["owner"], wait_minutes=20)
    relative_config = str(config_path.resolve().relative_to(ROOT))
    job_arguments = ["run_round.py", "all", "--config", relative_config, "--here", f"--rounds={rounds}"]
    command = run_round.gcloud_command(
        "run", "jobs", "execute", f"shipwright-{config['owner']}", f"--region={run_round.REGION}",
        f"--task-timeout={config['cloud']['task_timeout']}", "--args=^|^" + "|".join(job_arguments))
    print("starting:", " ".join(job_arguments))
    exit_code = subprocess.run(command).returncode
    print(f"\nFollow it: scripts/why.py <execution name>; results in gs://lexsi-hackathon-runs/{config['owner']}/"
          f"r{config['round']}-round/ and chain-r{config['round']}/")
    return exit_code


def run_here(config: dict, config_path: Path, rounds: int, dry_run: bool) -> int:
    if not dry_run and not in_the_image():
        raise SystemExit("a real chain runs in the cloud, not on a laptop: use --launch (or --dry-run here)")
    if config.get("git_sha") and not dry_run:
        running = os.environ.get("SHIPWRIGHT_GIT_SHA", "")
        if not running.startswith(config["git_sha"]) and not config["git_sha"].startswith(running or "-"):
            raise SystemExit(f"the config pins commit {config['git_sha']}, this image is {running or 'unknown'}")
    if dry_run:
        config = dry_run_config(config)
    resolved = resolved_config(config, config_path, rounds, dry_run)
    hooks = round_chain.ChainHooks(
        stage_functions=lambda round_number, state: Stages(config, round_number, state, dry_run).functions(),
        save_manifest=lambda round_number, manifest: save_manifest(config, round_number, manifest),
        load_manifest=lambda round_number: load_manifest(config, round_number),
        promote=lambda round_number, decision: promote(config, round_number, decision),
        write_next_round=lambda round_number, decision, chosen: write_next_round(config, round_number, decision, chosen),
        log=lambda message: logger.info(message))
    result = round_chain.run_chain(config, rounds, hooks, resolved)

    chain_directory = RUNS / f"{config.get('name_prefix', '')}chain-r{config['round']}"
    chain_directory.mkdir(parents=True, exist_ok=True)
    report = {"status": result.status, "reason": result.reason, "spent_usd": result.spent_usd,
              "total_cap_usd": config["spend"]["total"], "config": resolved,
              "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rounds": [{"round": r.round_number, "status": r.status, "reason": r.reason, "spent_usd": round(r.spent_usd, 4),
                          "manifest": f"runs/{round_directory(config, r.round_number).name}/round.json"}
                         for r in result.rounds]}
    (chain_directory / "chain.json").write_text(json.dumps(report, indent=2, default=str))
    upload_run_dir(config, chain_directory)
    print(f"\nChain: {result.status.upper()} ({result.reason}). Spent ${result.spent_usd:.2f} of "
          f"${config['spend']['total']:.2f}. Summary: {chain_directory.relative_to(ROOT)}/chain.json")
    return 1 if result.status == round_chain.STOPPED else 0


def main(argv: list[str]) -> int:
    options = parse_options(argv)
    config = round_chain.load_config(options.config)
    rounds = options.rounds or config.get("rounds", 1)
    if options.here or options.dry_run:
        return run_here(config, options.config, rounds, options.dry_run)
    print_plan(config, rounds)
    if options.launch:
        return launch_in_the_cloud(config, options.config, rounds)
    return 0


if __name__ == "__main__":
    sys.exit(log.run_script(lambda: main(sys.argv[1:]), step="run_round_all", meta=True))

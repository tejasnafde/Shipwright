"""Every stage of a round, as one method each. ``shipwright.loop.controller`` decides which run; this file only runs them.

API stages run a script of ``scripts/`` in this process's tree (in the cloud: the execution's own container), each with
its spend cap as ``--max-cost-usd``. GPU stages are HF Jobs (``shipwright.loop.jobs``), priced by their running seconds.
A dry run swaps the models for the scripts' fake LLM and the HF jobs for fakes, and writes run dirs named dry-*.

    baseline         Tiny Aya (no adapter) on the train tasks                       Cloud Run, HF router
    candidate_train  the refused candidate on the train tasks the teacher can solve  HF Job (vLLM serves the adapter)
    classify         AgentTune FailureDetector + FailureClassifier + our rules       Vertex classifier
    teacher          the teacher replays the failed tasks; passes are kept           Vertex teacher
    corrections      the teacher's reply at each turn the candidate got wrong        Vertex teacher
    data             SFT rows from teacher passes, corrections, general examples    pushed to the SFT repo, round-N
    train            AgentTune RetrainingTrigger, then LoRA with AgentTune           HF Job
    bfcl_screen      BFCL alone on the candidate: the gate is paid for only if it can pass   HF Job
    gate             the candidate against the deployed system, 8 checks             HF Job
    ceiling, smoke   side stages, by hand: the teacher on the held-out tasks; 5 train tasks
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from shipwright import cloud
from shipwright.heal import systems
from shipwright.heal.traces import teacher_pass_rates
from shipwright.loop import jobs
from shipwright.loop.config import gpu_settings
from shipwright.loop.controller import ChainState
from shipwright.loop.workspace import Workspace, read_json, relative, write_system
from shipwright.observability import log

ROOT = cloud.ROOT
logger = log.get_logger("loop")
CLASSIFY_ESTIMATE_USD = 0.05  # classify.py does not meter its calls: its estimate counts as spent
HEAL_CONFIG = ROOT / "configs" / "heal.yaml"


def run_script(arguments: list[str]) -> int:
    """scripts/<script> args, in this tree. Returns the exit code."""
    logger.info("stage command", command=arguments)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / arguments[0]), *arguments[1:]]).returncode


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text().splitlines() if line.strip()])


def api_stage_entry(exit_code: int, run_directory: Path) -> dict:
    summary = read_json(run_directory / "summary.json")
    return {"exit_code": exit_code, "run_dir": relative(run_directory),
            "actual_usd": (summary.get("cost_estimate_usd") or {}).get("total") or 0.0,
            "num_runs": summary.get("num_runs"), "num_passed": summary.get("num_passed"),
            "num_infra_errors": summary.get("num_infra_errors"), "cost_cap_hit": summary.get("cost_cap_hit")}


def copy_real_files(source: Path, destination: Path) -> None:
    """Copy a Hub download with real files: its links into the cache would dangle (round 4, 2026-11-12)."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=False)


def hub_file(prefix: str, name: str) -> Path | None:
    """A file the gate job pushed to the runs dataset, or None when it is not there."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(cloud.RUNS_REPO, f"{prefix}/{name}", repo_type="dataset", force_download=True,
                                    token=cloud.hf_token()))
    except EntryNotFoundError:
        return None


def fake_gate_decision(round_number: int, state: ChainState, candidate: dict) -> dict:
    """A gate decision with the real file's keys. It refuses on pass_k unless the guards are on, then it approves: a
    two-round dry run walks both the refusal and the promotion path."""
    approve = state.recipe.tool_guards
    checks = [{"name": "min_valid_runs", "passed": True}, {"name": "pass_k", "passed": approve},
              {"name": "bfcl", "passed": True}]
    metrics = {"pass_1": 0.1, "pass_k": 0.05, "tool_call_error_rate": 0.5, "label_rates": {},
               "bfcl": {"accuracy_points": 70}}
    return {"decision": "promote" if approve else "reject", "checks": checks, "metrics_current": metrics,
            "metrics_candidate": metrics, "thresholds": {"bfcl_max_drop_points": 2.0}, "round": round_number,
            "systems": {"candidate": candidate}, "dry_run": True}


class Stages:
    def __init__(self, config: dict, workspace: Workspace, round_number: int, state: ChainState, dry_run: bool):
        self.config = config
        self.workspace = workspace
        self.round_number = round_number
        self.state = state
        self.dry_run = dry_run
        self.models = config["models"]
        self.caps = config["spend"]

    def functions(self) -> dict:
        return {"baseline": self.baseline, "candidate_train": self.candidate_train, "classify": self.classify,
                "teacher": self.teacher, "corrections": self.corrections, "data": self.data, "train": self.train,
                "bfcl_screen": self.bfcl_screen, "gate": self.gate, "ceiling": self.ceiling, "smoke": self.smoke}

    def runs(self, name: str | None) -> Path | None:
        if name is None:
            return None
        return self.workspace.runs / name

    def dry_flag(self) -> list[str]:
        return ["--dry-run"] if self.dry_run else []

    # ------------------------------------------------------------ API stages

    def baseline(self) -> dict:
        output = self.runs(self.state.baseline_run)
        settings = self.config["baseline"]
        arguments = ["baseline.py", "--task-split", "train", "--num-tasks", str(settings["num_tasks"]),
                     "--trials", str(settings["trials"]), "--agent-model", self.models["agent"],
                     "--user-model", self.models["user"], "--max-concurrency", str(settings["max_concurrency"]),
                     "--seed", str(self.config["seed"]), "--max-cost-usd", str(self.caps["baseline"]),
                     "--output-dir", str(output), *self.dry_flag()]
        return api_stage_entry(run_script(arguments), output)

    def classify(self) -> dict:
        failures = self.runs(self.state.failures_run)
        arguments = ["classify.py", str(failures)]
        if self.dry_run:
            arguments.append("--no-llm")
        else:
            arguments += ["--classifier-model", self.models["classifier"]]
        exit_code = run_script(arguments)
        summary = read_json(failures / "failure_summary.json")
        actual = 0.0 if self.dry_run else CLASSIFY_ESTIMATE_USD
        return {"exit_code": exit_code, "run_dir": relative(failures), "actual_usd": actual,
                "num_failed": summary.get("num_failed"), "per_route": summary.get("per_route"),
                "per_label": summary.get("per_label")}

    def teacher(self) -> dict:
        output = self.runs(self.state.teacher_run)
        settings = self.config["teacher"]
        arguments = ["teacher_runs.py", "--teacher-model", self.models["teacher"], "--user-model", self.models["user"],
                     "--from-run", str(self.runs(self.state.failures_run)), f"--trials={settings['trials']}",
                     "--max-concurrency", str(settings["max_concurrency"]), "--seed", str(self.config["seed"]),
                     "--max-cost-usd", str(self.caps["teacher"]), "--output-dir", str(output), *self.dry_flag()]
        arguments += self.teacher_selection(settings)
        entry = api_stage_entry(run_script(arguments), output)
        entry["kept_traces"] = count_lines(output / "kept.jsonl")
        return entry

    def teacher_selection(self, settings: dict) -> list[str]:
        """Optional teacher modes (config ``teacher``): earlier teacher runs ("owner/dir") whose tasks to skip, or the
        harvest, which plays only tasks they solved (a solved task passes again about 48% of the time)."""
        arguments = []
        for key, flag in (("skip_unsolved_from", "--skip-unsolved-from"), ("only_solved_from", "--only-solved-from"),
                          ("skip_solved_from", "--skip-solved-from")):
            names = settings.get(key) or []
            if names:
                for name in names:
                    self.workspace.fetch(name)
                arguments += [flag, *[str(self.workspace.runs / name.rsplit("/", 1)[-1]) for name in names]]
        if settings.get("min_teacher_rate"):
            arguments += ["--min-teacher-rate", str(settings["min_teacher_rate"])]
        return arguments

    def corrections(self) -> dict:
        """The teacher's next reply at each turn the candidate got wrong (scripts/build_corrections.py)."""
        output = self.workspace.path(self.round_number, "corrections")
        if self.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            (output / "corrections.jsonl").write_text("")
            return {"exit_code": 0, "actual_usd": 0.0, "run_dir": relative(output), "rows": 0, "note": "dry run"}
        arguments = ["build_corrections.py", "--failures", str(self.runs(self.state.failures_run)),
                     "--teacher-model", self.models["teacher"], "--max-cost-usd", str(self.caps["corrections"]),
                     "--output-dir", str(output)]
        exit_code = run_script(arguments)
        statistics = read_json(output / "corrections_stats.json")
        entry = {"exit_code": exit_code, "actual_usd": statistics.get("cost_usd", 0.0), "run_dir": relative(output),
                 "rows": statistics.get("rows"), "wrong_turns_found": statistics.get("wrong_turns_found")}
        if exit_code == 0 and not statistics.get("rows"):
            entry.update(exit_code=1, no_retry=True, note="the teacher gave no usable correction")
        return entry

    # ------------------------------------------------------------ data

    def pool_directories(self) -> list[Path] | None:
        """The teacher runs whose passes are training data: this chain's own and the config's pool. None if one is
        missing (a pool the config names but nobody can find is a config error, not a smaller dataset)."""
        names = list(self.config["data"]["pool"])
        missing = [name for name in names if not self.workspace.fetch(name)]
        if missing:
            return None
        directories = [self.workspace.runs / name.rsplit("/", 1)[-1] for name in names]
        own_teacher = self.runs(self.state.teacher_run)
        if own_teacher is not None and (self.workspace.fetch(own_teacher.name) or own_teacher.is_dir()):
            directories.insert(0, own_teacher)
        return directories

    def all_corrections(self) -> Path:
        """This round's corrections and every earlier round's, in one file: the adapter is retrained from the base model
        each time, so a lesson of an earlier round must stay in the data."""
        combined = self.workspace.path(self.round_number, "corrections") / "all_corrections.jsonl"
        combined.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        lines: list[str] = []
        for round_number in range(self.round_number, 0, -1):
            directory = self.workspace.path(round_number, "corrections")
            self.workspace.fetch(directory.name)
            path = directory / "corrections.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if line.strip() and line not in seen:
                    seen.add(line)
                    lines.append(line)
        combined.write_text("".join(line + "\n" for line in lines))
        return combined

    def general_replay(self, per_category: int) -> Path:
        """General function-calling examples (scripts/build_bfcl_replay.py), built once per size."""
        output = self.workspace.runs / f"bfcl-replay-{per_category}"
        if not (output / "replay.jsonl").exists():
            exit_code = run_script(["build_bfcl_replay.py", "--per-category", str(per_category), "--output-dir", str(output)])
            if exit_code != 0:
                raise RuntimeError(f"build_bfcl_replay.py exited {exit_code}")
        return output / "replay.jsonl"

    def data(self) -> dict:
        recipe = self.state.recipe
        output = self.workspace.path(self.round_number, "sft")
        teacher_directories = self.pool_directories()
        if teacher_directories is None:
            return {"exit_code": 1, "actual_usd": 0.0, "no_retry": True,
                    "note": f"a pooled teacher run is missing: {self.config['data']['pool']}"}
        arguments = ["build_sft.py", "--output-dir", str(output), "--only-model", self.models["teacher"],
                     "--revision", f"round-{self.round_number}", "--repo", self.config["data"]["repo"],
                     "--boost", str(recipe.boost)]
        failures = self.runs(self.state.failures_run)
        if failures is not None and (failures / "failures.jsonl").exists():
            arguments += ["--boost-from", str(failures)]
        for directory in teacher_directories:
            arguments += ["--teacher", str(directory)]
        if recipe.general_replay and not self.dry_run:
            arguments += ["--general-replay", str(self.general_replay(recipe.general_replay))]
        if recipe.corrections_rounds:
            arguments += ["--corrections", str(self.all_corrections())]
        if not self.dry_run:
            arguments.append("--push")
        if not teacher_directories:
            if not self.dry_run:
                return {"exit_code": 1, "actual_usd": 0.0, "no_retry": True, "note": "no teacher runs to train on"}
            output.mkdir(parents=True, exist_ok=True)  # a dry run has no teacher passes: an empty dataset
            (output / "sft_stats.json").write_text(json.dumps({"examples": 0, "tasks": []}))
            exit_code = 0
        else:
            exit_code = run_script(arguments)
        statistics = read_json(output / "sft_stats.json")
        return {"exit_code": exit_code, "actual_usd": 0.0, "run_dir": relative(output),
                "examples": statistics.get("examples"), "tasks": len(statistics.get("tasks") or []),
                "general_replay": statistics.get("general_replay"), "corrections": statistics.get("corrections")}

    # ------------------------------------------------------------ GPU stages

    def retraining_trigger(self) -> dict:
        """AgentTune's RetrainingTrigger on the failures this round learns from (configs/heal.yaml)."""
        from shipwright.heal.agenttune_bridge import retrain_status_for_run

        status = retrain_status_for_run(self.runs(self.state.failures_run),
                                        self.workspace.path(self.round_number, "sft") / "sft.jsonl", HEAL_CONFIG)
        return {key: status.get(key) for key in ("should_train", "reason", "num_weight_fix_failures", "num_with_correction")}

    def train(self) -> dict:
        try:
            trigger = self.retraining_trigger()
        except Exception as error:  # noqa: BLE001 - recorded; require_trigger decides whether it blocks
            trigger = {"should_train": None, "reason": f"{type(error).__name__}: {error}"}
        if self.config["train"]["require_trigger"] and not trigger.get("should_train"):
            return {"exit_code": 1, "actual_usd": 0.0, "retrain_trigger": trigger, "no_retry": True,
                    "note": f"AgentTune's RetrainingTrigger says do not train ({trigger.get('reason')})"}
        if self.dry_run:
            return {"exit_code": 0, "actual_usd": 0.0, "job": "fake-train", "retrain_trigger": trigger}
        recipe, settings, revision = self.state.recipe, self.config["train"], f"round-{self.round_number}"
        training_arguments = ["--epochs", f"{recipe.epochs:g}", "--lr", f"{recipe.learning_rate:g}",
                              "--seed", str(self.config["seed"]), *settings.get("extra_args", [])]

        def launch(flavor: str) -> str:
            return jobs.launch_train(jobs.TrainJob(dataset_revision=revision, revision=revision, flavor=flavor,
                                                   timeout=settings["timeout"], dataset=self.config["data"]["repo"],
                                                   output_repo=settings["output_repo"],
                                                   training_arguments=training_arguments))

        result = jobs.run_on_first_free_gpu(launch, settings["flavors"])
        entry = gpu_entry(result)
        entry["retrain_trigger"] = trigger
        return entry

    def write_candidate(self) -> Path:
        """The candidate's system config, in the round dir. A round that trained serves its own adapter (from the
        config's template); a harness-only round serves the last candidate with the recipe's guards."""
        trained_this_round = self.state.adapter_round == self.round_number
        if trained_this_round or not self.state.candidate_system:
            system = yaml.safe_load((ROOT / self.config["gate"]["candidate_template"]).read_text())
            system["adapter"] = {"repo": self.config["train"]["output_repo"],
                                 "revision": f"round-{self.state.adapter_round}"}
        else:
            system = yaml.safe_load((ROOT / self.state.candidate_system).read_text())
        system["name"] = f"r{self.round_number}-weight"
        system["tool_guards"] = self.state.recipe.tool_guards
        for key, value in (self.config["gate"].get("candidate_extra") or {}).items():
            system[key] = value
        systems.validate_system(system)
        path = self.workspace.path(self.round_number, "round") / f"{system['name']}.yaml"
        reasons = "; ".join(self.state.recipe.reasons) or "the first round's recipe"
        write_system(path, system, f"round {self.round_number} candidate. Why: {reasons}")
        self.state.candidate_system = relative(path)
        return path

    def gate_job(self, mode: str, stage: str, candidate_path: Path, task_ids: list[str] | None = None) -> jobs.JobResult:
        settings = gpu_settings(self.config, stage)
        current_path = self.workspace.deployed_system_path()
        trials = (self.config.get("candidate_train") or {}).get("trials", 1)

        def launch(flavor: str) -> str:
            job = jobs.GateJob(round_number=self.round_number, current=current_path, candidate=candidate_path,
                               user_model=self.models["user"], flavor=flavor, timeout=settings["timeout"], mode=mode,
                               task_ids=task_ids or [], trials=trials, runs_repo=self.config["gate"]["runs_repo"])
            return jobs.launch_gate(job, needs_vertex_token=(mode == "gate"))

        return jobs.run_on_first_free_gpu(launch, settings["flavors"])

    def candidate_train(self) -> dict:
        """The refused candidate on the train tasks the teacher can solve: its own failures become the corrections."""
        output = self.workspace.path(self.round_number, "cand-train")
        source_path = ROOT / self.state.candidate_system
        source = yaml.safe_load(source_path.read_text())
        pool = self.pool_directories() or []
        rates = teacher_pass_rates([directory for directory in pool if (directory / "traces.jsonl").exists()])
        task_ids = sorted((task_id for task_id, rate in rates.items() if rate > 0), key=int)
        if self.dry_run:
            exit_code = run_script(["baseline.py", "--task-split", "train", "--num-tasks", "2", "--trials", "1",
                                    "--output-dir", str(output), "--dry-run"])
            entry = api_stage_entry(exit_code, output)
            entry.update(task_ids=len(task_ids), source=relative(source_path))
            return entry
        if not task_ids:
            return {"exit_code": 1, "actual_usd": 0.0, "no_retry": True, "note": "the teacher pool solves no train task"}
        prefix = f"gate/round-{self.round_number}/{source['name']}-train"
        result = None
        earlier_inputs = hub_file(prefix, "gate_inputs.json")
        already_run = (earlier_inputs is not None and hub_file(f"{prefix}/{source['name']}", "traces.jsonl") is not None
                       and same_candidate(json.loads(earlier_inputs.read_text()), source_path))
        if not already_run:
            result = self.gate_job("train_eval", "candidate_train", source_path, task_ids)
            if not result.completed:
                return gpu_entry(result)
        from huggingface_hub import snapshot_download

        downloaded = Path(snapshot_download(cloud.RUNS_REPO, repo_type="dataset", token=cloud.hf_token(),
                                            allow_patterns=[f"{prefix}/{source['name']}/*"]))
        copy_real_files(downloaded / prefix / source["name"], output)
        entry = api_stage_entry(0, output)
        if result is not None:
            entry.update(gpu_entry(result), run_dir=relative(output))
        else:
            entry.update(actual_usd=0.0, note="already on the Hub: not launched again")
        entry.update(task_ids=len(task_ids), source=relative(source_path))
        return entry

    def bfcl_screen(self) -> dict:
        """BFCL alone on the candidate (about 8 minutes). Fails the stage when the gate's BFCL check could not pass,
        against the deployed system's score in the last gate; with no gate yet, the score is recorded only."""
        candidate_path = self.write_candidate()
        if self.dry_run:
            return {"exit_code": 0, "actual_usd": 0.0, "job": "fake-bfcl", "bfcl_points": 70}
        result = self.gate_job("bfcl_screen", "bfcl_screen", candidate_path)
        if not result.completed:
            return gpu_entry(result)
        name = f"r{self.round_number}-weight"
        score_file = hub_file(f"gate/round-{self.round_number}/{name}-bfcl/{name}", "bfcl.json")
        entry = gpu_entry(result)
        if score_file is None:
            entry.update(exit_code=1, note="the BFCL screen pushed no bfcl.json")
            return entry
        points = json.loads(score_file.read_text())["accuracy_points"]
        entry["bfcl_points"] = points
        if not self.state.decisions:
            return entry
        last = self.state.decisions[-1]
        needed = last["metrics_current"]["bfcl"]["accuracy_points"] - last["thresholds"]["bfcl_max_drop_points"]
        entry["needed_points"] = needed
        if points < needed:
            entry.update(exit_code=1, no_retry=True,
                         note=f"BFCL screen: {points} points, the gate needs {needed:.2f}: the gate is not run")
        return entry

    def gate(self) -> dict:
        candidate_path = self.write_candidate()
        gate_directory = self.workspace.path(self.round_number, "gate")
        gate_directory.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            decision = fake_gate_decision(self.round_number, self.state, yaml.safe_load(candidate_path.read_text()))
            entry = {"exit_code": 0, "actual_usd": 0.0, "job": "fake-gate"}
        else:
            prefix = f"gate/round-{self.round_number}/r{self.round_number}-weight"
            decision_file = hub_file(prefix, "gate_decision.json")
            entry = {"exit_code": 0, "actual_usd": 0.0, "note": "already on the Hub: not launched again"}
            if decision_file is not None and not same_candidate(json.loads(decision_file.read_text()), candidate_path):
                decision_file = None  # another experiment used this round number: gate this candidate
            if decision_file is None:
                result = self.gate_job("gate", "gate", candidate_path)
                entry = gpu_entry(result)
                entry["actual_usd"] = round(entry["actual_usd"] + self.config["gate"].get("api_usd", 0.0), 4)
                if not result.completed:
                    return entry
                decision_file = hub_file(prefix, "gate_decision.json")
                if decision_file is None:
                    entry.update(exit_code=1, note="the gate job pushed no gate_decision.json")
                    return entry
            decision = json.loads(decision_file.read_text())
            metrics_file = hub_file(prefix, "metrics.jsonl")
            if metrics_file is not None:
                shutil.copy(metrics_file, gate_directory / "metrics.jsonl")
        (gate_directory / "gate_decision.json").write_text(json.dumps(decision, indent=1, default=str))
        round_directory = self.workspace.path(self.round_number, "round")
        shutil.copy(gate_directory / "gate_decision.json", round_directory / "gate_decision.json")
        if (gate_directory / "metrics.jsonl").exists():
            shutil.copy(gate_directory / "metrics.jsonl", round_directory / "metrics.jsonl")
        entry.update(run_dir=relative(gate_directory), candidate=relative(candidate_path),
                     decision={key: decision.get(key) for key in ("round", "decision", "checks", "metrics_current",
                                                                  "metrics_candidate", "thresholds", "systems")})
        return entry

    # ------------------------------------------------------------ side stages (by hand, never in a round)

    def ceiling(self) -> dict:
        """The teacher on the held-out tasks: the pass rate the student is read against (evaluation only)."""
        output = self.workspace.path(self.round_number, "ceiling")
        arguments = ["teacher_runs.py", "--ceiling", "--teacher-model", self.models["teacher"],
                     "--user-model", self.models["user"], "--trials=3",
                     "--max-concurrency", str(self.config["teacher"]["max_concurrency"]),
                     "--max-cost-usd", str(self.caps.get("ceiling", 1.0)), "--output-dir", str(output), *self.dry_flag()]
        entry = api_stage_entry(run_script(arguments), output)
        entry["teacher_pass_rate"] = read_json(output / "summary.json").get("pass_rate_excluding_infra_errors")
        return entry

    def smoke(self) -> dict:
        """5 train tasks, 1 trial: run it after any change to the user model or its provider."""
        output = self.workspace.path(self.round_number, "smoke")
        arguments = ["baseline.py", "--task-split", "train", "--num-tasks", "5", "--trials", "1",
                     "--agent-model", self.models["agent"], "--user-model", self.models["user"],
                     "--max-cost-usd", str(self.caps.get("smoke", 0.3)), "--output-dir", str(output), *self.dry_flag()]
        return api_stage_entry(run_script(arguments), output)


def same_candidate(decision: dict, candidate_path: Path) -> bool:
    """True when a gate decision found on the Hub gated exactly this candidate (adapter and serving switches), so an
    execution that died after its gate job can resume without paying for the gate twice."""
    gated = (decision.get("systems") or {}).get("candidate") or {}
    ours = yaml.safe_load(candidate_path.read_text())
    for key in ("adapter", "tool_guards", "state_reminders"):
        if bool(gated.get(key)) != bool(ours.get(key)) or (gated.get(key) or None) != (ours.get(key) or None):
            return False
    return True


def gpu_entry(result: jobs.JobResult) -> dict:
    entry = {"exit_code": 0 if result.completed else 1, "actual_usd": result.cost_usd, "job": result.job_id,
             "flavor": result.flavor, "tried": result.tried}
    if not result.completed:
        entry["note"] = f"the HF job ended as {result.stage}"
    return entry

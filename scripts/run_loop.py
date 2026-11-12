"""One more turn of the self-healing loop, started from the gate's refusal of the last round.

    .venv/bin/python scripts/run_loop.py --round 4 --from-round 3 --budget-usd 5          # plan: free
    .venv/bin/python scripts/run_loop.py --round 4 --from-round 3 --budget-usd 5 --launch # runs, spends

Round N learns from what round N-1's candidate did wrong, instead of training the same thing again:

  0. read the gate decision of round N-1; a promotion is recorded (scripts/promote.py) and the loop ends
  1. recipe       what to change, from why the gate refused (shipwright.heal.recipe)
  2. candidate_train  the refused candidate runs on the train tasks the teacher can solve (GPU job, no gate)
  3. classify     AgentTune + our rules label why it failed there (Cloud Run job)
  4. data         the pooled teacher examples, with the examples of the tasks it failed repeated (recipe.boost)
  5. train        AgentTune's retraining trigger decides, then a fresh LoRA on that data (HF Job)
  5b. bfcl_screen (with --screen-bfcl) the candidate's BFCL score alone, about 5 minutes: the full gate is only paid for
                  if that check can pass
  6. gate         the new candidate against the deployed system on the held-out tasks (HF Job)
  7. promote      only on a gate "promote": scripts/promote.py, with rollback

--failures-run NAME reuses a classified candidate-on-train run (steps 2 and 3 are skipped), so a round can change one
thing in the data and nothing else.

With --max-rounds (default 3) it goes round after round, each started from the gate decision of the one before, until one
is promoted, the budget cannot cover another round, or a step fails. Every step is written to runs/round-N.json as it finishes and is skipped on a rerun, so a failure midway is resumed
with the same command. A step starts only if its estimate fits in what is left of --budget-usd; each HF Job has a
time limit that bounds its worst case. Without --launch nothing is run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shipwright.control_room import gcs_runs  # noqa: E402
from shipwright.heal import promotion, recipe as recipe_module  # noqa: E402
from shipwright.llm.vertex import ROUTER_USER_MODEL, TEACHER_MODEL  # noqa: E402
from shipwright.observability import log  # noqa: E402

logger = log.get_logger("run_loop")

_specification = importlib.util.spec_from_file_location("run_round", ROOT / "scripts" / "run_round.py")
run_round = importlib.util.module_from_spec(_specification)
sys.modules["run_round"] = run_round
_specification.loader.exec_module(run_round)

RUNS_REPO = "mitushigoyal5/shipwright-runs"
CANDIDATE_TRAIN_FLAVOR, CANDIDATE_TRAIN_TIMEOUT = "a10g-large", "40m"
TRAIN_FLAVOR, TRAIN_TIMEOUT = run_round.TRAIN_FLAVOR, "45m"
GATE_FLAVOR, GATE_TIMEOUT = run_round.GATE_FLAVOR, "60m"
HF_PRICES = {"a10g-large": 1.50, "l40sx1": 1.80, "a100-large": 2.50}
# What each step is expected to cost (round 3: the gate 0.75, training 0.62, the classify stage 0.05).
STEP_ESTIMATES = {"candidate_train": 0.6, "classify": 0.05, "corrections": 0.3, "data": 0.0, "train": 1.2,
                  "bfcl_screen": 0.25, "gate": 0.9}
CORRECTIONS_COST_CAP_USD = 0.4
STEPS = list(STEP_ESTIMATES)
BFCL_SCREEN_TIMEOUT = "30m"
DEFAULT_POOL = ["tejas/r1-teacher-t1", "tejas/r1-teacher-all-t2", "tejas/r1-teacher-ds-a", "tejas/r1-teacher-ds-b",
                "tejas/r1-teacher-ds-c", "tejas/bake-dsv41flash", "r2-teacher", "r3-teacher"]
JOB_POLL_SECONDS = 45


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, required=True, help="the round to run (N)")
    parser.add_argument("--from-round", type=int, required=True, help="the earlier round whose gate decision it starts from")
    parser.add_argument("--budget-usd", type=float, required=True,
                        help="the most this run of the loop may spend, all rounds and steps together")
    parser.add_argument("--max-rounds", type=int, default=3, help="stop after this many rounds if none is promoted")
    parser.add_argument("--owner", default="ishank")
    parser.add_argument("--user-model", default=ROUTER_USER_MODEL)
    parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    parser.add_argument("--pool", nargs="*", default=DEFAULT_POOL, help="teacher run dirs whose passes make the data")
    parser.add_argument("--redo", nargs="*", default=[], choices=STEPS, help="run these steps again")
    parser.add_argument("--failures-run", default=None,
                        help="a classified candidate-on-train run (e.g. r4-cand-train) to take the failed tasks from: skips "
                             "the candidate_train and classify steps")
    parser.add_argument("--screen-bfcl", action="store_true",
                        help="score BFCL alone before the gate; stop if that check cannot pass")
    parser.add_argument("--gate-flavor", default=GATE_FLAVOR,
                        help="GPU of the gate job: a busy pool can leave an A10G waiting for an hour, an l40sx1 costs $0.30/h more")
    parser.add_argument("--launch", action="store_true", help="run the steps (spends credits); without it, plan only")
    options = parser.parse_args()
    # What run_round's cloud stage reads from its options.
    options.dry_run, options.wait_for_image_minutes, options.max_concurrency = False, 30, 10
    return options


# ---------------------------------------------------------------- reading what happened

def candidate_name(round_number: int) -> str:
    return f"r{round_number}-weight"


def fetch_gate_decision(round_number: int) -> dict:
    """gate_decision.json of a round, from the runs dataset the gate job pushed it to."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        RUNS_REPO, f"gate/round-{round_number}/{candidate_name(round_number)}/gate_decision.json",
        repo_type="dataset", force_download=True)
    return json.loads(Path(path).read_text())


def previous_recipe(round_number: int) -> recipe_module.Recipe:
    manifest_path = run_round.manifest_path(round_number)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text()).get("recipe")
        if saved:
            recipe = recipe_module.recipe_from_dict(saved)
            recipe.reasons = []
            return recipe
    return recipe_module.Recipe()


def solvable_task_ids(pool: list[str]) -> list[str]:
    """Train tasks the teacher has solved at least once: the only ones that can give a corrective example."""
    import teacher_runs

    pool_directories = [ROOT / "runs" / run_round.local_run_name(name) for name in pool]
    rates = teacher_runs.teacher_pass_rates(pool_directories)
    return sorted((task_id for task_id, rate in rates.items() if rate > 0), key=int)


# ---------------------------------------------------------------- HF jobs

def launched_job_id(command: list[str]) -> str:
    """Run a launcher script (train_job.py or gate_job.py with --launch) and return the HF job id it printed."""
    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(re.sub(r"\r[^\n]*", "", completed.stdout)[-1500:])
    match = re.search(r"Launched job (\w+)", completed.stdout + completed.stderr)
    if completed.returncode != 0 or not match:
        raise SystemExit(f"launch failed (exit {completed.returncode}): {(completed.stderr or completed.stdout)[-600:]}")
    return match.group(1)


def copy_run_directory(source: Path, destination: Path) -> None:
    """Copy a run dir with its real files. The Hub download is full of links into its cache: copying those as links
    leaves a folder of dangling links (round 4, 2026-11-12)."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=False)


def results_are_on_the_hub(round_number: int, source: str) -> bool:
    """True when an earlier launch of this step already pushed its traces, so the job need not run again."""
    from huggingface_hub import HfApi

    return HfApi().file_exists(
        RUNS_REPO, f"gate/round-{round_number}/{source}-train/{source}/traces.jsonl", repo_type="dataset")


def wait_for_job(job_id: str, queue_limit_seconds: float | None = None, hf_api=None,
                 sleep=time.sleep, clock=time.monotonic) -> tuple[str, float]:
    """Block until an HF job ends. Returns (final stage, cost in USD from its running seconds and flavor price).

    A job that sits in the GPU queue longer than ``queue_limit_seconds`` is cancelled (nothing was billed) and the stage
    "QUEUE_TIMEOUT" comes back, so the caller can try another GPU."""
    if hf_api is None:
        from huggingface_hub import HfApi

        hf_api = HfApi()
    queued_since = clock()
    while True:
        job = hf_api.inspect_job(job_id=job_id)
        stage = str(job.status.stage)
        if stage in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            seconds = job.durations.running_secs if job.durations else 0  # not the time spent waiting for a GPU
            return stage, round(seconds / 3600 * HF_PRICES.get(str(job.flavor), 2.5), 4)
        if stage != "SCHEDULING":
            queued_since = clock()  # it is running: the queue limit does not apply any more
        elif queue_limit_seconds is not None and clock() - queued_since > queue_limit_seconds:
            hf_api.cancel_job(job_id=job_id)
            logger.warning("cancelled a job that waited too long for a GPU", job=job_id, waited_s=round(clock() - queued_since))
            return "QUEUE_TIMEOUT", 0.0
        logger.info("waiting for the HF job", job=job_id, stage=stage)
        sleep(JOB_POLL_SECONDS)


QUEUE_LIMIT_MINUTES = 12
GATE_FLAVORS = ["a10g-large", "l40sx1"]      # the cheaper GPU first; the other one if the queue is long
TRAIN_FLAVORS = ["l40sx1", "a100-large"]


def run_hf_job(make_command, flavors: list[str], queue_limit_minutes: float = QUEUE_LIMIT_MINUTES, launch=None, wait=None) -> dict:
    """Launch a job on the first GPU type that gets a machine within the queue limit; the last one waits as long as it takes.

    ``make_command(flavor)`` builds the launcher command. Returns {"stage", "cost", "job", "flavor", "tried"}. A failed job is
    not retried here: a deterministic error would only cost twice, and the step reports it and stops."""
    launch = launch or launched_job_id
    wait = wait or wait_for_job
    tried = []
    for position, flavor in enumerate(flavors):
        is_last = position == len(flavors) - 1
        job_id = launch(make_command(flavor))
        stage, cost = wait(job_id, None if is_last else queue_limit_minutes * 60)
        tried.append({"flavor": flavor, "job": job_id, "stage": stage})
        if stage != "QUEUE_TIMEOUT":
            return {"stage": stage, "cost": cost, "job": job_id, "flavor": flavor, "tried": tried}
    raise SystemExit("no GPU type was tried")  # flavors was empty


# ---------------------------------------------------------------- the manifest

def load_loop_manifest(options: argparse.Namespace) -> dict:
    path = run_round.manifest_path(options.round)
    if path.exists():
        return json.loads(path.read_text())
    return {"round": options.round, "owner": options.owner, "user_model": options.user_model,
            "teacher_model": options.teacher_model, "round_budget_usd": options.budget_usd, "stages": {},
            "started_from_round": options.from_round}


def spent_so_far(manifest: dict) -> float:
    stages = sum(stage.get("actual_usd") or 0.0 for stage in manifest["stages"].values())
    return stages + sum(other["usd"] for other in manifest.get("other_costs", []))


def check_fits_in_budget(manifest: dict, step: str, budget_usd: float, spent_in_earlier_rounds: float = 0.0) -> None:
    spent, estimate = spent_in_earlier_rounds + spent_so_far(manifest), STEP_ESTIMATES[step]
    if spent + estimate > budget_usd:
        raise SystemExit(f"{step}: its estimate ${estimate:.2f} on top of ${spent:.2f} spent passes the budget ${budget_usd:.2f}")


def planned_steps(options: argparse.Namespace, chosen: recipe_module.Recipe | None = None,
                  previous: recipe_module.Recipe | None = None) -> list[str]:
    """The steps this round runs. A remedy that changes only the harness (the guards) runs only the gate. Otherwise: not
    the candidate_train and classify steps when the failures are given, the corrections only when the recipe has them,
    and the BFCL screen only when asked for."""
    if chosen is not None and previous is not None and not recipe_module.needs_training(chosen, previous):
        return ["gate"]
    skipped = set()
    if options.failures_run:
        skipped |= {"candidate_train", "classify"}
    if not options.screen_bfcl:
        skipped.add("bfcl_screen")
    if chosen is None or not chosen.corrections:
        skipped.add("corrections")
    return [step for step in STEPS if step not in skipped]


def cost_of_a_full_round(last_round_cost: float | None, steps: list[str] | None = None) -> float:
    """What the next round needs before it starts: its steps, or 10% more than the last round really cost."""
    estimate = sum(STEP_ESTIMATES[step] for step in (steps or STEPS))
    if last_round_cost is None:
        return estimate
    return max(estimate, last_round_cost * 1.1)


def record(manifest: dict, round_number: int, step: str, entry: dict) -> None:
    entry["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry["estimate_usd"] = STEP_ESTIMATES[step]
    manifest["stages"][step] = entry
    run_round.save_manifest(manifest, round_number)


def step_is_done(manifest: dict, step: str, options: argparse.Namespace) -> bool:
    entry = manifest["stages"].get(step)
    return bool(entry) and entry.get("exit_code") == 0 and step not in options.redo


# ---------------------------------------------------------------- the steps

def failures_directory(options: argparse.Namespace) -> Path:
    """The classified candidate-on-train run the failed tasks come from."""
    return ROOT / "runs" / (options.failures_run or f"r{options.round}-cand-train")


def step_candidate_train(options: argparse.Namespace, task_ids: list[str]) -> dict:
    """The refused candidate on the train tasks the teacher can solve (k=1), then its run dir to laptop and GCS."""
    source = candidate_name(options.from_round)

    def make_command(flavor: str) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / "gate_job.py"), "--round", str(options.round),
                "--current", "configs/systems/base.yaml", "--candidate", f"configs/systems/{source}.yaml",
                "--user-model", options.user_model, "--train-eval", "--trials", "1", "--task-ids", *task_ids,
                "--flavor", flavor, "--timeout", CANDIDATE_TRAIN_TIMEOUT, "--launch"]

    cost, job = 0.0, None
    if results_are_on_the_hub(options.round, source):
        print("the candidate's train run is already on the Hub: not launching it again")
    else:
        result = run_hf_job(make_command, GATE_FLAVORS)
        cost, job = result["cost"], result["job"]
        if result["stage"] != "COMPLETED":
            return {"exit_code": 1, "actual_usd": cost, "note": f"the HF job ended as {result['stage']}", "tried": result["tried"]}

    from huggingface_hub import snapshot_download

    local_name = f"r{options.round}-cand-train"
    downloaded = Path(snapshot_download(
        RUNS_REPO, repo_type="dataset", allow_patterns=[f"gate/round-{options.round}/{source}-train/{source}/*"]))
    run_directory = ROOT / "runs" / local_name
    copy_run_directory(downloaded / "gate" / f"round-{options.round}" / f"{source}-train" / source, run_directory)
    gcs_runs.push_run(options.owner, run_directory)
    summary = run_round.read_summary(run_directory)
    return {"exit_code": 0, "actual_usd": cost, "job": job, "run_dir": f"runs/{local_name}", "num_runs": summary.get("num_runs"),
            "num_passed": summary.get("num_passed"), "num_infra_errors": summary.get("num_infra_errors")}


def step_classify(options: argparse.Namespace) -> dict:
    options.failure_source = f"r{options.round}-cand-train"
    entry = run_round.run_cloud_stage("classify", options)
    entry["actual_usd"] = STEP_ESTIMATES["classify"]  # classify.py does not meter its calls: the estimate
    return entry


def corrections_directory(options: argparse.Namespace) -> Path:
    return ROOT / "runs" / f"r{options.round}-corrections"


def step_corrections(options: argparse.Namespace) -> dict:
    """The teacher's next action at the points where the candidate went wrong (scripts/build_corrections.py)."""
    command = [sys.executable, str(ROOT / "scripts" / "build_corrections.py"), "--failures", str(failures_directory(options)),
               "--teacher-model", options.teacher_model, "--max-cost-usd", str(CORRECTIONS_COST_CAP_USD),
               "--output-dir", str(corrections_directory(options))]
    exit_code = subprocess.run(command).returncode
    stats_path = corrections_directory(options) / "corrections_stats.json"
    statistics = json.loads(stats_path.read_text()) if exit_code == 0 and stats_path.exists() else {}
    if exit_code == 0 and not statistics.get("rows"):
        return {"exit_code": 1, "actual_usd": statistics.get("cost_usd", 0.0), "note": "the teacher gave no usable correction"}
    return {"exit_code": exit_code, "actual_usd": statistics.get("cost_usd", 0.0), "rows": statistics.get("rows"),
            "wrong_turns_found": statistics.get("wrong_turns_found"), "rejected_by_the_guards": statistics.get("rejected_by_the_guards")}


def build_general_replay(per_category: int) -> Path:
    """The general function-calling examples (scripts/build_bfcl_replay.py), built once per size."""
    output_directory = ROOT / "runs" / f"bfcl-replay-{per_category}"
    if not (output_directory / "replay.jsonl").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_bfcl_replay.py"), "--per-category", str(per_category),
                        "--output-dir", str(output_directory)], check=True)
    return output_directory / "replay.jsonl"


def step_data(options: argparse.Namespace, chosen: recipe_module.Recipe) -> dict:
    output_directory = ROOT / "runs" / run_round.sft_dir_name(options.round)
    command = [sys.executable, str(ROOT / "scripts" / "build_sft.py"), "--output-dir", str(output_directory),
               "--only-model", options.teacher_model, "--push", "--revision", f"round-{options.round}",
               "--boost-from", str(failures_directory(options)), "--boost", str(chosen.boost)]
    if chosen.general_replay:
        command += ["--general-replay", str(build_general_replay(chosen.general_replay))]
    if chosen.corrections:
        command += ["--corrections", str(corrections_directory(options) / "corrections.jsonl")]
    for name in options.pool:
        command += ["--teacher", str(ROOT / "runs" / run_round.local_run_name(name))]
    exit_code = subprocess.run(command).returncode
    statistics = json.loads((output_directory / "sft_stats.json").read_text()) if exit_code == 0 else {}
    return {"exit_code": exit_code, "actual_usd": 0.0, "run_dir": f"runs/{output_directory.name}",
            "examples": statistics.get("examples"), "examples_before_boost": statistics.get("examples_before_boost"),
            "boosted_tasks": len(statistics.get("boosted_tasks") or []), "boost": statistics.get("boost"),
            "general_replay": statistics.get("general_replay"), "corrections": statistics.get("corrections")}


def train_command(options: argparse.Namespace, chosen: recipe_module.Recipe, flavor: str = TRAIN_FLAVOR) -> list[str]:
    """train_job.py's own options first, then `--` and the options it hands to train_lora.py."""
    revision = f"round-{options.round}"
    return [sys.executable, str(ROOT / "scripts" / "train_job.py"), "--dataset-revision", revision,
            "--revision", revision, "--flavor", flavor, "--timeout", TRAIN_TIMEOUT, "--launch",
            "--", "--epochs", f"{chosen.epochs:g}", "--lr", f"{chosen.learning_rate:g}"]


def step_train(options: argparse.Namespace, chosen: recipe_module.Recipe) -> dict:
    """AgentTune's RetrainingTrigger, then the LoRA job. A trigger that says no stops the loop here."""
    status = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "round_status.py"), str(failures_directory(options)),
         "--sft", str(ROOT / "runs" / run_round.sft_dir_name(options.round) / "sft.jsonl")],
        capture_output=True, text=True)
    status_path = failures_directory(options) / "round_status.json"
    if status.returncode != 0 or not status_path.exists():
        return {"exit_code": 1, "actual_usd": 0.0, "note": "the retraining trigger wrote no round_status.json"}
    trigger = json.loads(status_path.read_text())
    summary = {key: trigger.get(key) for key in ("should_train", "reason", "num_weight_fix_failures", "num_with_correction")}
    if not trigger.get("should_train"):
        return {"exit_code": 1, "actual_usd": 0.0, "retrain_trigger": summary, "note": "AgentTune says do not train"}

    result = run_hf_job(lambda flavor: train_command(options, chosen, flavor), TRAIN_FLAVORS)
    return {"exit_code": 0 if result["stage"] == "COMPLETED" else 1, "actual_usd": result["cost"], "retrain_trigger": summary,
            "job": result["job"], "flavor": result["flavor"], "tried": result["tried"],
            "note": None if result["stage"] == "COMPLETED" else f"the HF job ended as {result['stage']}"}


def write_candidate_config(options: argparse.Namespace, chosen: recipe_module.Recipe | None = None,
                           retrain: bool = True) -> str:
    """configs/systems/rN-weight.yaml, committed because the gate job uploads `git archive HEAD`.

    A round that trained has its own adapter (revision round-N). A round whose only change is the harness (the tool
    guards) keeps the adapter of the round before and sets ``tool_guards``."""
    import yaml

    from shipwright.heal import systems

    source_path = ROOT / "configs" / "systems" / f"{candidate_name(options.from_round)}.yaml"
    system = yaml.safe_load(source_path.read_text())
    system["name"] = candidate_name(options.round)
    if retrain:
        system["adapter"]["revision"] = f"round-{options.round}"
    if chosen is not None:
        system["tool_guards"] = chosen.tool_guards
    systems.validate_system(system)  # a config the gate could not load must not be committed

    reasons = "; ".join(chosen.reasons) if chosen else ""
    header = (f"# Round {options.round} candidate, written by scripts/run_loop.py.\n"
              f"# Adapter: {'trained this round' if retrain else 'the same as the round before'}. Tool guards: "
              f"{'on' if system.get('tool_guards') else 'off'}.\n" + (f"# Why: {reasons}\n" if reasons else ""))
    path = ROOT / "configs" / "systems" / f"{candidate_name(options.round)}.yaml"
    path.write_text(header + yaml.safe_dump(system, sort_keys=False))
    relative_path = str(path.relative_to(ROOT))
    subprocess.run(["git", "-C", str(ROOT), "add", relative_path], check=True)
    unchanged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "--", relative_path]).returncode == 0
    if unchanged:  # a rerun of the gate step: the config is already committed
        return relative_path
    # Only this file, whatever else is staged: the loop must not sweep up work in progress.
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m",
                    f"loop: round {options.round} candidate config", "--", relative_path], check=True)
    # Pushed so the next round's cloud classify step finds HEAD built (run_round checks it).
    subprocess.run(["git", "-C", str(ROOT), "push", "-q"], check=False)
    return relative_path


def bfcl_screen_passes(candidate_points: float, base_points: float, maximum_drop: float) -> bool:
    """The gate's BFCL rule: the candidate may score at most ``maximum_drop`` points below the live system."""
    return candidate_points >= base_points - maximum_drop


def step_bfcl_screen(options: argparse.Namespace, chosen: recipe_module.Recipe, retrain: bool) -> dict:
    """The candidate's BFCL alone, on the gate's own 300 samples. Fails the step if the gate's BFCL check could not pass."""
    from huggingface_hub import hf_hub_download

    write_candidate_config(options, chosen, retrain)
    name = candidate_name(options.round)

    def make_command(flavor: str) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / "gate_job.py"), "--round", str(options.round),
                "--current", run_round.current_system_path(), "--candidate", f"configs/systems/{name}.yaml",
                "--user-model", run_round.GATE_USER_MODEL, "--bfcl-screen", "--flavor", flavor,
                "--timeout", BFCL_SCREEN_TIMEOUT, "--launch"]

    result = run_hf_job(make_command, [options.gate_flavor, *[f for f in GATE_FLAVORS if f != options.gate_flavor]])
    if result["stage"] != "COMPLETED":
        return {"exit_code": 1, "actual_usd": result["cost"], "job": result["job"], "tried": result["tried"],
                "note": f"the HF job ended as {result['stage']}"}
    result_path = hf_hub_download(RUNS_REPO, f"gate/round-{options.round}/{name}-bfcl/{name}/bfcl.json",
                                  repo_type="dataset", force_download=True)
    candidate_points = json.loads(Path(result_path).read_text())["accuracy_points"]

    previous = fetch_gate_decision(options.from_round)
    base_points = previous["metrics_current"]["bfcl"]["accuracy_points"]
    maximum_drop = previous["thresholds"]["bfcl_max_drop_points"]
    needed = base_points - maximum_drop
    entry = {"exit_code": 0, "actual_usd": result["cost"], "job": result["job"], "flavor": result["flavor"],
             "bfcl_points": candidate_points, "base_points": base_points, "needed_points": needed}
    if not bfcl_screen_passes(candidate_points, base_points, maximum_drop):
        entry.update(exit_code=1, note=f"BFCL screen failed: {candidate_points} points, the gate needs at least {needed:.2f} "
                                       f"(base {base_points}): the full gate is not run")
    return entry


def step_gate(options: argparse.Namespace, chosen: recipe_module.Recipe, retrain: bool) -> dict:
    write_candidate_config(options, chosen, retrain)

    def make_command(flavor: str) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / "gate_job.py"), "--round", str(options.round),
                "--current", run_round.current_system_path(),
                "--candidate", f"configs/systems/{candidate_name(options.round)}.yaml",
                "--user-model", run_round.GATE_USER_MODEL, "--flavor", flavor, "--timeout", GATE_TIMEOUT, "--launch"]

    result = run_hf_job(make_command, [options.gate_flavor, *[f for f in GATE_FLAVORS if f != options.gate_flavor]])
    if result["stage"] != "COMPLETED":
        return {"exit_code": 1, "actual_usd": result["cost"], "job": result["job"], "tried": result["tried"],
                "note": f"the HF job ended as {result['stage']}"}
    decision = fetch_gate_decision(options.round)
    return {"exit_code": 0, "actual_usd": result["cost"], "job": result["job"], "flavor": result["flavor"],
            "decision": decision["decision"], "checks": {check["name"]: check["passed"] for check in decision["checks"]},
            "pass_1": {"current": decision["metrics_current"]["pass_1"], "candidate": decision["metrics_candidate"]["pass_1"]},
            "pass_k": {"current": decision["metrics_current"]["pass_k"], "candidate": decision["metrics_candidate"]["pass_k"]},
            "tool_call_error_rate": {"current": decision["metrics_current"]["tool_call_error_rate"],
                                     "candidate": decision["metrics_candidate"]["tool_call_error_rate"]}}


# ---------------------------------------------------------------- main

def print_plan(options: argparse.Namespace, previous_decision: dict, chosen: recipe_module.Recipe,
               previous: recipe_module.Recipe, task_ids: list[str]) -> None:
    print(f"Round {options.round}, started from the gate decision of round {options.from_round}: "
          f"{previous_decision['decision'].upper()}")
    failed = [check["name"] for check in previous_decision["checks"] if not check["passed"]]
    print(f"  failed checks: {failed}")
    print(f"  recipe: epochs {chosen.epochs:g}, learning rate {chosen.learning_rate:g}, boost x{chosen.boost}, "
          f"general examples {chosen.general_replay} per category, tool guards {'on' if chosen.tool_guards else 'off'}, "
          f"corrections {'yes' if chosen.corrections else 'no'}")
    for reason in chosen.reasons:
        print(f"    - {reason}")
    steps = planned_steps(options, chosen, previous)
    if steps == ["gate"]:
        print("  the same adapter is gated again, with the harness change: no training")
    elif options.failures_run:
        print(f"  failed tasks taken from {options.failures_run} (candidate_train and classify are skipped)")
    else:
        print(f"  the candidate runs on {len(task_ids)} train tasks the teacher can solve")
    total = 0.0
    for step in steps:
        total += STEP_ESTIMATES[step]
        print(f"  {step:16} est ${STEP_ESTIMATES[step]:.2f}")
    print(f"  total est ${total:.2f} of the ${options.budget_usd:.2f} budget")


def run_one_round(options: argparse.Namespace, spent_in_earlier_rounds: float,
                  last_round_cost: float | None = None) -> tuple[str, float, dict]:
    """One turn of the loop. Returns (outcome, what it spent, the round's summary for the report).

    Outcomes: "promote", "reject", "stopped" (a step failed: the same command resumes), "budget" (the money cannot
    cover the round), "exhausted" (every remedy is already in use), "planned" (no --launch)."""
    previous_decision = fetch_gate_decision(options.from_round)
    manifest = load_loop_manifest(options)
    previous = previous_recipe(options.from_round)
    chosen = next_recipe_for(manifest, options, previous_decision)
    retrain = recipe_module.needs_training(chosen, previous)
    task_ids = solvable_task_ids(options.pool)
    print_plan(options, previous_decision, chosen, previous, task_ids)
    summary = {"round": options.round, "from_round": options.from_round, "recipe": recipe_module.recipe_to_dict(chosen),
               "steps": planned_steps(options, chosen, previous)}
    if not options.launch:
        return "planned", 0.0, summary
    if chosen.exhausted:
        print(f"\nNo further remedy to try after round {options.from_round}: every one is already in use.")
        return "exhausted", 0.0, summary
    needed = cost_of_a_full_round(last_round_cost, summary["steps"])
    if needed + spent_in_earlier_rounds > options.budget_usd:
        print(f"\nStopping before round {options.round}: it needs about ${needed:.2f} and "
              f"${options.budget_usd - spent_in_earlier_rounds:.2f} is left of ${options.budget_usd:.2f}.")
        return "budget", 0.0, summary

    steps = {
        "candidate_train": lambda: step_candidate_train(options, task_ids),
        "classify": lambda: step_classify(options),
        "corrections": lambda: step_corrections(options),
        "data": lambda: step_data(options, chosen),
        "train": lambda: step_train(options, chosen),
        "bfcl_screen": lambda: step_bfcl_screen(options, chosen, retrain),
        "gate": lambda: step_gate(options, chosen, retrain),
    }
    if options.failures_run:
        manifest["failures_run"] = options.failures_run
    for step in summary["steps"]:
        if step_is_done(manifest, step, options):
            print(f"{step}: already done")
            continue
        check_fits_in_budget(manifest, step, options.budget_usd, spent_in_earlier_rounds)
        print(f"\n== round {options.round}: {step}")
        entry = steps[step]()
        record(manifest, options.round, step, entry)
        print(json.dumps({step: {key: value for key, value in entry.items() if key not in ("checks", "tried")}}, indent=2))
        if entry["exit_code"] != 0:
            print(f"stopped at {step}: {entry.get('note')}. The same command resumes from here.")
            summary["stopped_at"], summary["note"] = step, entry.get("note")
            return "stopped", spent_so_far(manifest), summary

    gate_entry = manifest["stages"]["gate"]
    summary.update(decision=gate_entry["decision"], checks=gate_entry["checks"], spent_usd=round(spent_so_far(manifest), 4))
    print(f"\nGATE round {options.round}: {gate_entry['decision'].upper()} (this round ${spent_so_far(manifest):.2f})")
    if gate_entry["decision"] == "promote":
        decision_path = ROOT / "runs" / f"gate-r{options.round}" / "gate_decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(fetch_gate_decision(options.round)))
        subprocess.run([sys.executable, str(ROOT / "scripts" / "promote.py"), "promote", str(decision_path)], check=True)
        return "promote", spent_so_far(manifest), summary
    return "reject", spent_so_far(manifest), summary


def environment() -> dict:
    """What a rerun needs to reproduce this run: the code, the libraries, the frozen configs, the seeds."""
    import hashlib
    from importlib import metadata

    def sha(path: str) -> str:
        return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()[:12]

    versions = {}
    for library in ("agenttune", "auditkit", "litellm", "huggingface_hub"):
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            versions[library] = "not installed"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"git_sha": head, "libraries": versions, "gate_config_sha": sha("configs/gate.yaml"),
            "splits_sha": sha("configs/splits.yaml"), "seed": 300,
            "training_pins": "transformers==5.22.2 trl==1.7.1 peft==0.21.2 (header of scripts/train_lora.py)"}


def write_report(options: argparse.Namespace, rounds: list[dict], spent: float, outcome: str) -> Path:
    report = {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "outcome": outcome,
              "budget_usd": options.budget_usd, "spent_usd": round(spent, 4), "rounds": rounds, "environment": environment(),
              "command": " ".join(sys.argv)}
    path = ROOT / "runs" / f"loop-report-round-{options.round}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def main() -> int:
    options = parse_options()
    if fetch_gate_decision(options.from_round)["decision"] == "promote":
        print(f"round {options.from_round} was promoted: nothing to heal. Run scripts/promote.py with its decision.")
        return 0

    spent, last_round_cost, rounds, outcome = 0.0, None, [], "no promotion"
    for round_index in range(options.max_rounds):
        round_options = argparse.Namespace(**vars(options))
        round_options.round = options.round + round_index
        round_options.from_round = options.from_round + round_index
        if round_index > 0:
            round_options.failures_run = None  # a later round takes the failures of the adapter before it
        outcome, round_cost, summary = run_one_round(round_options, spent, last_round_cost)
        rounds.append(summary)
        spent += round_cost
        last_round_cost = round_cost or last_round_cost
        if outcome == "planned":
            return 0
        if outcome in ("promote", "stopped", "budget", "exhausted"):
            break
    if options.launch:
        print(f"\nOutcome: {outcome}. Spent ${spent:.2f} of ${options.budget_usd:.2f}. "
              f"Report: {write_report(options, rounds, spent, outcome)}")
    return 1 if outcome == "stopped" else 0


def next_recipe_for(manifest: dict, options: argparse.Namespace, previous_decision: dict) -> recipe_module.Recipe:
    """The recipe, kept in the manifest once chosen so a rerun uses the same one."""
    if manifest.get("recipe"):
        return recipe_module.recipe_from_dict(manifest["recipe"])
    chosen = recipe_module.next_recipe(previous_decision, previous_recipe(options.from_round))
    manifest["recipe"] = recipe_module.recipe_to_dict(chosen)
    return chosen


if __name__ == "__main__":
    sys.exit(log.run_script(main, step="run_loop", meta=True))

"""A whole round (or several) as one chain: the decisions only, no I/O of its own.

``scripts/run_round.py all`` gives this module one function per stage (the real scripts and HF jobs, or fakes in a
dry run and in the tests) and the round config (``configs/rounds/<name>.yaml``). The chain decides:

- which stages a round runs (round 1 of a chain: all of them; a later round: what the recipe of the last gate
  refusal needs, see ``planned_stages``),
- whether a stage may start (its spend cap plus everything spent so far must fit in the total cap),
- whether a finished stage is good enough to build on (exit code, infra error share, SFT example count, the
  retraining trigger),
- whether to try a failed stage once more (into the same output dir, so finished runs are kept), and
- the round's final status.

A stage function returns a dict (the manifest entry): ``exit_code``, ``actual_usd`` and whatever it measured. It may
set ``no_retry`` when another try cannot help (too few examples, AgentTune says do not train).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from shipwright.heal import recipe as recipe_module

STAGES = ["baseline", "classify", "teacher", "data", "train", "gate"]
# What a later round of the same chain runs when the recipe trains again: the base system is unchanged after a
# refusal, so the first round's baseline, failures and teacher passes are still the right input.
RETRAIN_STAGES = ["data", "train", "gate"]

# Round statuses.
PROMOTED = "promoted"          # the gate approved the candidate; scripts/promote.py recorded it
REJECTED = "rejected"          # the gate refused it; next_round.json says why and what to change
STOPPED = "stopped"            # a stage failed twice, or its output was not good enough to build on
OVER_BUDGET = "over_budget"    # a stage's cap would pass the total cap: nothing more was started
EXHAUSTED = "exhausted"        # the recipe has no further remedy
HANDOFF = "handoff"            # the recipe needs the teacher's corrections: scripts/run_loop.py does that step


# ---------------------------------------------------------------- the round config

REQUIRED_KEYS = {
    "round": int, "owner": str, "seed": int, "models": dict, "baseline": dict, "teacher": dict, "data": dict,
    "train": dict, "gate": dict, "stop": dict, "spend": dict, "frozen": dict,
}
# GPU price per hour (HF Jobs, AGENTS.md), to check that price x timeout fits in the stage's cap.
GPU_PRICES = {"t4-small": 0.40, "l4x1": 0.80, "a10g-small": 1.00, "a10g-large": 1.50, "l40sx1": 1.80,
              "a100-large": 2.50}


def hours(timeout_text: str) -> float:
    """"45m", "1h", "90m" -> hours."""
    if timeout_text.endswith("h"):
        return float(timeout_text[:-1])
    if timeout_text.endswith("m"):
        return float(timeout_text[:-1]) / 60
    raise ValueError(f"timeout {timeout_text!r}: write it as 45m or 1h")


def worst_case_gpu_usd(job_settings: dict) -> float:
    """The most one launch can cost: the dearest flavor it may fall back to, for its whole timeout."""
    dearest = max(GPU_PRICES[flavor] for flavor in job_settings["flavors"])
    return dearest * hours(job_settings["timeout"]) + job_settings.get("api_usd", 0.0)


def validate_config(config: dict) -> list[str]:
    """Problems that would make the chain spend money on a run that cannot work. An empty list means it is fine."""
    problems = []
    for key, expected_type in REQUIRED_KEYS.items():
        if not isinstance(config.get(key), expected_type):
            problems.append(f"{key}: missing or not a {expected_type.__name__}")
    if problems:
        return problems
    caps = config["spend"]
    for stage in STAGES:
        if not isinstance(caps.get(stage), (int, float)):
            problems.append(f"spend.{stage}: missing")
    if not isinstance(caps.get("total"), (int, float)):
        problems.append("spend.total: missing")
    if problems:
        return problems
    for stage in ("train", "gate"):
        for flavor in config[stage].get("flavors", []):
            if flavor not in GPU_PRICES:
                problems.append(f"{stage}.flavors: no price for {flavor}")
        if not problems and worst_case_gpu_usd(config[stage]) > caps[stage]:
            problems.append(f"{stage}: worst case ${worst_case_gpu_usd(config[stage]):.2f} (price x timeout) passes its "
                            f"cap ${caps[stage]:.2f}")
    if config["gate"].get("timeout") and hours(config["gate"]["timeout"]) < 1:
        problems.append("gate.timeout: under 60m; a gate needs 45 to 50 minutes of wall time (AGENTS.md, gate 8)")
    return problems


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    problems = validate_config(config)
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return config


def run_name(config: dict, round_number: int, kind: str) -> str:
    """runs/ dir name of a round's stage output: r10-baseline, r10-teacher, r10-sft, r10-round (the manifest) and
    gate-r10. A dry run sets ``name_prefix`` so it never mixes with real rounds."""
    prefix = config.get("name_prefix", "")
    if kind == "gate":
        return f"{prefix}gate-r{round_number}"
    return f"{prefix}r{round_number}-{kind}"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- one stage

@dataclass
class StopRules:
    max_infra_error_rate: float
    min_sft_examples: int
    retries: int

    @classmethod
    def from_config(cls, config: dict) -> "StopRules":
        stop = config["stop"]
        return cls(stop["max_infra_error_rate"], stop["min_sft_examples"], stop["retries"])


def stage_problem(stage: str, entry: dict, rules: StopRules) -> str | None:
    """Why a finished stage is not good enough to build on, or None."""
    if entry.get("exit_code") != 0:
        return entry.get("note") or f"exit code {entry.get('exit_code')}"
    runs = entry.get("num_runs") or 0
    infra_errors = entry.get("num_infra_errors") or 0
    if runs and infra_errors / runs > rules.max_infra_error_rate:
        return f"{infra_errors} of {runs} runs were infra errors (over {rules.max_infra_error_rate:.0%})"
    if stage == "data" and (entry.get("examples") or 0) < rules.min_sft_examples:
        entry["no_retry"] = True  # the same inputs give the same data
        return f"the SFT data has {entry.get('examples') or 0} examples (under {rules.min_sft_examples})"
    return None


def stage_spend(entry: dict) -> float:
    return sum(attempt.get("actual_usd") or 0.0 for attempt in entry.get("attempts", [entry]))


def round_spend(manifest: dict) -> float:
    return sum(stage_spend(entry) for entry in manifest["stages"].values())


@dataclass
class StageOutcome:
    status: str          # "ok", STOPPED or OVER_BUDGET
    reason: str | None = None


def run_stage(stage: str, run: Callable[[], dict], manifest: dict, caps: dict, spent_before_round: float,
              rules: StopRules) -> StageOutcome:
    """Run one stage, try it once more if it failed and may be retried, and write its entry into the manifest."""
    attempts = []
    problem = None
    for attempt_number in range(1, rules.retries + 2):
        spent = spent_before_round + round_spend(manifest) + sum(a.get("actual_usd") or 0.0 for a in attempts)
        if spent + caps[stage] > caps["total"]:
            reason = (f"{stage}: its cap ${caps[stage]:.2f} on top of ${spent:.2f} spent passes the total cap "
                      f"${caps['total']:.2f}")
            if attempts:
                manifest["stages"][stage] = finished_entry(attempts, problem)
            return StageOutcome(OVER_BUDGET, reason)
        try:
            entry = run()
        except (Exception, SystemExit) as error:  # noqa: BLE001 - one stage's crash is a failed attempt, not a dead chain
            entry = {"exit_code": 1, "actual_usd": 0.0, "note": f"{type(error).__name__}: {error}"}
        entry["attempt"] = attempt_number
        attempts.append(entry)
        problem = stage_problem(stage, entry, rules)
        if problem is None or entry.get("no_retry"):
            break
    manifest["stages"][stage] = finished_entry(attempts, problem)
    if problem:
        return StageOutcome(STOPPED, f"{stage}: {problem}")
    return StageOutcome("ok")


def finished_entry(attempts: list[dict], problem: str | None) -> dict:
    """The last attempt's entry, with every attempt's spend kept (a retry is paid for too)."""
    entry = dict(attempts[-1])
    entry["ok"] = problem is None
    entry["problem"] = problem
    entry["actual_usd"] = round(sum(a.get("actual_usd") or 0.0 for a in attempts), 4)
    if len(attempts) > 1:
        entry["attempts"] = [{key: a.get(key) for key in ("attempt", "exit_code", "actual_usd", "note")} for a in attempts]
    return entry


# ---------------------------------------------------------------- one round, several rounds

def planned_stages(round_index: int, chosen: recipe_module.Recipe, previous: recipe_module.Recipe) -> tuple[list[str], str | None]:
    """(stages, status) for a round of the chain. A non-None status ends the chain before the round starts."""
    if round_index == 0:
        return list(STAGES), None
    if chosen.exhausted:
        return [], EXHAUSTED
    if chosen.corrections:
        return [], HANDOFF
    if not recipe_module.needs_training(chosen, previous):
        return ["gate"], None  # a harness change only (the tool guards): the same adapter is gated again
    return list(RETRAIN_STAGES), None


@dataclass
class RoundResult:
    round_number: int
    status: str
    reason: str | None
    spent_usd: float
    decision: dict | None = None


@dataclass
class ChainHooks:
    """What the chain calls outside itself. The script gives real ones; the tests and --dry-run give fakes."""
    stage_functions: Callable[[int, dict], dict[str, Callable[[], dict]]]  # (round number, chain state) -> stage -> run
    save_manifest: Callable[[int, dict], None]
    load_manifest: Callable[[int], dict | None]
    promote: Callable[[int, dict], None]
    write_next_round: Callable[[int, dict, recipe_module.Recipe], None]
    log: Callable[[str], None] = print


def run_round(round_number: int, stages: list[str], hooks: ChainHooks, state: dict, caps: dict, rules: StopRules,
              spent_before_round: float, resolved_config: dict) -> RoundResult:
    manifest = hooks.load_manifest(round_number) or {"round": round_number, "stages": {}}
    manifest.update(config=resolved_config, planned_stages=stages, recipe=recipe_module.recipe_to_dict(state["recipe"]))
    if state.get("failures_run"):
        manifest["failures_run"] = state["failures_run"]
    functions = hooks.stage_functions(round_number, state)
    for stage in stages:
        if (manifest["stages"].get(stage) or {}).get("ok"):
            hooks.log(f"round {round_number} {stage}: already done")
            continue
        hooks.log(f"round {round_number} {stage}: start")
        outcome = run_stage(stage, functions[stage], manifest, caps, spent_before_round, rules)
        manifest["spent_usd"] = round(round_spend(manifest), 4)
        hooks.save_manifest(round_number, manifest)
        if outcome.status != "ok":
            manifest.update(status=outcome.status, reason=outcome.reason)
            hooks.save_manifest(round_number, manifest)
            return RoundResult(round_number, outcome.status, outcome.reason, round_spend(manifest))
        hooks.log(f"round {round_number} {stage}: ok (${stage_spend(manifest['stages'][stage]):.2f})")

    decision = manifest["stages"]["gate"]["decision"]
    if decision["decision"] == "promote":
        try:
            hooks.promote(round_number, decision)
            status, reason = PROMOTED, "the gate approved the candidate"
        except (Exception, SystemExit) as error:  # noqa: BLE001 - recorded, so the approval is not lost
            status, reason = STOPPED, f"the gate approved, but the promotion failed: {type(error).__name__}: {error}"
    else:
        failed = sorted(recipe_module.failed_check_names(decision))
        status, reason = REJECTED, f"the gate refused the candidate: failed checks {failed}"
        hooks.write_next_round(round_number, decision, recipe_module.next_recipe(decision, state["recipe"]))
    manifest.update(status=status, reason=reason)
    hooks.save_manifest(round_number, manifest)
    return RoundResult(round_number, status, reason, round_spend(manifest), decision)


@dataclass
class ChainResult:
    status: str
    reason: str | None
    spent_usd: float
    rounds: list[RoundResult] = field(default_factory=list)


def run_chain(config: dict, rounds: int, hooks: ChainHooks, resolved_config: dict) -> ChainResult:
    """Round after round until one is promoted, a stage stops the chain, the money runs out or no remedy is left.

    After a refusal the deployed system stays the base of the next round, and the recipe of the refusal
    (``recipe.next_recipe``) says what that round changes."""
    caps, rules = config["spend"], StopRules.from_config(config)
    first_round = config["round"]
    # The first round's recipe: the config's ``recipe`` block (what earlier rounds already learned: the guards,
    # general function-calling examples), else a bare one.
    starting_recipe = config.get("recipe") or {}
    state = {"recipe": recipe_module.Recipe(epochs=config["train"]["epochs"],
                                            learning_rate=config["train"]["learning_rate"],
                                            boost=int(starting_recipe.get("boost", recipe_module.DEFAULT_BOOST)),
                                            general_replay=int(starting_recipe.get("general_replay", 0)),
                                            tool_guards=bool(starting_recipe.get("tool_guards", False))),
             "baseline_run": run_name(config, first_round, "baseline"),
             "teacher_run": run_name(config, first_round, "teacher"), "adapter_round": None}
    previous = state["recipe"]
    results: list[RoundResult] = []
    spent = 0.0
    for round_index in range(rounds):
        round_number = first_round + round_index
        if results:
            chosen = recipe_module.next_recipe(results[-1].decision, state["recipe"])
            previous, state["recipe"] = state["recipe"], chosen
            state["failures_run"] = state["baseline_run"]
        stages, early_status = planned_stages(round_index, state["recipe"], previous)
        if early_status:
            reason = "; ".join(state["recipe"].reasons) or early_status
            return ChainResult(early_status, reason, round(spent, 4), results)
        if "train" in stages:
            state["adapter_round"] = round_number  # this round trains its own adapter (revision round-N)
        result = run_round(round_number, stages, hooks, state, caps, rules, spent, resolved_config)
        results.append(result)
        spent += result.spent_usd
        if result.status != REJECTED:
            return ChainResult(result.status, result.reason, round(spent, 4), results)
    return ChainResult(REJECTED, f"{results[-1].reason} (no promotion in {rounds} round(s))", round(spent, 4), results)

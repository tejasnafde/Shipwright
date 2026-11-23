"""The self-healing loop's decisions, with no I/O of its own: which stages a round runs, whether a stage may start, whether
its output is good enough to build on, whether to retry it, and what the next round changes.

The caller (``shipwright.loop.service``) gives one function per stage (the real scripts and HF jobs, or fakes in a dry
run and in the tests) and the round config. A stage function returns a dict, the stage's manifest entry: ``exit_code``,
``actual_usd`` and whatever it measured. It may set ``no_retry`` when another try cannot help.

A round, by what the recipe changed since the last refused candidate (``recipe.next_recipe``):

    fresh round             baseline, classify, teacher, data, train, [bfcl_screen], gate
    harness change only     gate                                     (the same adapter, with the tool guards on)
    more general examples   data, train, [bfcl_screen], gate         (the first round's teacher passes are still right)
    corrections             candidate_train, classify, corrections, data, train, [bfcl_screen], gate
                            (the refused candidate's own failures on the train tasks: on-policy)
    capacity (short policy) [teacher], data, train, [bfcl_screen], gate
                            (teacher: a harvest of every train task under the short policy, if the config asks for it)

The chain goes round after round until a candidate is promoted, a stage fails twice, the money runs out or the recipe
has no further remedy. Nothing in it needs a person: every way it can end is one of the statuses below.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from shipwright.loop import recipe as recipe_module
from shipwright.loop.config import run_name

# Round and chain statuses.
PROMOTED = "promoted"        # the gate approved the candidate; the promotion is recorded
REJECTED = "rejected"        # the gate refused it; next_round.json says why and what the next round changes
STOPPED = "stopped"          # a stage failed twice, or its output was not good enough to build on
OVER_BUDGET = "over_budget"  # a stage's cap would pass the total cap: nothing more was started
EXHAUSTED = "exhausted"      # the recipe has no further remedy

FRESH_STAGES = ["baseline", "classify", "teacher", "data", "train", "bfcl_screen", "gate"]
RETRAIN_STAGES = ["data", "train", "bfcl_screen", "gate"]
CORRECTION_STAGES = ["candidate_train", "classify", "corrections", "data", "train", "bfcl_screen", "gate"]
CAPACITY_STAGES = ["teacher", "data", "train", "bfcl_screen", "gate"]


@dataclass
class StopRules:
    max_infra_error_rate: float
    min_sft_examples: int
    retries: int

    @classmethod
    def from_config(cls, config: dict) -> "StopRules":
        stop = config["stop"]
        return cls(stop["max_infra_error_rate"], stop["min_sft_examples"], stop["retries"])


@dataclass
class ChainState:
    """What one round hands the next."""
    recipe: recipe_module.Recipe
    baseline_run: str | None = None         # the first round's baseline (train tasks), when the chain started fresh
    teacher_run: str | None = None          # the first round's teacher passes
    failures_run: str | None = None         # the classified run the current round learns from
    adapter_round: int | None = None        # the round whose adapter the candidate serves (revision round-N)
    candidate_system: str | None = None     # the last candidate's system config: what candidate_train runs
    decisions: list[dict] = field(default_factory=list)  # every gate decision of this chain, the start's first


# ---------------------------------------------------------------- one stage

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
    return entry.get("actual_usd") or 0.0


def round_spend(manifest: dict) -> float:
    return sum(stage_spend(entry) for entry in manifest["stages"].values())


@dataclass
class StageOutcome:
    status: str            # "ok", STOPPED or OVER_BUDGET
    reason: str | None = None


def finished_entry(attempts: list[dict], problem: str | None) -> dict:
    """The last attempt's entry, with every attempt's spend kept (a retry is paid for too)."""
    entry = dict(attempts[-1])
    entry["ok"] = problem is None
    entry["problem"] = problem
    entry["actual_usd"] = round(sum(attempt.get("actual_usd") or 0.0 for attempt in attempts), 4)
    if len(attempts) > 1:
        entry["attempts"] = [{key: attempt.get(key) for key in ("attempt", "exit_code", "actual_usd", "note")}
                             for attempt in attempts]
    return entry


def run_stage(stage: str, run: Callable[[], dict], manifest: dict, caps: dict, spent_before_round: float,
              rules: StopRules) -> StageOutcome:
    """Run one stage, try it again while it may be retried, and write its entry into the manifest."""
    attempts: list[dict] = []
    problem = None
    for attempt_number in range(1, rules.retries + 2):
        spent = spent_before_round + round_spend(manifest) + sum(attempt.get("actual_usd") or 0.0 for attempt in attempts)
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


# ---------------------------------------------------------------- which stages a round runs

def planned_stages(fresh: bool, chosen: recipe_module.Recipe, previous: recipe_module.Recipe,
                   screen_bfcl: bool, capacity_harvest: bool = False) -> list[str]:
    """The stages of one round. ``fresh``: the chain's first round, with no refused candidate to start from.
    ``capacity_harvest``: the round that turns the short policy on also harvests teacher passes under it."""
    if fresh:
        stages = list(FRESH_STAGES)
    elif chosen.capacity and not previous.capacity:
        stages = list(CAPACITY_STAGES)
        if not capacity_harvest:
            stages.remove("teacher")
    elif chosen.corrections_rounds > previous.corrections_rounds:
        stages = list(CORRECTION_STAGES)
    elif recipe_module.needs_training(chosen, previous):
        stages = list(RETRAIN_STAGES)
    else:
        return ["gate"]  # a harness change only (the tool guards): the same adapter is gated again
    if not screen_bfcl:
        stages.remove("bfcl_screen")
    return stages


def update_state_for_round(state: ChainState, round_number: int, stages: list[str], config: dict) -> None:
    """Point the state at this round's own outputs before its stages run."""
    if "baseline" in stages:
        state.baseline_run = run_name(config, round_number, "baseline")
        state.teacher_run = run_name(config, round_number, "teacher")
        state.failures_run = state.baseline_run
    elif "teacher" in stages:  # a capacity round's harvest: its passes join the data from here on
        state.teacher_run = run_name(config, round_number, "teacher")
    if "candidate_train" in stages:
        state.failures_run = run_name(config, round_number, "cand-train")
    if "train" in stages:
        state.adapter_round = round_number


# ---------------------------------------------------------------- one round, several rounds

@dataclass
class RoundResult:
    round_number: int
    status: str
    reason: str | None
    spent_usd: float
    decision: dict | None = None


@dataclass
class ChainResult:
    status: str
    reason: str | None
    spent_usd: float
    rounds: list[RoundResult] = field(default_factory=list)


@dataclass
class Hooks:
    """What the chain calls outside itself. The service gives real ones; the tests and --dry-run give fakes."""
    stage_functions: Callable[[int, ChainState], dict[str, Callable[[], dict]]]
    save_manifest: Callable[[int, dict], None]
    load_manifest: Callable[[int], dict | None]
    promote: Callable[[int, dict], None]
    write_next_round: Callable[[int, dict, recipe_module.Recipe], None]
    log: Callable[[str], None] = print


def run_round(round_number: int, stages: list[str], hooks: Hooks, state: ChainState, config: dict,
              spent_before_round: float, resolved_config: dict) -> RoundResult:
    caps, rules = config["spend"], StopRules.from_config(config)
    manifest = hooks.load_manifest(round_number) or {"round": round_number, "stages": {}}
    manifest.update(config=resolved_config, planned_stages=stages, recipe=recipe_module.recipe_to_dict(state.recipe),
                    failures_run=state.failures_run, teacher_run=state.teacher_run, adapter_round=state.adapter_round)
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

    gate_entry = manifest["stages"]["gate"]
    decision = gate_entry["decision"]
    if gate_entry.get("candidate"):
        state.candidate_system = gate_entry["candidate"]  # also when the gate was done by an earlier execution
    if decision["decision"] == "promote":
        try:
            hooks.promote(round_number, decision)
            status, reason = PROMOTED, "the gate approved the candidate"
        except (Exception, SystemExit) as error:  # noqa: BLE001 - recorded, so the approval is not lost
            status, reason = STOPPED, f"the gate approved, but the promotion failed: {type(error).__name__}: {error}"
    else:
        failed = sorted(recipe_module.failed_check_names(decision))
        status, reason = REJECTED, f"the gate refused the candidate: failed checks {failed}"
        earlier = state.decisions[-1] if state.decisions else None
        hooks.write_next_round(round_number, decision, recipe_module.next_recipe(decision, state.recipe, earlier))
    manifest.update(status=status, reason=reason)
    hooks.save_manifest(round_number, manifest)
    return RoundResult(round_number, status, reason, round_spend(manifest), decision)


def starting_recipe(config: dict) -> recipe_module.Recipe:
    """The first round's recipe: the config's ``recipe`` block (what earlier rounds learned), else a bare one."""
    block = config.get("recipe") or {}
    return recipe_module.Recipe(epochs=config["train"]["epochs"], learning_rate=config["train"]["learning_rate"],
                                boost=int(block.get("boost", recipe_module.DEFAULT_BOOST)),
                                general_replay=int(block.get("general_replay", 0)),
                                tool_guards=bool(block.get("tool_guards", False)),
                                corrections_rounds=int(block.get("corrections_rounds", 0)))


def run_chain(config: dict, rounds: int, hooks: Hooks, resolved_config: dict,
              start_decision: dict | None = None, start_adapter_round: int | None = None) -> ChainResult:
    """Round after round until one is promoted, a stage stops the chain, the money runs out or no remedy is left.

    Config ``start``: continue from a refused candidate instead of a fresh round. ``start_decision`` is its gate
    decision, ``start_adapter_round`` the round whose adapter it serves. With ``start.regate`` the first round only gates
    that candidate again (its old decision was measured on an older harness or a broken gate) and the remedy is chosen
    from the new decision. With ``start.corrections`` the first round is a corrections round whatever the start decision
    says: used to keep improving an adapter the gate already promoted (a promotion alone changes no recipe). With
    ``start.retrain`` the first round retrains with the same recipe (data, train, gate): used when data.pool grew."""
    state = ChainState(recipe=starting_recipe(config))
    start = config.get("start") or {}
    regate = bool(start.get("regate"))
    another_corrections_round = bool(start.get("corrections"))
    retrain = bool(start.get("retrain"))
    if start:
        state.adapter_round = start_adapter_round
        state.candidate_system = start["candidate"]
        state.failures_run = start.get("failures_run")  # "owner/dir": the start candidate's classified train-task failures
    if start_decision is not None and not regate:
        state.decisions.append(start_decision)
    results: list[RoundResult] = []
    spent = 0.0
    for round_index in range(rounds):
        round_number = config["round"] + round_index
        fresh = round_index == 0 and not start
        previous = state.recipe
        if round_index == 0 and regate:
            pass  # re-measure the start candidate on today's harness and gate before anything is trained
        elif round_index == 0 and another_corrections_round:
            if previous.corrections_rounds >= recipe_module.MAX_CORRECTION_ROUNDS:
                return ChainResult(EXHAUSTED, f"start.corrections: {recipe_module.MAX_CORRECTION_ROUNDS} correction "
                                   f"rounds are the limit", 0.0, results)
            state.recipe = replace(previous, corrections=True, corrections_rounds=previous.corrections_rounds + 1,
                                   reasons=[f"corrections round {previous.corrections_rounds + 1}, asked for by "
                                            f"start.corrections: the start adapter's own failures become corrections"])
        elif round_index == 0 and retrain:
            state.recipe = replace(previous, reasons=["retrain asked for by start.retrain: the same recipe on a larger "
                                                      "data.pool (new teacher runs)"])
        elif not fresh:
            earlier = state.decisions[-2] if len(state.decisions) > 1 else None
            state.recipe = recipe_module.next_recipe(state.decisions[-1], previous, earlier)
            if state.recipe.exhausted:
                reason = "; ".join(state.recipe.reasons) or EXHAUSTED
                return ChainResult(EXHAUSTED, reason, round(spent, 4), results)
        if round_index == 0 and regate:
            stages = ["gate"]
        elif round_index == 0 and retrain:
            stages = [stage for stage in RETRAIN_STAGES
                      if stage != "bfcl_screen" or config["gate"].get("screen_bfcl")]
        else:
            stages = planned_stages(fresh, state.recipe, previous, bool(config["gate"].get("screen_bfcl")),
                                    bool(config["teacher"].get("capacity_harvest_trials")))
        update_state_for_round(state, round_number, stages, config)
        result = run_round(round_number, stages, hooks, state, config, spent, resolved_config)
        results.append(result)
        spent += result.spent_usd
        if result.status != REJECTED:
            return ChainResult(result.status, result.reason, round(spent, 4), results)
        state.decisions.append(result.decision)  # the gate stage already pointed state.candidate_system at its config
    return ChainResult(REJECTED, f"{results[-1].reason} (no promotion in {rounds} round(s))", round(spent, 4), results)

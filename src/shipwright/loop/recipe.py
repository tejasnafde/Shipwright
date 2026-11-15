"""What to change in the next round, from why the gate refused the last candidate.

The loop must learn from a rejection, not retrain the same thing. Each rule below reads one finding in the gate's
metrics and changes one part of the recipe, and records why (``Recipe.reasons`` goes into the round manifest). The rules
are small on purpose: a judge can read them, and each can be argued with. A rule stays only while the evidence for it
does; two rules from round 3 were tested in round 4 and dropped:

  BFCL dropped            -> mix general function-calling examples into the data (``general_replay``, rows per
                             category). The adapter did not forget how to call functions: it learned to answer in words
                             in every task (BFCL simple 67 to 4, irrelevance 4 to 90).
                             Dropped in round 4: fewer epochs and half the learning rate. BFCL went 35.0 to 33.7.
  looping, over-acting,   -> tool guards on the candidate (``tool_guards``): the harness rejects a repeated lookup, an
  context overflow rose     exchange to another product type and a second change to one order, and answers in words
                             instead. Cheapest remedy: no training, only a gate run. Added in round 5, after the data
                             showed the adapter repeating one lookup up to 8 times and exchanging a bookshelf for a camera.
                             Dropped in round 4: repeating the examples of the tasks the candidate failed (boost).
                             Loops went 45% to 40%: not enough to count.
  guards already on and   -> capacity (``capacity``): the candidate serves the short policy (about 950 tokens more room on
  context overflow rose      every turn) and the data is rebuilt with it, plus a teacher harvest of every train task
  by more than 0.05          under the same prompt when the config allows it, then train. Added 2026-11-15: gates 11 to 13
                             failed only non_target, on context overflow (5 to 15 runs against 0), and a correction round
                             made it worse (5 to 15): no guard calls a long chain of lookups wrong, so corrections have
                             nothing to correct there.
  guards already on and   -> correction examples: the teacher's next action at the points where the candidate went
  still failing              wrong (``corrections``), trained on with the replay data. Costs a training run.
  corrections already in  -> another round of corrections, from the failures of the adapter that round trained (on-policy:
  use and the checks         the corrections of one round are the training data of the next), while the gate shows
  still improve              progress, at most MAX_CORRECTION_ROUNDS times in all. When a round shows none, the loop stops.
  too few valid runs      -> no recipe change (a serving or infrastructure problem, not the weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_EPOCHS = 3.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BOOST = 1
MAX_CORRECTION_ROUNDS = 3  # the loop's own limit on correction rounds, so spend is bounded without anyone watching
REPLAY_STEP = 100  # general examples per BFCL category added each time the BFCL check fails
SCENE_LABELS = ("loop", "context_overflow", "tool_error_ignored")  # a candidate that gets stuck in a conversation


@dataclass
class Recipe:
    epochs: float = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    boost: int = DEFAULT_BOOST  # copies of each failed task's teacher examples in the training data
    general_replay: int = 0  # general function-calling examples per BFCL category in the training data
    tool_guards: bool = False  # the candidate system runs with the tool guards (src/shipwright/llm/guards.py)
    corrections: bool = False  # the data holds the teacher's corrections of the candidate's wrong turns
    corrections_rounds: int = 0  # how many rounds of corrections the adapter has been trained on
    capacity: bool = False  # the short policy, in serving and in the training data (prompted_tools.CONDENSED_POLICY)
    exhausted: bool = False  # every remedy is already in use and the checks still fail: the loop stops
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """``corrections`` and ``corrections_rounds`` say the same thing: keep them in step (older code sets only one)."""
        if self.corrections and not self.corrections_rounds:
            self.corrections_rounds = 1
        if self.corrections_rounds and not self.corrections:
            self.corrections = True


def recipe_to_dict(recipe: Recipe) -> dict:
    return {"epochs": recipe.epochs, "learning_rate": recipe.learning_rate, "boost": recipe.boost,
            "general_replay": recipe.general_replay, "tool_guards": recipe.tool_guards,
            "corrections": recipe.corrections, "corrections_rounds": recipe.corrections_rounds,
            "capacity": recipe.capacity, "exhausted": recipe.exhausted, "reasons": list(recipe.reasons)}


def recipe_from_dict(saved: dict) -> Recipe:
    """A recipe from a manifest (older manifests lack the newer fields)."""
    return Recipe(
        epochs=saved.get("epochs", DEFAULT_EPOCHS), learning_rate=saved.get("learning_rate", DEFAULT_LEARNING_RATE),
        boost=saved.get("boost", DEFAULT_BOOST), general_replay=saved.get("general_replay", 0),
        tool_guards=saved.get("tool_guards", False), corrections=saved.get("corrections", False),
        corrections_rounds=saved.get("corrections_rounds", 1 if saved.get("corrections") else 0),
        capacity=saved.get("capacity", False), exhausted=saved.get("exhausted", False), reasons=list(saved.get("reasons", [])))


def failed_check_names(gate_decision: dict) -> set[str]:
    return {check["name"] for check in gate_decision.get("checks", []) if not check["passed"]}


def needs_training(recipe: Recipe, previous: Recipe) -> bool:
    """False when the only change is the harness (the guards): the same adapter is gated again."""
    return (recipe.general_replay != previous.general_replay
            or recipe.corrections_rounds != previous.corrections_rounds
            or recipe.capacity != previous.capacity)


def progress(gate_decision: dict) -> tuple[int, float]:
    """How good a gate decision is: (checks passed, the candidate's pass^1). Compared, not thresholded."""
    checks = gate_decision.get("checks", [])
    passed = sum(1 for check in checks if check["passed"])
    return passed, float((gate_decision.get("metrics_candidate") or {}).get("pass_1") or 0.0)


def next_recipe(gate_decision: dict, previous: Recipe, earlier_decision: dict | None = None) -> Recipe:
    """The recipe for the round after a gate decision. A promotion changes nothing.

    ``earlier_decision`` is the gate decision before this one (None when there is none): another round of corrections is
    only started if the checks passed or the pass^1 rose since then."""
    recipe = Recipe(previous.epochs, previous.learning_rate, previous.boost, previous.general_replay,
                    previous.tool_guards, previous.corrections, previous.corrections_rounds, previous.capacity, False, [])
    failed = failed_check_names(gate_decision)
    if not failed:
        return recipe

    if "bfcl" in failed:
        recipe.general_replay = previous.general_replay + REPLAY_STEP
        recipe.reasons.append(
            f"bfcl check failed: general function-calling examples per category {previous.general_replay} to "
            f"{recipe.general_replay} (the adapter answers in words where a function should be called)"
        )

    current_rates = (gate_decision.get("metrics_current") or {}).get("label_rates") or {}
    candidate_rates = (gate_decision.get("metrics_candidate") or {}).get("label_rates") or {}
    rose = [
        label for label in SCENE_LABELS
        if candidate_rates.get(label, 0.0) - current_rates.get(label, 0.0) > 0.05
    ]
    stuck = bool(rose) or bool(failed & {"non_target", "pass_k", "pass_1", "target"})
    if stuck and not previous.tool_guards:
        recipe.tool_guards = True
        recipe.reasons.append(
            f"{', '.join(rose) or 'non-target or pass^k check'} failed: turn the tool guards on (a repeated lookup, a "
            f"change to another product type and a second change to one order are rejected). No training needed"
        )
    elif "context_overflow" in rose and previous.tool_guards and not previous.capacity:
        recipe.capacity = True
        recipe.reasons.append(
            f"context_overflow rose ({current_rates.get('context_overflow', 0.0):.2f} to "
            f"{candidate_rates.get('context_overflow', 0.0):.2f}) with the guards on: serve the short policy (about 950 "
            f"tokens more room every turn), rebuild the data with it and train")
    elif stuck and previous.tool_guards and previous.corrections_rounds == 0:
        recipe.corrections = True
        recipe.corrections_rounds = 1
        recipe.reasons.append(
            f"{', '.join(rose) or 'non-target or pass^k check'} still failing with the guards on: add the teacher's "
            f"corrections of the candidate's wrong turns to the data and train"
        )
    elif stuck and previous.corrections_rounds >= MAX_CORRECTION_ROUNDS:
        recipe.exhausted = True
        recipe.reasons.append(f"{MAX_CORRECTION_ROUNDS} rounds of corrections are done and the checks still fail: stop")
    elif stuck and earlier_decision is not None and progress(gate_decision) <= progress(earlier_decision):
        recipe.exhausted = True
        recipe.reasons.append(
            f"the last round of corrections did not improve the gate (checks passed and pass^1 {progress(earlier_decision)} "
            f"then {progress(gate_decision)}): stop")
    elif stuck:
        recipe.corrections_rounds = previous.corrections_rounds + 1
        recipe.reasons.append(
            f"corrections round {recipe.corrections_rounds}: the checks still fail and the gate improved "
            f"({progress(earlier_decision) if earlier_decision else 'first comparison'} to {progress(gate_decision)}); "
            f"the adapter's own failures become the next corrections")

    if not recipe.reasons:
        recipe.reasons.append(f"failed checks {sorted(failed)} have no recipe rule: same recipe, new data")
    return recipe

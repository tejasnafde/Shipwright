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
  guards already on and   -> correction examples: the teacher's next action at the points where the candidate went
  still failing              wrong (``corrections``), trained on with the replay data. Costs a training run.
  too few valid runs      -> no recipe change (a serving or infrastructure problem, not the weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_EPOCHS = 3.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BOOST = 1
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
    exhausted: bool = False  # every remedy is already in use and the checks still fail: the loop stops
    reasons: list[str] = field(default_factory=list)


def recipe_to_dict(recipe: Recipe) -> dict:
    return {"epochs": recipe.epochs, "learning_rate": recipe.learning_rate, "boost": recipe.boost,
            "general_replay": recipe.general_replay, "tool_guards": recipe.tool_guards,
            "corrections": recipe.corrections, "exhausted": recipe.exhausted, "reasons": list(recipe.reasons)}


def recipe_from_dict(saved: dict) -> Recipe:
    """A recipe from a manifest (older manifests lack the newer fields)."""
    return Recipe(
        epochs=saved.get("epochs", DEFAULT_EPOCHS), learning_rate=saved.get("learning_rate", DEFAULT_LEARNING_RATE),
        boost=saved.get("boost", DEFAULT_BOOST), general_replay=saved.get("general_replay", 0),
        tool_guards=saved.get("tool_guards", False), corrections=saved.get("corrections", False),
        exhausted=saved.get("exhausted", False), reasons=list(saved.get("reasons", [])))


def failed_check_names(gate_decision: dict) -> set[str]:
    return {check["name"] for check in gate_decision.get("checks", []) if not check["passed"]}


def needs_training(recipe: Recipe, previous: Recipe) -> bool:
    """False when the only change is the harness (the guards): the same adapter is gated again."""
    return recipe.general_replay != previous.general_replay or recipe.corrections != previous.corrections


def next_recipe(gate_decision: dict, previous: Recipe) -> Recipe:
    """The recipe for the round after a gate decision. A promotion changes nothing."""
    recipe = Recipe(previous.epochs, previous.learning_rate, previous.boost, previous.general_replay,
                    previous.tool_guards, previous.corrections, False, [])
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
    stuck = bool(rose) or "non_target" in failed or "pass_k" in failed
    if stuck and not previous.tool_guards:
        recipe.tool_guards = True
        recipe.reasons.append(
            f"{', '.join(rose) or 'non-target or pass^k check'} failed: turn the tool guards on (a repeated lookup, a "
            f"change to another product type and a second change to one order are rejected). No training needed"
        )
    elif stuck and previous.tool_guards and not previous.corrections:
        recipe.corrections = True
        recipe.reasons.append(
            f"{', '.join(rose) or 'non-target or pass^k check'} still failing with the guards on: add the teacher's "
            f"corrections of the candidate's wrong turns to the data and train"
        )
    elif stuck:
        recipe.exhausted = True
        recipe.reasons.append("guards and corrections are both in use and the checks still fail: no further remedy to try")

    if not recipe.reasons:
        recipe.reasons.append(f"failed checks {sorted(failed)} have no recipe rule: same recipe, new data")
    return recipe

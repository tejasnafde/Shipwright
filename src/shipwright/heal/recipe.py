"""What to change in the next round, from why the gate refused the last candidate.

The loop must learn from a rejection, not retrain the same thing. Each rule below reads one finding in the gate's
metrics and changes one part of the recipe, and records why (``Recipe.reasons`` goes into the round manifest). The rules
are small on purpose: a judge can read them, and each can be argued with.

  BFCL dropped            -> fewer epochs and half the learning rate: less drift away from the base model
                             (round 3: BFCL 43.33 to 35.0 after 3 epochs at 2e-4 on 185 examples)
  looping or context      -> repeat the teacher examples of the tasks the candidate failed (boost): its own
  overflow rose              failures become a larger share of what it learns from
  too few valid runs      -> no recipe change (a serving or infrastructure problem, not the weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_EPOCHS = 3.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_BOOST = 1
BOOST_STEP = 2
SCENE_LABELS = ("loop", "context_overflow", "tool_error_ignored")  # a candidate that gets stuck in a conversation


@dataclass
class Recipe:
    epochs: float = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    boost: int = DEFAULT_BOOST  # copies of each failed task's teacher examples in the training data
    reasons: list[str] = field(default_factory=list)


def failed_check_names(gate_decision: dict) -> set[str]:
    return {check["name"] for check in gate_decision.get("checks", []) if not check["passed"]}


def next_recipe(gate_decision: dict, previous: Recipe) -> Recipe:
    """The recipe for the round after a gate decision. A promotion changes nothing."""
    recipe = Recipe(previous.epochs, previous.learning_rate, previous.boost, [])
    failed = failed_check_names(gate_decision)
    if not failed:
        return recipe

    if "bfcl" in failed:
        recipe.epochs = max(1.0, previous.epochs - 1)
        recipe.learning_rate = previous.learning_rate / 2
        recipe.reasons.append(
            f"bfcl check failed: epochs {previous.epochs:g} to {recipe.epochs:g}, "
            f"learning rate {previous.learning_rate:g} to {recipe.learning_rate:g} (less drift from the base model)"
        )

    current_rates = (gate_decision.get("metrics_current") or {}).get("label_rates") or {}
    candidate_rates = (gate_decision.get("metrics_candidate") or {}).get("label_rates") or {}
    rose = [
        label for label in SCENE_LABELS
        if candidate_rates.get(label, 0.0) - current_rates.get(label, 0.0) > 0.05
    ]
    if "non_target" in failed or "pass_k" in failed or rose:
        recipe.boost = previous.boost + BOOST_STEP
        recipe.reasons.append(
            f"{', '.join(rose) or 'non-target or pass^k check'} got worse: the examples of the tasks the candidate "
            f"failed repeat x{previous.boost} to x{recipe.boost}"
        )

    if not recipe.reasons:
        recipe.reasons.append(f"failed checks {sorted(failed)} have no recipe rule: same recipe, new data")
    return recipe

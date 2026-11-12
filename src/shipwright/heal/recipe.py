"""What to change in the next round, from why the gate refused the last candidate.

The loop must learn from a rejection, not retrain the same thing. Each rule below reads one finding in the gate's
metrics and changes one part of the recipe, and records why (``Recipe.reasons`` goes into the round manifest). The rules
are small on purpose: a judge can read them, and each can be argued with. A rule stays only while the evidence for it
does; two rules from round 3 were tested in round 4 and dropped:

  BFCL dropped            -> mix general function-calling examples into the data (``general_replay``, rows per
                             category). The adapter did not forget how to call functions: it learned to answer in words
                             in every task (BFCL simple 67 to 4, irrelevance 4 to 90).
                             Dropped in round 4: fewer epochs and half the learning rate. BFCL went 35.0 to 33.7.
  looping or context      -> no recipe change, said so in the reasons. Dropped in round 4: repeating the examples of the
  overflow rose              tasks the candidate failed (boost). Loops went 45% to 40%: not enough to count.
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
    reasons: list[str] = field(default_factory=list)


def failed_check_names(gate_decision: dict) -> set[str]:
    return {check["name"] for check in gate_decision.get("checks", []) if not check["passed"]}


def next_recipe(gate_decision: dict, previous: Recipe) -> Recipe:
    """The recipe for the round after a gate decision. A promotion changes nothing."""
    recipe = Recipe(previous.epochs, previous.learning_rate, previous.boost, previous.general_replay, [])
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
    if rose:
        recipe.reasons.append(
            f"{', '.join(rose)} rose: no recipe change. Repeating the failed tasks' examples did not reduce looping "
            f"in round 4 (45% to 40%), so it needs a different fix (a guard on repeated calls, or new examples)"
        )

    if not recipe.reasons:
        recipe.reasons.append(f"failed checks {sorted(failed)} have no recipe rule: same recipe, new data")
    return recipe

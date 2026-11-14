from shipwright.heal.recipe import Recipe, needs_training, next_recipe


def decision(failed: list[str], current: dict | None = None, candidate: dict | None = None, pass_1: float = 0.1) -> dict:
    names = ["comparable", "valid_runs", "pass_1", "pass_k", "target", "non_target", "bfcl", "agenttune_gate"]
    return {
        "checks": [{"name": name, "passed": name not in failed} for name in names],
        "metrics_current": {"label_rates": current or {}},
        "metrics_candidate": {"label_rates": candidate or {}, "pass_1": pass_1},
    }


def test_a_promotion_changes_nothing():
    recipe = next_recipe(decision([]), Recipe())
    assert (recipe.epochs, recipe.learning_rate, recipe.boost, recipe.general_replay) == (3.0, 2e-4, 1, 0)


def test_a_bfcl_failure_adds_general_examples_and_leaves_the_training_intensity_alone():
    """Round 4 showed that fewer epochs and a lower rate did not help BFCL: they are no longer touched."""
    recipe = next_recipe(decision(["bfcl"]), Recipe(epochs=2.0, learning_rate=1e-4, boost=3))
    assert recipe.general_replay == 100
    assert (recipe.epochs, recipe.learning_rate, recipe.boost) == (2.0, 1e-4, 3)
    assert any("bfcl" in reason for reason in recipe.reasons)
    assert next_recipe(decision(["bfcl"]), recipe).general_replay == 200  # asked again if it still fails


def test_looping_turns_the_guards_on_and_needs_no_training():
    round_5 = decision(["pass_k", "non_target"], current={"loop": 0.17}, candidate={"loop": 0.60, "context_overflow": 0.37})
    previous = Recipe(boost=3, general_replay=100)
    recipe = next_recipe(round_5, previous)
    assert recipe.tool_guards and not recipe.corrections and recipe.boost == 3
    assert not needs_training(recipe, previous)  # only the harness changed: the same adapter is gated again
    assert any("guards" in reason for reason in recipe.reasons)


def test_still_failing_with_the_guards_on_adds_corrections_and_trains():
    round_6 = decision(["pass_k", "non_target"], current={"loop": 0.17}, candidate={"loop": 0.30})
    previous = Recipe(boost=3, general_replay=100, tool_guards=True)
    recipe = next_recipe(round_6, previous)
    assert recipe.corrections and recipe.tool_guards and needs_training(recipe, previous)
    assert recipe.corrections_rounds == 1


def test_a_serving_failure_is_not_a_recipe_problem():
    recipe = next_recipe(decision(["valid_runs"]), Recipe(epochs=2.0, learning_rate=1e-4, boost=3, general_replay=100))
    assert (recipe.epochs, recipe.learning_rate, recipe.boost, recipe.general_replay) == (2.0, 1e-4, 3, 100)
    assert "no recipe rule" in recipe.reasons[0]


def test_corrections_repeat_while_the_gate_keeps_improving():
    earlier = decision(["pass_k", "non_target", "bfcl"], pass_1=0.10)
    later = decision(["non_target"], pass_1=0.23)
    previous = Recipe(tool_guards=True, corrections_rounds=1)
    recipe = next_recipe(later, previous, earlier)
    assert recipe.corrections_rounds == 2 and not recipe.exhausted and needs_training(recipe, previous)
    assert "corrections round 2" in recipe.reasons[0]


def test_the_loop_stops_when_a_round_of_corrections_did_not_improve_the_gate():
    earlier = decision(["non_target"], pass_1=0.23)
    later = decision(["non_target"], pass_1=0.20)  # same checks passed, pass^1 not higher
    recipe = next_recipe(later, Recipe(tool_guards=True, corrections_rounds=1), earlier)
    assert recipe.exhausted and "did not improve" in recipe.reasons[0]


def test_the_loop_stops_after_the_most_rounds_of_corrections_even_if_it_improves():
    from shipwright.heal.recipe import MAX_CORRECTION_ROUNDS

    recipe = next_recipe(decision(["non_target"], pass_1=0.5), Recipe(tool_guards=True, corrections_rounds=MAX_CORRECTION_ROUNDS),
                         decision(["non_target", "pass_k"], pass_1=0.1))
    assert recipe.exhausted and "stop" in recipe.reasons[0]


def test_an_old_manifest_with_only_the_corrections_flag_counts_one_round():
    from shipwright.heal.recipe import recipe_from_dict

    assert recipe_from_dict({"corrections": True}).corrections_rounds == 1
    assert recipe_from_dict({}).corrections_rounds == 0

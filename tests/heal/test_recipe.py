from shipwright.heal.recipe import Recipe, next_recipe


def decision(failed: list[str], current: dict | None = None, candidate: dict | None = None) -> dict:
    names = ["comparable", "valid_runs", "pass_1", "pass_k", "target", "non_target", "bfcl", "agenttune_gate"]
    return {
        "checks": [{"name": name, "passed": name not in failed} for name in names],
        "metrics_current": {"label_rates": current or {}},
        "metrics_candidate": {"label_rates": candidate or {}},
    }


def test_a_promotion_changes_nothing():
    recipe = next_recipe(decision([]), Recipe())
    assert (recipe.epochs, recipe.learning_rate, recipe.boost) == (3.0, 2e-4, 1)


def test_the_round_3_rejection_gives_fewer_epochs_a_lower_rate_and_a_boost():
    round_3 = decision(
        ["pass_k", "non_target", "bfcl"],
        current={"loop": 0.05, "context_overflow": 0.05},
        candidate={"loop": 0.45, "context_overflow": 0.30},
    )
    recipe = next_recipe(round_3, Recipe())
    assert recipe.epochs == 2.0 and recipe.learning_rate == 1e-4 and recipe.boost == 3
    assert any("bfcl" in reason for reason in recipe.reasons)
    assert any("loop" in reason and "context_overflow" in reason for reason in recipe.reasons)


def test_a_serving_failure_is_not_a_recipe_problem():
    recipe = next_recipe(decision(["valid_runs"]), Recipe(epochs=2.0, learning_rate=1e-4, boost=3))
    assert (recipe.epochs, recipe.learning_rate, recipe.boost) == (2.0, 1e-4, 3)
    assert "no recipe rule" in recipe.reasons[0]


def test_epochs_never_go_below_one():
    assert next_recipe(decision(["bfcl"]), Recipe(epochs=1.0)).epochs == 1.0

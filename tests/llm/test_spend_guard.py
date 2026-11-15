import pytest

from shipwright.llm.costs import BudgetExceeded, SpendGuard


def test_guard_allows_calls_below_the_limit():
    spent = {"usd": 0.5}
    guard = SpendGuard(1.0, lambda: spent["usd"])
    guard()
    assert not guard.hit


def test_guard_raises_at_the_limit_and_stays_tripped():
    spent = {"usd": 0.5}
    guard = SpendGuard(1.0, lambda: spent["usd"])
    spent["usd"] = 1.0
    with pytest.raises(BudgetExceeded, match="max-cost-usd"):
        guard()
    spent["usd"] = 0.0  # a lower reading later does not reopen the budget
    with pytest.raises(BudgetExceeded):
        guard()
    assert guard.hit

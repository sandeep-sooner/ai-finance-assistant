import pytest
from pydantic import ValidationError

from src.core.schemas import GoalRequest
from src.goals.planner import future_value, parse_goal, plan_goal, required_monthly


def test_future_value_matches_closed_form():
    # 1,000 for 12 months at 0% is just the sum; at 12%/yr it compounds.
    assert future_value(1000, 100, 0.0, 12) == pytest.approx(2200)
    assert future_value(1000, 0, 0.12, 12) == pytest.approx(1120)


def test_required_monthly_reaches_target_exactly():
    need = required_monthly(50_000, 5_000, 0.06, 120)
    assert future_value(5_000, need, 0.06, 120) == pytest.approx(50_000)
    assert required_monthly(1_000, 5_000, 0.06, 120) == 0  # already there


def test_plan_shortfall_and_inflation():
    p = plan_goal(GoalRequest(target_amount=50_000, years=10, current_savings=5_000, monthly_contribution=300))
    assert p.target_future == pytest.approx(50_000 * 1.03**10, rel=1e-6)
    assert not p.on_track and p.gap < 0
    assert {f.code for f in p.flags} == {"shortfall"}
    assert len(p.projection) == 11 and p.projection[0].balance == 5_000
    assert p.projection[-1].balance == pytest.approx(p.projected_value)
    assert p.total_contributed == 5_000 + 300 * 120


def test_plan_on_track_and_scenarios_are_ordered():
    p = plan_goal(GoalRequest(target_amount=10_000, years=10, monthly_contribution=500))
    assert p.on_track and p.gap > 0
    values = [s.projected_value for s in p.scenarios]
    assert values == sorted(values)  # conservative < balanced < growth


def test_return_override_and_flags():
    p = plan_goal(GoalRequest(target_amount=20_000, years=2, risk_profile="growth", annual_return=0.15))
    assert p.annual_return == 0.15
    assert {"short_horizon_growth", "optimistic_return", "no_contributions"} <= {f.code for f in p.flags}


def test_request_validation():
    with pytest.raises(ValidationError):
        GoalRequest(target_amount=0, years=10)
    with pytest.raises(ValidationError):
        GoalRequest(target_amount=1000, years=0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want to save $50,000 in 10 years. I have $5,000 and can put $300 a month", (50_000, 10, 5_000, 300, "balanced")),
        ("retire with 2 million in 25 years, growth, I already saved 100k", (2_000_000, 25, 100_000, 0, "growth")),
        ("house down payment of 60k in 2 yrs", (60_000, 2, 0, 0, "balanced")),
    ],
)
def test_parse_goal(text, expected):
    g = parse_goal(text)
    assert (g.target_amount, g.years, g.current_savings, g.monthly_contribution, g.risk_profile) == expected


def test_parse_goal_needs_amount_and_years():
    assert parse_goal("Plan my savings for a 10-year goal") is None
    assert parse_goal("I want to save $50,000") is None

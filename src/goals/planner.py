"""Savings-goal projections. Pure math with explicit assumptions; the agent only explains the result.

Model: monthly compounding at the annual return, contributions at the end of each
month, and a target that grows with inflation so it keeps today's buying power.
"""
from __future__ import annotations

import re

from src.core.schemas import GoalPlan, GoalRequest, GoalScenario, PortfolioFlag, ProjectionPoint, RiskProfile

# Long-run nominal annual return assumptions. Illustrative, not forecasts.
RETURN_ASSUMPTIONS: dict[str, float] = {"conservative": 0.04, "balanced": 0.06, "growth": 0.08}
PROFILE_MIX = {
    "conservative": "mostly bonds and cash",
    "balanced": "a mix of stocks and bonds",
    "growth": "mostly stocks",
}
ASSUMPTIONS_NOTE = (
    "Assumes a steady {ret:.1%} yearly return and {inf:.1%} inflation, compounded monthly. Real returns vary "
    "from year to year and can be negative; these are planning estimates, not forecasts."
)


def _monthly_rate(annual: float) -> float:
    return (1 + annual) ** (1 / 12) - 1


def future_value(present: float, monthly: float, annual_return: float, months: int) -> float:
    r = _monthly_rate(annual_return)
    if r == 0:
        return present + monthly * months
    growth = (1 + r) ** months
    return present * growth + monthly * (growth - 1) / r


def required_monthly(target: float, present: float, annual_return: float, months: int) -> float:
    """Monthly saving needed to reach target from present. 0 if present already gets there."""
    if months <= 0:
        return 0.0
    r = _monthly_rate(annual_return)
    if r == 0:
        return max(0.0, (target - present) / months)
    growth = (1 + r) ** months
    return max(0.0, (target - present * growth) * r / (growth - 1))


def plan_goal(req: GoalRequest) -> GoalPlan:
    annual = req.annual_return if req.annual_return is not None else RETURN_ASSUMPTIONS[req.risk_profile]
    months = req.years * 12
    target_future = req.target_amount * (1 + req.inflation_rate) ** req.years
    projected = future_value(req.current_savings, req.monthly_contribution, annual, months)
    need = required_monthly(target_future, req.current_savings, annual, months)
    contributed = req.current_savings + req.monthly_contribution * months

    projection = [
        ProjectionPoint(
            year=y,
            balance=round(future_value(req.current_savings, req.monthly_contribution, annual, y * 12), 2),
            contributed=round(req.current_savings + req.monthly_contribution * 12 * y, 2),
        )
        for y in range(req.years + 1)
    ]

    scenarios = []
    for profile, r in RETURN_ASSUMPTIONS.items():
        fv = future_value(req.current_savings, req.monthly_contribution, r, months)
        scenarios.append(GoalScenario(
            risk_profile=profile,  # type: ignore[arg-type]
            annual_return=r,
            projected_value=round(fv, 2),
            required_monthly=round(required_monthly(target_future, req.current_savings, r, months), 2),
            on_track=fv >= target_future,
        ))

    plan = GoalPlan(
        name=req.name.strip() or "My goal",
        years=req.years,
        annual_return=annual,
        inflation_rate=req.inflation_rate,
        target_today=round(req.target_amount, 2),
        target_future=round(target_future, 2),
        projected_value=round(projected, 2),
        projected_value_today=round(projected / (1 + req.inflation_rate) ** req.years, 2),
        gap=round(projected - target_future, 2),
        progress=projected / target_future,
        on_track=projected >= target_future,
        required_monthly=round(need, 2),
        total_contributed=round(contributed, 2),
        growth=round(projected - contributed, 2),
        projection=projection,
        scenarios=scenarios,
        flags=[],
        assumptions_note=ASSUMPTIONS_NOTE.format(ret=annual, inf=req.inflation_rate),
    )
    plan.flags = _flags(req, plan)
    return plan


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _flags(req: GoalRequest, p: GoalPlan) -> list[PortfolioFlag]:
    flags: list[PortfolioFlag] = []
    if p.on_track:
        flags.append(PortfolioFlag(
            level="info", code="on_track",
            message=f"At {_money(req.monthly_contribution)} a month you are projected to reach "
                    f"{_money(p.projected_value)}, about {_money(p.gap)} above the target.",
        ))
    else:
        flags.append(PortfolioFlag(
            level="warning", code="shortfall",
            message=f"Projected shortfall of {_money(-p.gap)}. Saving about {_money(p.required_monthly)} a month "
                    f"would close it under these assumptions.",
        ))
    if req.years < 3 and req.risk_profile == "growth":
        flags.append(PortfolioFlag(
            level="warning", code="short_horizon_growth",
            message="With under 3 years to go, a stock-heavy mix can fall sharply right before you need "
                    "the money. Short-term goals are often kept in steadier options.",
        ))
    elif req.years < 3:
        flags.append(PortfolioFlag(
            level="info", code="short_horizon",
            message="This is a short-term goal, so the return assumption matters less than how much you save.",
        ))
    if req.annual_return is not None and req.annual_return > 0.10:
        flags.append(PortfolioFlag(
            level="warning", code="optimistic_return",
            message=f"A {req.annual_return:.0%} yearly return is optimistic to rely on for planning.",
        ))
    if req.monthly_contribution == 0 and not p.on_track:
        flags.append(PortfolioFlag(
            level="info", code="no_contributions",
            message="You have not set a monthly amount yet. Regular saving usually matters more than returns.",
        ))
    return flags


def to_prompt_context(p: GoalPlan, req: GoalRequest) -> str:
    lines = [
        f"Goal: {p.name}; target {_money(p.target_today)} in today's money, in {p.years} years.",
        f"Target in future dollars at {p.inflation_rate:.1%} inflation: {_money(p.target_future)}.",
        f"Starting savings {_money(req.current_savings)}; monthly saving {_money(req.monthly_contribution)}.",
        f"Assumed return {p.annual_return:.1%} a year ({req.risk_profile}: {PROFILE_MIX[req.risk_profile]}).",
        f"Projected value {_money(p.projected_value)} ({_money(p.projected_value_today)} in today's money); "
        f"{'on track' if p.on_track else 'shortfall'} by {_money(abs(p.gap))}.",
        f"Monthly saving needed to reach the target: {_money(p.required_monthly)}.",
        f"Total contributed {_money(p.total_contributed)}; investment growth {_money(p.growth)}.",
        "Scenarios: " + "; ".join(
            f"{s.risk_profile} {s.annual_return:.0%}: {_money(s.projected_value)}, needs {_money(s.required_monthly)}/mo"
            for s in p.scenarios
        ),
    ]
    if p.flags:
        lines.append("Flags: " + " | ".join(f"[{f.level}] {f.message}" for f in p.flags))
    lines.append(p.assumptions_note)
    return "\n".join(lines)


# ---------- Parsing a goal out of a chat message ----------

_AMOUNT = r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million|lakh|lakhs|crore|crores)?\b"
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7}
_YEARS = re.compile(r"(\d{1,2})[\s-]*(?:years?|yrs?)\b", re.I)
_MONTHLY = re.compile(_AMOUNT + r"\s*(?:/|per|a|each|every)\s*month", re.I)
_CURRENT = re.compile(r"(?:have|saved|already|currently|starting with|start with)\D{0,25}?" + _AMOUNT, re.I)
_ANY = re.compile(_AMOUNT, re.I)
_PROFILE = re.compile(r"\b(conservative|balanced|moderate|growth|aggressive)\b", re.I)


def _to_amount(num: str, unit: str | None) -> float:
    return float(num.replace(",", "")) * _MULT.get((unit or "").lower(), 1)


def parse_goal(text: str) -> GoalRequest | None:
    """Best-effort extraction of target, years, savings and monthly amount. None if target or years is missing."""
    years_m = _YEARS.search(text)
    if not years_m:
        return None
    used: list[tuple[int, int]] = [years_m.span()]

    def take(m: re.Match | None) -> float | None:
        if not m:
            return None
        used.append(m.span())
        return _to_amount(m.group(1), m.group(2))

    monthly = take(_MONTHLY.search(text))
    current = take(_CURRENT.search(text))
    target = None
    for m in _ANY.finditer(text):
        if any(a <= m.start() < b for a, b in used):
            continue
        value = _to_amount(m.group(1), m.group(2))
        if value >= 100:  # skip stray small numbers such as ages
            target = value
            break
    if target is None:
        return None

    profile: RiskProfile = "balanced"
    if pm := _PROFILE.search(text):
        word = pm.group(1).lower()
        profile = {"moderate": "balanced", "aggressive": "growth"}.get(word, word)  # type: ignore[assignment]
    return GoalRequest(
        name="Savings goal", target_amount=target, years=max(1, min(int(years_m.group(1)), 60)),
        current_savings=current or 0, monthly_contribution=monthly or 0, risk_profile=profile,
    )

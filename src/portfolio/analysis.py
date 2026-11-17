"""Portfolio math. No LLM here: the agent only explains these numbers.

analyze() is pure: give it holdings plus quotes/profiles and it returns metrics.
analyze_live() fetches those quotes/profiles from the MarketDataService first.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Mapping

from src.core.schemas import (
    AllocationSlice, Holding, HoldingResult, PortfolioAnalysis, PortfolioFlag, Profile, Quote,
)
from src.market.instruments import lookup, normalize_symbol
from src.market.service import MarketDataService

POSITION_LIMIT = 0.20   # one non-diversified holding above this is flagged
SECTOR_LIMIT = 0.35     # one sector above this is flagged
FEW_HOLDINGS = 5        # effective holdings below this (and no broad fund) is flagged
HIGH_FEE = 0.005        # a value-weighted fund cost above 0.5% a year is flagged

# Rough riskiness of each asset class (0 = cash-like, 1 = single stocks), for the simple risk score.
ASSET_RISK = {
    "Stock": 1.0, "Equity fund": 0.9, "Broad equity fund": 0.8, "International equity fund": 0.85,
    "Mutual fund": 0.8, "Bond fund": 0.3, "Commodity fund": 0.7, "Crypto": 1.5, "Cash": 0.0,
}
EQUITY_CLASSES = {"Stock", "Equity fund", "Broad equity fund", "International equity fund", "Mutual fund"}
RISK_LEVELS = ((35, "Low"), (60, "Moderate"), (85, "High"), (101, "Very high"))


# ---------- Parsing holdings out of a chat message ----------

_NUM = r"(\d+(?:\.\d+)?)"
_TICKER = r"([A-Z][A-Z0-9]{0,4}(?:\.[A-Z])?)"
_COST = r"(?:\s*(?:@|at)\s*\$?" + _NUM + r")?"
# "10 AAPL", "10 shares of AAPL", "10 AAPL @ 150", "10 AAPL at $150"
_QTY_FIRST = re.compile(r"(?<![\w.$])" + _NUM + r"\s*(?:shares?\s+(?:of\s+)?)?" + _TICKER + r"\b" + _COST)
# "AAPL: 10", "AAPL x10", "AAPL = 10 @ 150"
_TICKER_FIRST = re.compile(r"\b" + _TICKER + r"\s*[:=x×]\s*" + _NUM + r"\b" + _COST)
# Upper-case words that look like tickers but are not.
_NOT_TICKERS = {"ETF", "ETFS", "IRA", "USD", "INR", "EUR", "GBP", "PE", "EPS", "CEO", "AI", "I", "A", "K", "M", "B", "X"}


def parse_holdings(text: str) -> list[Holding]:
    """Pull holdings out of free text. Tickers must be upper case to avoid matching ordinary words."""
    found: list[Holding] = []
    for m in _QTY_FIRST.finditer(text):
        shares, symbol, cost = m.group(1), m.group(2), m.group(3)
        if symbol not in _NOT_TICKERS and float(shares) > 0:
            found.append(Holding(symbol=symbol, shares=float(shares), avg_cost=float(cost) if cost else None))
    if not found:
        for m in _TICKER_FIRST.finditer(text):
            symbol, shares, cost = m.group(1), m.group(2), m.group(3)
            if symbol not in _NOT_TICKERS and float(shares) > 0:
                found.append(Holding(symbol=symbol, shares=float(shares), avg_cost=float(cost) if cost else None))
    return found


# ---------- Pricing ----------

def merge_holdings(holdings: list[Holding]) -> list[Holding]:
    """Combine repeated symbols; average cost is share-weighted when every lot has one."""
    shares: dict[str, float] = defaultdict(float)
    cost_total: dict[str, float] = defaultdict(float)
    cost_known: dict[str, bool] = defaultdict(lambda: True)
    for h in holdings:
        s = normalize_symbol(h.symbol)
        shares[s] += h.shares
        if h.avg_cost is None:
            cost_known[s] = False
        else:
            cost_total[s] += h.avg_cost * h.shares
    return [
        Holding(symbol=s, shares=n, avg_cost=(cost_total[s] / n) if cost_known[s] and n else None)
        for s, n in shares.items()
    ]


def _price_source(q: Quote) -> str:
    if q.is_sample:
        return "sample"
    return "cache" if q.is_stale else "live"


def _default_profile(symbol: str) -> Profile:
    inst = lookup(symbol)
    if inst is None:
        return Profile(symbol=symbol)
    return Profile(symbol=symbol, name=inst.name, asset_class=inst.asset_class,
                   sector=inst.sector, diversified=inst.diversified)


def _price_note(sources: set[str]) -> str:
    parts = []
    if "live" in sources or "cache" in sources:
        parts.append("Prices come from Yahoo Finance and may be delayed by about 15 minutes.")
    if "cache" in sources:
        parts.append("Some prices are from an earlier fetch because the data source did not respond.")
    if "sample" in sources:
        parts.append("Prices marked 'sample' are illustrative, not real quotes.")
    if "cost" in sources:
        parts.append("Holdings without a price are valued at the average cost you entered.")
    return " ".join(parts) or "No prices were available."


# ---------- Analysis ----------

def _slices(values: dict[str, float], total: float) -> list[AllocationSlice]:
    rows = [AllocationSlice(name=k, value=round(v, 2), weight=(v / total) if total else 0.0) for k, v in values.items()]
    return sorted(rows, key=lambda r: r.value, reverse=True)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def analyze(
    holdings: list[Holding],
    quotes: Mapping[str, Quote] | None = None,
    profiles: Mapping[str, Profile] | None = None,
) -> PortfolioAnalysis:
    """Pure function. Missing quotes fall back to average cost; missing profiles to the static table."""
    merged = merge_holdings(holdings)
    quotes = quotes or {}
    profiles = profiles or {}

    rows: list[dict] = []
    for h in merged:
        prof = profiles.get(h.symbol) or _default_profile(h.symbol)
        q = quotes.get(h.symbol)
        if q is not None:
            price, source = q.price, _price_source(q)
        elif h.avg_cost is not None:
            price, source = h.avg_cost, "cost"
        else:
            price, source = None, "missing"
        value = h.shares * price if price is not None else 0.0
        cost_basis = h.shares * h.avg_cost if h.avg_cost is not None else None
        # A gain is only meaningful when the price is a market price, not the cost echoed back.
        gain = value - cost_basis if cost_basis is not None and q is not None else None
        rows.append(dict(h=h, prof=prof, price=price, source=source, value=value, cost_basis=cost_basis, gain=gain))

    total = sum(r["value"] for r in rows)
    results = [
        HoldingResult(
            symbol=r["h"].symbol,
            shares=r["h"].shares,
            price=r["price"],
            price_source=r["source"],
            value=round(r["value"], 2),
            weight=(r["value"] / total) if total else 0.0,
            avg_cost=r["h"].avg_cost,
            cost_basis=round(r["cost_basis"], 2) if r["cost_basis"] is not None else None,
            gain=round(r["gain"], 2) if r["gain"] is not None else None,
            gain_pct=(r["gain"] / r["cost_basis"]) if r["gain"] is not None and r["cost_basis"] else None,
            asset_class=r["prof"].asset_class,
            sector=r["prof"].sector,
            expense_ratio=getattr(lookup(r["h"].symbol), "expense_ratio", None),
        )
        for r in rows
    ]
    results.sort(key=lambda r: r.value, reverse=True)

    gain_rows = [r for r in rows if r["gain"] is not None]
    gain_cost = sum(r["cost_basis"] for r in gain_rows)
    total_gain = sum(r["gain"] for r in gain_rows) if gain_rows else None
    cost_rows = [r["cost_basis"] for r in rows if r["cost_basis"] is not None]

    weights = [r.weight for r in results]
    hhi = sum(w * w for w in weights)

    by_class: dict[str, float] = defaultdict(float)
    by_sector: dict[str, float] = defaultdict(float)
    for r in rows:
        by_class[r["prof"].asset_class] += r["value"]
        by_sector[r["prof"].sector] += r["value"]
    sector_slices = _slices(by_sector, total)
    diversified = {r["h"].symbol for r in rows if r["prof"].diversified}
    div_score = diversification_score(results, sector_slices, diversified)
    risk = risk_score(results, diversified)
    fund_cost = sum(r.value * r.expense_ratio for r in results if r.expense_ratio is not None)
    fund_value = sum(r.value for r in results if r.expense_ratio is not None)
    flags = _flags(results, sector_slices, hhi, diversified)
    if fund_value and fund_cost / fund_value > HIGH_FEE:
        flags.append(PortfolioFlag(
            level="info", code="high_fund_costs",
            message=f"The funds held cost about {fund_cost / fund_value * 100:.2f}% a year on average. "
                    f"Broad index funds often cost under 0.10%.",
        ))

    return PortfolioAnalysis(
        total_value=round(total, 2),
        total_cost=round(sum(cost_rows), 2) if cost_rows else None,
        total_gain=round(total_gain, 2) if total_gain is not None else None,
        total_gain_pct=(total_gain / gain_cost) if total_gain is not None and gain_cost else None,
        largest_symbol=results[0].symbol if total else None,
        largest_weight=weights[0] if total else 0.0,
        top3_weight=sum(weights[:3]),
        effective_holdings=round(1 / hhi, 1) if hhi else 0.0,
        holdings=results,
        by_asset_class=_slices(by_class, total),
        by_sector=sector_slices,
        flags=flags,
        price_note=_price_note({r["source"] for r in rows}),
        diversification_score=div_score,
        risk_score=risk,
        risk_level=next(label for limit, label in RISK_LEVELS if risk < limit) if total else "Unknown",
        equity_weight=sum(r.weight for r in results if r.asset_class in EQUITY_CLASSES),
        expense_ratio=round(fund_cost / fund_value, 6) if fund_value else None,
        annual_fund_cost=round(fund_cost, 2) if fund_value else None,
    )


def diversification_score(results: list[HoldingResult], sectors: list[AllocationSlice], diversified: set[str]) -> int:
    """0..100. Starts from 1 - sum(weight^2), counting a broad fund as (almost) no concentration because
    it is diversified inside, then subtracts a penalty when one sector dominates."""
    if not results or not any(r.weight for r in results):
        return 0
    concentration = sum((0.05 if r.symbol in diversified else 1.0) * r.weight ** 2 for r in results)
    score = (1 - concentration) * 100
    top_sector = max((s.weight for s in sectors if s.name not in {"Diversified", "Unknown", "Fixed income"}), default=0.0)
    if top_sector > SECTOR_LIMIT:
        score -= (top_sector - SECTOR_LIMIT) * 60
    return int(round(max(0.0, min(score, 100.0))))


def risk_score(results: list[HoldingResult], diversified: set[str]) -> int:
    """0..100 from the asset-class mix, nudged up when one single stock dominates."""
    if not results or not any(r.weight for r in results):
        return 0
    score = sum(r.weight * ASSET_RISK.get(r.asset_class, 0.9) for r in results) * 85
    top_single = max((r.weight for r in results if r.symbol not in diversified and r.asset_class == "Stock"), default=0.0)
    if top_single > 0.4:
        score += (top_single - 0.4) * 25
    return int(round(max(0.0, min(score, 100.0))))


async def analyze_live(holdings: list[Holding], market: MarketDataService) -> PortfolioAnalysis:
    """Fetch quotes and profiles concurrently, then analyze. Per-symbol failures degrade, never raise."""
    symbols = [h.symbol for h in merge_holdings(holdings)]
    (quote_list, _errors), profile_list = await asyncio.gather(
        market.quotes(symbols),
        asyncio.gather(*(market.profile(s) for s in symbols)),
    )
    return analyze(holdings, {q.symbol: q for q in quote_list}, {p.symbol: p for p in profile_list})


def _flags(
    results: list[HoldingResult], sectors: list[AllocationSlice], hhi: float, diversified: set[str]
) -> list[PortfolioFlag]:
    flags: list[PortfolioFlag] = []

    for r in results:
        if r.symbol not in diversified and r.weight > POSITION_LIMIT:
            flags.append(PortfolioFlag(
                level="warning", code="position_concentration",
                message=f"{r.symbol} is {_pct(r.weight)} of the portfolio. A single holding above "
                        f"{_pct(POSITION_LIMIT)} means its ups and downs drive your results.",
            ))
    for s in sectors:
        if s.name not in {"Diversified", "Unknown", "Fixed income"} and s.weight > SECTOR_LIMIT:
            flags.append(PortfolioFlag(
                level="warning", code="sector_concentration",
                message=f"{s.name} is {_pct(s.weight)} of the portfolio. Companies in one sector "
                        f"often fall together.",
            ))
    if hhi and 1 / hhi < FEW_HOLDINGS and not diversified:
        flags.append(PortfolioFlag(
            level="warning", code="few_holdings",
            message=f"The portfolio behaves like about {1 / hhi:.1f} equal positions and holds no broad "
                    f"fund, so it is lightly diversified.",
        ))
    if results and not any(r.sector == "Fixed income" for r in results):
        flags.append(PortfolioFlag(
            level="info", code="no_bonds",
            message="There are no bond holdings. Bonds usually move less than stocks; whether you "
                    "need them depends on your timeline and comfort with swings.",
        ))
    at_cost = [r.symbol for r in results if r.price_source == "cost"]
    if at_cost:
        flags.append(PortfolioFlag(
            level="info", code="priced_at_cost",
            message=f"No market price for {', '.join(at_cost)}, so they are valued at your average cost.",
        ))
    missing = [r.symbol for r in results if r.price_source == "missing"]
    if missing:
        flags.append(PortfolioFlag(
            level="warning", code="missing_price",
            message=f"No price for {', '.join(missing)}. Check the ticker, or add an average cost to include them.",
        ))
    return flags


def to_prompt_context(a: PortfolioAnalysis) -> str:
    """Compact, factual summary handed to the LLM. Data only, no instructions."""
    lines = [f"Total value: ${a.total_value:,.2f}"]
    if a.total_gain is not None and a.total_gain_pct is not None:
        lines.append(f"Unrealized gain/loss: ${a.total_gain:,.2f} ({a.total_gain_pct * 100:+.1f}%)")
    lines.append(f"Effective number of holdings: {a.effective_holdings}")
    lines.append(f"Diversification score: {a.diversification_score}/100 (100 = broadly diversified)")
    lines.append(f"Risk level: {a.risk_level} (score {a.risk_score}/100); stocks and stock funds = {_pct(a.equity_weight)}")
    if a.expense_ratio is not None:
        lines.append(f"Fund costs: {a.expense_ratio * 100:.2f}% a year on the funds held, about ${a.annual_fund_cost:,.2f} a year")
    if a.largest_symbol:
        lines.append(f"Largest position: {a.largest_symbol} at {_pct(a.largest_weight)}; top 3 = {_pct(a.top3_weight)}")
    lines.append("Holdings: " + "; ".join(
        f"{h.symbol} {_pct(h.weight)} ({h.asset_class}, {h.sector}, price {h.price_source})" for h in a.holdings
    ))
    lines.append("By asset class: " + ", ".join(f"{s.name} {_pct(s.weight)}" for s in a.by_asset_class))
    lines.append("By sector: " + ", ".join(f"{s.name} {_pct(s.weight)}" for s in a.by_sector))
    if a.flags:
        lines.append("Flags: " + " | ".join(f"[{f.level}] {f.message}" for f in a.flags))
    lines.append(f"Price note: {a.price_note}")
    return "\n".join(lines)

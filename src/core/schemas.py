"""Pydantic models shared by the API and the agents. Mirror these in the React types.ts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2000)
    level: Literal["beginner", "intermediate"] = "beginner"  # how much jargon the answer may assume


class Source(BaseModel):
    title: str
    url: str | None = None
    category: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    sources: list[Source]
    meta: dict = Field(default_factory=dict)  # model tier, cache hit, fallbacks, latency


# ---------- Portfolio ----------

class Holding(BaseModel):
    symbol: str = Field(min_length=1, max_length=10, pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")
    shares: float = Field(gt=0, le=1e9)
    avg_cost: float | None = Field(default=None, ge=0, le=1e7)


class PortfolioRequest(BaseModel):
    holdings: list[Holding] = Field(min_length=1, max_length=50)


class HoldingResult(BaseModel):
    symbol: str
    shares: float
    price: float | None
    price_source: str          # "sample" | "cost" | "missing"
    value: float
    weight: float              # 0..1 share of total value
    avg_cost: float | None
    cost_basis: float | None
    gain: float | None
    gain_pct: float | None
    asset_class: str
    sector: str
    expense_ratio: float | None = None  # fraction per year; funds only


class AllocationSlice(BaseModel):
    name: str
    value: float
    weight: float


class PortfolioFlag(BaseModel):
    level: str                 # "info" | "warning"
    code: str
    message: str


class PortfolioAnalysis(BaseModel):
    total_value: float
    total_cost: float | None
    total_gain: float | None
    total_gain_pct: float | None
    largest_symbol: str | None
    largest_weight: float
    top3_weight: float
    effective_holdings: float  # 1 / sum(weight^2): how many equal positions this behaves like
    holdings: list[HoldingResult]
    by_asset_class: list[AllocationSlice]
    by_sector: list[AllocationSlice]
    flags: list[PortfolioFlag]
    price_note: str
    diversification_score: int = 0          # 0 (one stock) .. 100 (broadly diversified)
    risk_score: int = 0                     # 0 .. 100, from the asset-class mix and concentration
    risk_level: str = "Unknown"             # "Low" | "Moderate" | "High" | "Very high"
    equity_weight: float = 0.0              # share of the portfolio in stocks and stock funds
    expense_ratio: float | None = None      # value-weighted yearly cost of the funds held
    annual_fund_cost: float | None = None   # dollars per year paid in fund expenses


# ---------- Market data ----------

class Quote(BaseModel):
    symbol: str
    name: str | None = None
    price: float
    previous_close: float | None = None
    change: float
    change_pct: float          # fraction, e.g. 0.012 for +1.2%
    currency: str = "USD"
    year_high: float | None = None
    year_low: float | None = None
    fetched_at: str            # ISO 8601, when the provider returned this price
    source: str                # "yfinance" | "cache" | "sample"
    is_stale: bool = False     # served from cache after a provider failure
    is_sample: bool = False    # illustrative price, not market data


class Profile(BaseModel):
    symbol: str
    name: str | None = None
    asset_class: str = "Unknown"
    sector: str = "Unknown"
    diversified: bool = False


class PricePoint(BaseModel):
    date: str                  # ISO date
    close: float


class PriceHistory(BaseModel):
    symbol: str
    period: str
    points: list[PricePoint]
    source: str
    is_stale: bool = False
    is_sample: bool = False


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    url: str | None = None
    published_at: str | None = None
    symbol: str | None = None


class QuoteError(BaseModel):
    symbol: str
    error_code: str
    message: str


class QuotesResponse(BaseModel):
    quotes: list[Quote]
    errors: list[QuoteError]


class SymbolDetail(BaseModel):
    quote: Quote
    profile: Profile


class NewsResponse(BaseModel):
    symbol: str | None
    items: list[NewsItem]


# ---------- Goal planning ----------

RiskProfile = Literal["conservative", "balanced", "growth"]


class GoalRequest(BaseModel):
    name: str = Field(default="My goal", max_length=60)
    target_amount: float = Field(gt=0, le=1e10)          # in today's money
    years: int = Field(ge=1, le=60)
    current_savings: float = Field(default=0, ge=0, le=1e10)
    monthly_contribution: float = Field(default=0, ge=0, le=1e8)
    risk_profile: RiskProfile = "balanced"
    inflation_rate: float = Field(default=0.03, ge=0, le=0.15)
    annual_return: float | None = Field(default=None, ge=0, le=0.20)  # overrides the profile's assumption


class ProjectionPoint(BaseModel):
    year: int
    balance: float
    contributed: float


class GoalScenario(BaseModel):
    risk_profile: RiskProfile
    annual_return: float
    projected_value: float
    required_monthly: float
    on_track: bool


class GoalPlan(BaseModel):
    name: str
    years: int
    annual_return: float
    inflation_rate: float
    target_today: float
    target_future: float        # target grown by inflation: what is needed in future dollars
    projected_value: float
    projected_value_today: float
    gap: float                  # projected - target_future (negative = shortfall)
    progress: float             # projected / target_future
    on_track: bool
    required_monthly: float     # monthly saving that reaches the target under these assumptions
    total_contributed: float
    growth: float               # projected - total contributed
    projection: list[ProjectionPoint]
    scenarios: list[GoalScenario]
    flags: list[PortfolioFlag]
    assumptions_note: str

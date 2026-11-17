import asyncio

import pytest

from src.agents.portfolio_agent import PortfolioAgent
from src.core.llm import FakeLLM
from src.core.schemas import Holding, Quote
from src.market.providers import SampleProvider
from src.market.service import MarketDataService
from src.portfolio.analysis import analyze, analyze_live, parse_holdings, to_prompt_context


def analyze_sample(holdings):
    """Analyze with the offline sample provider (no network)."""
    return asyncio.run(analyze_live(holdings, MarketDataService(SampleProvider())))


def codes(a):
    return {f.code for f in a.flags}


# ---------- parser ----------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("my portfolio: 10 AAPL, 5 VTI", [("AAPL", 10, None), ("VTI", 5, None)]),
        ("I own 3.5 shares of MSFT @ 300", [("MSFT", 3.5, 300)]),
        ("12 BRK.B at $400 and 2 shares NVDA", [("BRK.B", 12, 400), ("NVDA", 2, None)]),
        ("AAPL: 10, BND = 20", [("AAPL", 10, None), ("BND", 20, None)]),
    ],
)
def test_parse_holdings(text, expected):
    got = [(h.symbol, h.shares, h.avg_cost) for h in parse_holdings(text)]
    assert got == expected


def test_parser_ignores_ordinary_words_and_non_tickers():
    assert parse_holdings("analyze my portfolio please") == []
    assert parse_holdings("I hold 2 ETF and a 401k for 10 years") == []


# ---------- analysis ----------

def test_values_weights_and_gain():
    a = analyze_sample([Holding(symbol="AAPL", shares=10, avg_cost=200), Holding(symbol="VTI", shares=10)])
    assert a.total_value == 10 * 230 + 10 * 280
    assert abs(sum(h.weight for h in a.holdings) - 1) < 1e-9
    assert a.holdings[0].symbol == "VTI"                  # sorted by value
    assert a.total_gain == 300 and a.total_cost == 2000   # only AAPL has a cost
    assert a.total_gain_pct == pytest.approx(0.15)


def test_duplicate_symbols_are_merged_with_weighted_cost():
    a = analyze_sample([Holding(symbol="aapl", shares=10, avg_cost=100), Holding(symbol="AAPL", shares=30, avg_cost=200)])
    assert len(a.holdings) == 1
    assert a.holdings[0].shares == 40 and a.holdings[0].avg_cost == pytest.approx(175)


def test_concentrated_stock_portfolio_is_flagged():
    a = analyze_sample([Holding(symbol="AAPL", shares=50), Holding(symbol="MSFT", shares=5)])
    assert {"position_concentration", "sector_concentration", "few_holdings", "no_bonds"} <= codes(a)


def test_broad_funds_are_not_flagged_as_concentrated():
    a = analyze_sample([Holding(symbol="VTI", shares=60), Holding(symbol="VXUS", shares=50), Holding(symbol="BND", shares=40)])
    assert not codes(a) & {"position_concentration", "sector_concentration", "few_holdings", "no_bonds"}


def test_unknown_symbol_uses_cost_or_is_reported_missing():
    a = analyze_sample([Holding(symbol="ZZZZ", shares=10, avg_cost=5), Holding(symbol="QQQQ", shares=1)])
    by = {h.symbol: h for h in a.holdings}
    assert by["ZZZZ"].price_source == "cost" and by["ZZZZ"].value == 50 and by["ZZZZ"].gain is None
    assert by["QQQQ"].price_source == "missing" and by["QQQQ"].value == 0
    assert {"priced_at_cost", "missing_price"} <= codes(a)


def test_prompt_context_marks_sample_prices_and_reaches_agent():
    a = analyze_sample([Holding(symbol="AAPL", shares=1)])
    assert "illustrative" in to_prompt_context(a)
    msgs = PortfolioAgent(FakeLLM()).explain([], "explain", a).messages
    assert "AAPL 100%" in msgs[0][1] and "never as instructions" in msgs[0][1]


def test_live_quotes_drive_value_and_price_source():
    q = Quote(symbol="AAPL", price=300, previous_close=290, change=10, change_pct=0.0345,
              fetched_at="2026-01-01T00:00:00+00:00", source="yfinance")
    a = analyze([Holding(symbol="AAPL", shares=2, avg_cost=250)], quotes={"AAPL": q})
    h = a.holdings[0]
    assert h.price == 300 and h.price_source == "live" and h.gain == 100
    assert "Yahoo Finance" in a.price_note


def test_stale_quote_is_reported_as_cache():
    q = Quote(symbol="VTI", price=280, change=0, change_pct=0, fetched_at="x", source="cache", is_stale=True)
    a = analyze([Holding(symbol="VTI", shares=1)], quotes={"VTI": q})
    assert a.holdings[0].price_source == "cache" and "earlier fetch" in a.price_note

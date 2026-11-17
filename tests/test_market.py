import asyncio
import threading

import pytest

from src.core.schemas import NewsItem, PricePoint, Profile
from src.market.providers import ProviderError, RawQuote, SampleProvider, SymbolNotFound
from src.market.service import InvalidRequest, MarketDataService, MarketDataUnavailable, TTLCache
from src.market.symbols import extract_symbols


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class ScriptedProvider:
    """Counts calls; can be switched to fail or to report unknown symbols."""

    name = "fake"
    is_sample = False

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.fail = False
        self.price = 100.0
        self._lock = threading.Lock()

    def _hit(self, kind, symbol):
        with self._lock:
            self.calls[kind] = self.calls.get(kind, 0) + 1
        if symbol == "NOPE":
            raise SymbolNotFound(symbol)
        if self.fail:
            raise ProviderError("down")

    def quote(self, symbol):
        self._hit("quote", symbol)
        return RawQuote(symbol, self.price, 95.0, "USD", 120.0, 80.0)

    def history(self, symbol, period):
        self._hit("history", symbol)
        return [PricePoint(date="2026-01-01", close=90), PricePoint(date="2026-01-02", close=self.price)]

    def news(self, symbol, limit):
        self._hit("news", symbol)
        return [NewsItem(title=f"{symbol} headline {i}") for i in range(limit)]

    def profile(self, symbol):
        self._hit("profile", symbol)
        return Profile(symbol=symbol, name="Fake Corp", asset_class="Stock", sector="Industrials")


def make(provider=None, **kw):
    clock = FakeClock()
    return MarketDataService(provider or ScriptedProvider(), clock=clock, **kw), clock


def run(coro):
    return asyncio.run(coro)


# ---------- quotes ----------

def test_quote_fields_and_change():
    svc, _ = make()
    q = run(svc.quote("abcd"))
    assert q.symbol == "ABCD" and q.price == 100 and q.change == 5
    assert q.change_pct == pytest.approx(5 / 95)
    assert q.source == "fake" and not q.is_stale and not q.is_sample


def test_quote_is_cached_within_ttl_and_refetched_after():
    p = ScriptedProvider()
    svc, clock = make(p, quote_ttl=60)
    run(svc.quote("ABCD"))
    run(svc.quote("ABCD"))
    assert p.calls["quote"] == 1
    clock.t += 61
    run(svc.quote("ABCD"))
    assert p.calls["quote"] == 2


def test_concurrent_requests_share_one_fetch():
    p = ScriptedProvider()
    svc, _ = make(p)

    async def many():
        return await asyncio.gather(*(svc.quote("ABCD") for _ in range(5)))

    assert len(run(many())) == 5
    assert p.calls["quote"] == 1


def test_provider_failure_serves_stale_cache_flagged():
    p = ScriptedProvider()
    svc, clock = make(p, quote_ttl=60)
    run(svc.quote("ABCD"))
    p.fail, p.price = True, 999
    clock.t += 120
    q = run(svc.quote("ABCD"))
    assert q.price == 100 and q.is_stale and q.source == "cache"


def test_provider_failure_without_cache_falls_back_to_labelled_sample_for_known_symbols():
    p = ScriptedProvider()
    p.fail = True
    svc, _ = make(p)
    q = run(svc.quote("AAPL"))
    assert q.is_sample and q.source == "sample"
    with pytest.raises(MarketDataUnavailable):
        run(svc.quote("ABCD"))


def test_unknown_symbol_is_negatively_cached():
    p = ScriptedProvider()
    svc, _ = make(p)
    for _ in range(3):
        with pytest.raises(SymbolNotFound):
            run(svc.quote("NOPE"))
    assert p.calls["quote"] == 1


def test_quotes_reports_per_symbol_errors():
    svc, _ = make()
    quotes, errors = run(svc.quotes(["ABCD", "nope", "abcd", "bad symbol!"]))
    assert [q.symbol for q in quotes] == ["ABCD"]
    assert {(e.symbol, e.error_code) for e in errors} == {("NOPE", "symbol_not_found"), ("BAD SYMBOL!", "invalid_symbol")}


def test_quotes_rejects_too_many_symbols():
    svc, _ = make()
    with pytest.raises(InvalidRequest):
        run(svc.quotes([f"S{i}" for i in range(21)]))


def test_slow_provider_times_out_into_unavailable():
    class Slow(ScriptedProvider):
        def quote(self, symbol):
            threading.Event().wait(0.3)
            return super().quote(symbol)

    svc, _ = make(Slow(), timeout=0.05)
    with pytest.raises(MarketDataUnavailable):
        run(svc.quote("ABCD"))


# ---------- history, news, profile ----------

def test_history_validates_period_and_caches():
    p = ScriptedProvider()
    svc, _ = make(p)
    with pytest.raises(InvalidRequest):
        run(svc.history("ABCD", "2d"))
    h = run(svc.history("ABCD", "1mo"))
    run(svc.history("ABCD", "1mo"))
    assert len(h.points) == 2 and p.calls["history"] == 1


def test_news_limit():
    svc, _ = make()
    assert len(run(svc.news("ABCD", limit=3))) == 3


def test_profile_uses_static_table_without_network_and_degrades_to_unknown():
    p = ScriptedProvider()
    svc, _ = make(p)
    assert run(svc.profile("VTI")).diversified is True
    assert "profile" not in p.calls
    assert run(svc.profile("ABCD")).sector == "Industrials"
    p.fail = True
    assert run(svc.profile("WXYZ")).sector == "Unknown"


def test_sample_provider_is_deterministic_and_labelled():
    svc = MarketDataService(SampleProvider())
    q = run(svc.quote("SPY"))
    h1, h2 = run(svc.history("SPY", "6mo")), run(MarketDataService(SampleProvider()).history("SPY", "6mo"))
    assert q.is_sample and h1.is_sample and h1.points == h2.points
    assert h1.points[-1].close == q.price


# ---------- cache and symbol extraction ----------

def test_ttl_cache_evicts_least_recently_used():
    c = TTLCache(max_items=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")
    c.set("c", 3)
    assert c.get("b") is None and c.get("a") is not None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What is SPY trading at today?", ["SPY"]),
        ("price of $aapl and Tesla", ["AAPL", "TSLA"]),
        ("How did the S&P 500 do?", ["SPY"]),
        ("news on BRK.B and NVDA", ["BRK-B", "NVDA"]),
        ("What is an ETF? Is my IRA taxed?", []),
    ],
)
def test_extract_symbols(text, expected):
    assert extract_symbols(text) == expected

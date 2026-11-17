"""MarketDataService: the one place the app gets market data from.

Wraps a synchronous provider with:
  * a TTL cache per data kind, plus negative caching of unknown symbols;
  * request coalescing, so concurrent callers share one in-flight fetch;
  * a timeout and a concurrency cap on provider calls (run in worker threads);
  * retries with exponential backoff, then a provider chain
    (for example yfinance -> Alpha Vantage);
  * graceful degradation: fresh data -> stale cache (flagged) -> labelled
    sample data (quotes only) -> MarketDataUnavailable.

Every Quote / PriceHistory carries freshness fields (fetched_at, source,
is_stale, is_sample) so the UI can always say how current a number is.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from src.core.cache import TTLCache
from src.core.schemas import NewsItem, PriceHistory, Profile, Quote, QuoteError
from src.market.instruments import lookup, normalize_symbol
from src.market.providers import PERIODS, MarketDataError, MarketDataProvider, ProviderError, SymbolNotFound

log = logging.getLogger(__name__)

MAX_SYMBOLS = 20


class MarketDataUnavailable(MarketDataError):
    """The provider failed and there is no cached or sample data to fall back on."""


class InvalidRequest(MarketDataError):
    pass


class _NotFound:
    """Negative-cache marker for symbols the provider does not know."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_symbol(symbol: str) -> str:
    s = normalize_symbol(symbol)
    if not s or len(s) > 12 or not all(ch.isalnum() or ch in "-^=" for ch in s):
        raise InvalidRequest(f"'{symbol}' is not a valid ticker symbol.")
    return s


class MarketDataService:
    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        quote_ttl: float = 60,
        history_ttl: float = 900,
        news_ttl: float = 600,
        profile_ttl: float = 86_400,
        not_found_ttl: float = 600,
        max_stale: float = 86_400,
        timeout: float = 8.0,
        max_concurrency: int = 6,
        retries: int = 0,
        backoff_base: float = 0.4,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable = asyncio.sleep,
    ):
        self.provider = provider
        self._ttl = {"quote": quote_ttl, "history": history_ttl, "news": news_ttl, "profile": profile_ttl}
        self._not_found_ttl, self._max_stale, self._timeout = not_found_ttl, max_stale, timeout
        self._cache = TTLCache(clock=clock)
        self._max_concurrency = max_concurrency
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self._inflight: dict[tuple, asyncio.Future] = {}
        self._retries, self._backoff, self._sleep = retries, backoff_base, sleep
        self.counters = {"cache_hits": 0, "provider_calls": 0, "retries": 0, "provider_errors": 0,
                         "stale_served": 0, "sample_served": 0}

    @property
    def source(self) -> str:
        return self.provider.name

    # ---------- core ----------

    def _semaphore(self) -> asyncio.Semaphore:
        # A semaphore belongs to one event loop; tests and scripts may use several.
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self._max_concurrency), loop
        return self._sem

    async def _call(self, fn: Callable, *args):
        """Run a provider call in a worker thread with a timeout; retry transient failures with backoff."""
        for attempt in range(self._retries + 1):
            self.counters["provider_calls"] += 1
            try:
                async with self._semaphore():
                    return await asyncio.wait_for(asyncio.to_thread(fn, *args), self._timeout)
            except (ProviderError, asyncio.TimeoutError, OSError):
                if attempt >= self._retries:
                    raise
                self.counters["retries"] += 1
                await self._sleep(self._backoff * (2 ** attempt))

    def stats(self) -> dict:
        provider = self.provider
        return {
            "provider": provider.name,
            "chain": getattr(provider, "chain", [provider.name]),
            "provider_switches": getattr(provider, "switches", 0),
            "cached_entries": len(self._cache),
            "ttl_seconds": dict(self._ttl),
            **self.counters,
        }

    async def _fetch(self, key: tuple, fetch: Callable[[], Awaitable[Any]]) -> tuple[Any, bool]:
        """Return (value, is_stale). Raises SymbolNotFound or MarketDataUnavailable."""
        kind = key[0]
        hit = self._cache.get(key)
        if hit is not None:
            value, age = hit
            if isinstance(value, _NotFound):
                if age < self._not_found_ttl:
                    raise SymbolNotFound(key[1])
            elif age < self._ttl[kind]:
                self.counters["cache_hits"] += 1
                return value, False

        # Coalesce concurrent fetches of the same key.
        fut = self._inflight.get(key)
        if fut is None:
            fut = asyncio.ensure_future(fetch())
            self._inflight[key] = fut
            fut.add_done_callback(lambda _: self._inflight.pop(key, None))
        try:
            value = await asyncio.shield(fut)
        except SymbolNotFound:
            self._cache.set(key, _NotFound())
            raise
        except (ProviderError, asyncio.TimeoutError, OSError) as exc:
            log.warning("market data %s failed: %s", key, exc)
            self.counters["provider_errors"] += 1
            if hit is not None and not isinstance(hit[0], _NotFound) and hit[1] < self._max_stale:
                self.counters["stale_served"] += 1
                return hit[0], True
            raise MarketDataUnavailable("Market data is temporarily unavailable. Please try again shortly.") from exc
        self._cache.set(key, value)
        return value, False

    # ---------- public API ----------

    async def quote(self, symbol: str) -> Quote:
        symbol = validate_symbol(symbol)

        async def fetch():
            raw = await self._call(self.provider.quote, symbol)
            return raw, _now_iso()

        try:
            (raw, fetched_at), stale = await self._fetch(("quote", symbol), fetch)
        except MarketDataUnavailable:
            sample = self._sample_quote(symbol)
            if sample is None:
                raise
            self.counters["sample_served"] += 1
            return sample

        change = raw.price - raw.previous_close if raw.previous_close else 0.0
        return Quote(
            symbol=symbol,
            name=self._known_name(symbol),
            price=round(raw.price, 4),
            previous_close=raw.previous_close,
            change=round(change, 4),
            change_pct=(change / raw.previous_close) if raw.previous_close else 0.0,
            currency=raw.currency,
            year_high=raw.year_high,
            year_low=raw.year_low,
            fetched_at=fetched_at,
            source="cache" if stale else self.provider.name,
            is_stale=stale,
            is_sample=self.provider.is_sample,
        )

    async def quotes(self, symbols: Iterable[str]) -> tuple[list[Quote], list[QuoteError]]:
        """Fetch several quotes concurrently. Per-symbol failures are reported, not raised."""
        unique = list(dict.fromkeys(s for s in (normalize_symbol(x) for x in symbols) if s))
        if len(unique) > MAX_SYMBOLS:
            raise InvalidRequest(f"Ask for at most {MAX_SYMBOLS} symbols at a time.")
        results = await asyncio.gather(*(self.quote(s) for s in unique), return_exceptions=True)
        quotes: list[Quote] = []
        errors: list[QuoteError] = []
        for symbol, r in zip(unique, results):
            if isinstance(r, Quote):
                quotes.append(r)
            elif isinstance(r, SymbolNotFound):
                errors.append(QuoteError(symbol=symbol, error_code="symbol_not_found", message=str(r)))
            elif isinstance(r, InvalidRequest):
                errors.append(QuoteError(symbol=symbol, error_code="invalid_symbol", message=str(r)))
            elif isinstance(r, MarketDataUnavailable):
                errors.append(QuoteError(symbol=symbol, error_code="market_data_unavailable", message=str(r)))
            else:
                log.error("unexpected quote error for %s", symbol, exc_info=r)
                errors.append(QuoteError(symbol=symbol, error_code="internal_error", message="Unexpected error."))
        return quotes, errors

    async def history(self, symbol: str, period: str = "1mo") -> PriceHistory:
        symbol = validate_symbol(symbol)
        if period not in PERIODS:
            raise InvalidRequest(f"period must be one of {', '.join(PERIODS)}.")
        points, stale = await self._fetch(
            ("history", symbol, period), lambda: self._call(self.provider.history, symbol, period)
        )
        return PriceHistory(
            symbol=symbol, period=period, points=points,
            source="cache" if stale else self.provider.name, is_stale=stale, is_sample=self.provider.is_sample,
        )

    async def news(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        symbol = validate_symbol(symbol)
        limit = max(1, min(limit, 20))
        items, _ = await self._fetch(("news", symbol), lambda: self._call(self.provider.news, symbol, 20))
        return items[:limit]

    async def profile(self, symbol: str) -> Profile:
        """Classification for a symbol. Known tickers never hit the network; failures degrade to 'Unknown'."""
        symbol = validate_symbol(symbol)
        inst = lookup(symbol)
        if inst is not None:
            return Profile(symbol=symbol, name=inst.name, asset_class=inst.asset_class,
                           sector=inst.sector, diversified=inst.diversified)
        try:
            value, _ = await self._fetch(("profile", symbol), lambda: self._call(self.provider.profile, symbol))
            return value
        except MarketDataError:
            return Profile(symbol=symbol)

    # ---------- helpers ----------

    def _known_name(self, symbol: str) -> str | None:
        inst = lookup(symbol)
        if inst is not None:
            return inst.name
        hit = self._cache.get(("profile", symbol))
        return hit[0].name if hit is not None and isinstance(hit[0], Profile) else None

    @staticmethod
    def _sample_quote(symbol: str) -> Quote | None:
        inst = lookup(symbol)
        if inst is None:
            return None
        return Quote(
            symbol=symbol, name=inst.name, price=inst.sample_price, previous_close=inst.sample_price,
            change=0.0, change_pct=0.0, fetched_at=_now_iso(), source="sample", is_stale=True, is_sample=True,
        )


def create_market_service(settings_or_name, quote_ttl: float = 60, timeout: float = 8.0) -> MarketDataService:
    """Build the service from Settings (preferred) or from a bare provider name (tests, scripts).

    With Settings, the provider chain is: the configured primary, then Alpha Vantage as a second
    live source when ALPHAVANTAGE_API_KEY is set (or yfinance when Alpha Vantage is primary).
    """
    from src.market.providers import AlphaVantageProvider, FallbackProvider, SampleProvider, YFinanceProvider

    if isinstance(settings_or_name, str):
        name, av_key, kwargs = settings_or_name, None, {"quote_ttl": quote_ttl, "timeout": timeout}
    else:
        st = settings_or_name
        name, av_key = st.market_data_provider, st.alphavantage_api_key
        kwargs = {"quote_ttl": st.market_quote_ttl, "history_ttl": st.market_history_ttl,
                  "news_ttl": st.market_news_ttl, "timeout": st.market_timeout, "retries": st.market_retries}

    if name == "sample":
        provider: MarketDataProvider = SampleProvider()
    elif name == "yfinance":
        chain = [YFinanceProvider()] + ([AlphaVantageProvider(av_key, kwargs["timeout"])] if av_key else [])
        provider = FallbackProvider(chain) if len(chain) > 1 else chain[0]
    elif name == "alphavantage":
        if not av_key:
            raise RuntimeError("MARKET_DATA_PROVIDER=alphavantage needs ALPHAVANTAGE_API_KEY")
        provider = FallbackProvider([AlphaVantageProvider(av_key, kwargs["timeout"]), YFinanceProvider()])
    else:
        raise RuntimeError(f"Unknown MARKET_DATA_PROVIDER={name!r}; use yfinance, alphavantage or sample")
    return MarketDataService(provider, **kwargs)

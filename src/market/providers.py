"""Market data providers.

Providers are synchronous and do no caching or fallback: MarketDataService wraps
them with threads, timeouts, a cache and graceful degradation. That keeps each
provider small and easy to swap (yfinance today, a paid API later).
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

from src.core.schemas import NewsItem, PricePoint, Profile
from src.market.instruments import lookup

log = logging.getLogger(__name__)

# period -> (yfinance period, bar interval)
PERIODS: dict[str, tuple[str, str]] = {
    "1mo": ("1mo", "1d"),
    "6mo": ("6mo", "1d"),
    "1y": ("1y", "1d"),
    "5y": ("5y", "1wk"),
}


class MarketDataError(Exception):
    """Base class for market data failures."""


class SymbolNotFound(MarketDataError):
    def __init__(self, symbol: str):
        super().__init__(f"No market data found for {symbol}.")
        self.symbol = symbol


class ProviderError(MarketDataError):
    """The provider failed (network, rate limit, unexpected payload). Retrying later may work."""


@dataclass(frozen=True)
class RawQuote:
    symbol: str
    price: float
    previous_close: float | None
    currency: str
    year_high: float | None
    year_low: float | None


class MarketDataProvider(Protocol):
    name: str
    is_sample: bool

    def quote(self, symbol: str) -> RawQuote: ...
    def history(self, symbol: str, period: str) -> list[PricePoint]: ...
    def news(self, symbol: str, limit: int) -> list[NewsItem]: ...
    def profile(self, symbol: str) -> Profile: ...


def _finite(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------- yfinance ----------

_ASSET_CLASS = {"EQUITY": "Stock", "ETF": "Equity fund", "MUTUALFUND": "Mutual fund", "INDEX": "Index",
                "CRYPTOCURRENCY": "Crypto", "CURRENCY": "Currency", "FUTURE": "Future"}


class YFinanceProvider:
    """Yahoo Finance via the unofficial yfinance library. Prices may be delayed ~15 minutes."""

    name = "yfinance"
    is_sample = False

    def __init__(self):
        import yfinance  # imported lazily so tests and offline mode never need it

        self._yf = yfinance

    def _ticker(self, symbol: str):
        return self._yf.Ticker(symbol)

    def quote(self, symbol: str) -> RawQuote:
        try:
            fi = self._ticker(symbol).fast_info
            price = _finite(fi.last_price)
            prev = _finite(fi.previous_close)
            currency = fi.currency or "USD"
            high, low = _finite(fi.year_high), _finite(fi.year_low)
        except KeyError as exc:  # yfinance raises KeyError when Yahoo has no such symbol
            raise SymbolNotFound(symbol) from exc
        except Exception as exc:
            raise ProviderError(f"quote {symbol}: {exc}") from exc
        if price is None or price <= 0:
            raise SymbolNotFound(symbol)
        return RawQuote(symbol, price, prev, currency, high, low)

    def history(self, symbol: str, period: str) -> list[PricePoint]:
        yf_period, interval = PERIODS[period]
        try:
            df = self._ticker(symbol).history(period=yf_period, interval=interval, auto_adjust=True)
        except Exception as exc:
            raise ProviderError(f"history {symbol}: {exc}") from exc
        if df is None or df.empty:
            raise SymbolNotFound(symbol)
        points = [
            PricePoint(date=idx.date().isoformat(), close=round(c, 4))
            for idx, c in zip(df.index, df["Close"]) if _finite(c) is not None
        ]
        if not points:
            raise SymbolNotFound(symbol)
        return points

    def news(self, symbol: str, limit: int) -> list[NewsItem]:
        try:
            raw = self._ticker(symbol).news or []
        except Exception as exc:
            raise ProviderError(f"news {symbol}: {exc}") from exc
        items: list[NewsItem] = []
        for entry in raw:
            # yfinance >= 0.2.50 nests fields under "content"; older versions are flat.
            c = entry.get("content", entry) if isinstance(entry, dict) else {}
            title = c.get("title")
            if not title:
                continue
            provider = c.get("provider") or {}
            url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link")
            published = c.get("pubDate")
            if published is None and c.get("providerPublishTime"):
                published = datetime.fromtimestamp(c["providerPublishTime"], tz=timezone.utc).isoformat()
            items.append(NewsItem(
                title=title, url=url, published_at=published, symbol=symbol,
                publisher=provider.get("displayName") if isinstance(provider, dict) else c.get("publisher"),
            ))
            if len(items) >= limit:
                break
        return items

    def profile(self, symbol: str) -> Profile:
        try:
            info = self._ticker(symbol).info or {}
        except Exception as exc:
            raise ProviderError(f"profile {symbol}: {exc}") from exc
        quote_type = info.get("quoteType")
        if not quote_type:
            raise SymbolNotFound(symbol)
        is_fund = quote_type in {"ETF", "MUTUALFUND"}
        category = (info.get("category") or "").lower()
        if is_fund and "bond" in category:
            asset_class, sector = "Bond fund", "Fixed income"
        else:
            asset_class = _ASSET_CLASS.get(quote_type, quote_type.title())
            sector = info.get("sector") or ("Diversified" if is_fund else "Unknown")
        return Profile(
            symbol=symbol,
            name=info.get("shortName") or info.get("longName"),
            asset_class=asset_class,
            sector=sector,
            # Broad "blend" funds are diversified; sector or thematic funds are not.
            diversified=is_fund and any(k in category for k in ("blend", "bond", "foreign", "world", "diversified")),
        )


# ---------- Alpha Vantage ----------

class AlphaVantageProvider:
    """Alpha Vantage REST API (https://www.alphavantage.co/documentation/). Needs ALPHAVANTAGE_API_KEY.

    The free tier allows very few calls per day, and signals "slow down" with HTTP 200 and a
    "Note" / "Information" field, so those are turned into ProviderError: the service then
    serves cached data or moves to the next provider in the chain.
    """

    name = "alphavantage"
    is_sample = False
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: float = 8.0, http_get=None):
        if not api_key:
            raise ValueError("AlphaVantageProvider needs an API key")
        self._key, self._timeout = api_key, timeout
        self._http_get = http_get  # injectable for tests: (url, params, timeout) -> dict

    def _get(self, **params) -> dict:
        params["apikey"] = self._key
        try:
            if self._http_get is not None:
                data = self._http_get(self.URL, params, self._timeout)
            else:
                import httpx  # imported lazily

                resp = httpx.get(self.URL, params=params, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise ProviderError(f"alphavantage {params.get('function')}: {exc}") from exc
        if not isinstance(data, dict):
            raise ProviderError("alphavantage returned an unexpected payload")
        for key in ("Note", "Information"):
            if key in data:  # rate limit or premium-only endpoint
                raise ProviderError(f"alphavantage: {str(data[key])[:120]}")
        if "Error Message" in data:
            raise SymbolNotFound(str(params.get("symbol") or params.get("tickers") or "?"))
        return data

    def quote(self, symbol: str) -> RawQuote:
        q = self._get(function="GLOBAL_QUOTE", symbol=symbol).get("Global Quote") or {}
        price = _finite(q.get("05. price"))
        if price is None or price <= 0:
            raise SymbolNotFound(symbol)
        return RawQuote(symbol, price, _finite(q.get("08. previous close")), "USD", None, None)

    def history(self, symbol: str, period: str) -> list[PricePoint]:
        weekly = period in ("1y", "5y")
        data = self._get(function="TIME_SERIES_WEEKLY" if weekly else "TIME_SERIES_DAILY", symbol=symbol)
        series = data.get("Weekly Time Series" if weekly else "Time Series (Daily)") or {}
        if not series:
            raise SymbolNotFound(symbol)
        days = {"1mo": 31, "6mo": 183, "1y": 366, "5y": 1827}[period]
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        points = [
            PricePoint(date=day, close=round(close, 4))
            for day, row in sorted(series.items())
            if day >= cutoff and (close := _finite(row.get("4. close"))) is not None
        ]
        if not points:
            raise SymbolNotFound(symbol)
        return points

    def news(self, symbol: str, limit: int) -> list[NewsItem]:
        feed = self._get(function="NEWS_SENTIMENT", tickers=symbol, limit=str(max(limit, 1))).get("feed") or []
        items: list[NewsItem] = []
        for entry in feed[:limit]:
            if not entry.get("title"):
                continue
            stamp = entry.get("time_published") or ""
            try:
                published = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                published = None
            items.append(NewsItem(title=entry["title"], url=entry.get("url"), publisher=entry.get("source"),
                                  published_at=published, symbol=symbol))
        return items

    def profile(self, symbol: str) -> Profile:
        info = self._get(function="OVERVIEW", symbol=symbol)
        if not info.get("Symbol"):
            raise SymbolNotFound(symbol)
        asset_type = (info.get("AssetType") or "").lower()
        return Profile(symbol=symbol, name=info.get("Name"),
                       asset_class="Stock" if "stock" in asset_type else (info.get("AssetType") or "Unknown"),
                       sector=(info.get("Sector") or "Unknown").title())


# ---------- Provider chain ----------

class FallbackProvider:
    """Try each provider in order. A ProviderError (outage, rate limit) moves on to the next one;
    SymbolNotFound from the first provider that answers is trusted and returned straight away."""

    is_sample = False

    def __init__(self, providers: list):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.name = providers[0].name
        self.switches = 0  # how many calls were answered by a provider other than the first

    @property
    def chain(self) -> list[str]:
        return [p.name for p in self.providers]

    def _try(self, method: str, *args):
        last: Exception | None = None
        for i, provider in enumerate(self.providers):
            try:
                result = getattr(provider, method)(*args)
            except ProviderError as exc:
                log.warning("market provider %s.%s failed: %s", provider.name, method, exc)
                last = exc
                continue
            if i:
                self.switches += 1
            return result
        raise last or ProviderError("no market data provider available")

    def quote(self, symbol: str) -> RawQuote:
        return self._try("quote", symbol)

    def history(self, symbol: str, period: str) -> list[PricePoint]:
        return self._try("history", symbol, period)

    def news(self, symbol: str, limit: int) -> list[NewsItem]:
        return self._try("news", symbol, limit)

    def profile(self, symbol: str) -> Profile:
        return self._try("profile", symbol)


# ---------- Offline sample data ----------

class SampleProvider:
    """Deterministic, clearly-labelled sample data for tests, demos and offline development."""

    name = "sample"
    is_sample = True

    def quote(self, symbol: str) -> RawQuote:
        inst = lookup(symbol)
        if inst is None:
            raise SymbolNotFound(symbol)
        return RawQuote(symbol, inst.sample_price, inst.sample_price, "USD", None, None)

    def history(self, symbol: str, period: str) -> list[PricePoint]:
        inst = lookup(symbol)
        if inst is None:
            raise SymbolNotFound(symbol)
        _, interval = PERIODS[period]
        days = {"1mo": 30, "6mo": 182, "1y": 365, "5y": 1825}[period]
        step = 7 if interval == "1wk" else 1
        seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
        today = date.today()
        n = days // step
        points = []
        for i in range(n + 1):
            # A smooth, repeatable wiggle that ends exactly at the sample price.
            t = (n - i) / max(n, 1)
            drift = 1 - 0.08 * t
            wave = 0.03 * math.sin(i / 6 + seed % 7)
            points.append(PricePoint(
                date=(today - timedelta(days=(n - i) * step)).isoformat(),
                close=round(inst.sample_price * (drift + wave * t), 2),
            ))
        return points

    def news(self, symbol: str, limit: int) -> list[NewsItem]:
        return []

    def profile(self, symbol: str) -> Profile:
        inst = lookup(symbol)
        if inst is None:
            raise SymbolNotFound(symbol)
        return Profile(symbol=symbol, name=inst.name, asset_class=inst.asset_class,
                       sector=inst.sector, diversified=inst.diversified)

from typing import Literal

from fastapi import APIRouter, Depends, Query

from src.core.schemas import NewsResponse, PriceHistory, QuotesResponse, SymbolDetail
from src.market.instruments import OVERVIEW_SYMBOLS
from src.market.service import MAX_SYMBOLS, MarketDataService, validate_symbol
from src.web_app.api.deps import api_rate_limit, get_market

router = APIRouter(prefix="/api/market", tags=["market"], dependencies=[Depends(api_rate_limit)])

GENERAL_NEWS_SYMBOL = "SPY"


@router.get("/overview", response_model=QuotesResponse)
async def overview(market: MarketDataService = Depends(get_market)):
    quotes, errors = await market.quotes(OVERVIEW_SYMBOLS)
    return QuotesResponse(quotes=quotes, errors=errors)


@router.get("/quotes", response_model=QuotesResponse)
async def quotes(
    symbols: str = Query(min_length=1, max_length=300, description=f"Comma-separated, at most {MAX_SYMBOLS}"),
    market: MarketDataService = Depends(get_market),
):
    found, errors = await market.quotes(s for s in symbols.split(",") if s.strip())
    return QuotesResponse(quotes=found, errors=errors)


@router.get("/symbols/{symbol}", response_model=SymbolDetail)
async def symbol_detail(symbol: str, market: MarketDataService = Depends(get_market)):
    symbol = validate_symbol(symbol)
    quote = await market.quote(symbol)  # SymbolNotFound -> 404, MarketDataUnavailable -> 503
    profile = await market.profile(symbol)
    if quote.name is None and profile.name:
        quote = quote.model_copy(update={"name": profile.name})
    return SymbolDetail(quote=quote, profile=profile)


@router.get("/history/{symbol}", response_model=PriceHistory)
async def history(
    symbol: str,
    period: Literal["1mo", "6mo", "1y", "5y"] = "1mo",
    market: MarketDataService = Depends(get_market),
):
    return await market.history(symbol, period)


@router.get("/news", response_model=NewsResponse)
async def news(
    symbol: str | None = Query(default=None, max_length=12),
    limit: int = Query(default=8, ge=1, le=20),
    market: MarketDataService = Depends(get_market),
):
    target = validate_symbol(symbol) if symbol else GENERAL_NEWS_SYMBOL
    return NewsResponse(symbol=target if symbol else None, items=await market.news(target, limit))

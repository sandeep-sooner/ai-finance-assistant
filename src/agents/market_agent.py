import asyncio
import re

from src.agents.base_agent import AgentReply, BaseAgent, recent_user_texts
from src.core.llm import LLM, Message
from src.core.schemas import PriceHistory, Quote
from src.market.instruments import OVERVIEW_SYMBOLS
from src.market.service import MarketDataError, MarketDataService
from src.market.symbols import extract_symbols
from src.rag.retriever import Retriever

HELP_TEXT = (
    "I can look up prices for you. Mention a ticker such as **AAPL** or **$SPY**, or a well-known name like "
    "*Apple* or *the S&P 500*. You can also browse prices and charts in the **Market** tab."
)
_GENERAL_MARKET = re.compile(r"\b(market|stocks?)\b.*\b(today|doing|now|this week)\b|\bhow is the market\b", re.I)


def yahoo_source(symbol: str) -> dict:
    return {"title": f"{symbol} on Yahoo Finance", "url": f"https://finance.yahoo.com/quote/{symbol}",
            "category": "market data", "score": 1.0}


def describe_quote(q: Quote, h: PriceHistory | None) -> str:
    name = f" ({q.name})" if q.name else ""
    line = (f"{q.symbol}{name}: {q.price:,.2f} {q.currency}; today {q.change:+,.2f} ({q.change_pct * 100:+.2f}%) "
            f"vs previous close {q.previous_close or 0:,.2f}")
    if h and len(h.points) > 1:
        first, last = h.points[0].close, h.points[-1].close
        line += f"; 1-month change {(last / first - 1) * 100:+.1f}% (from {first:,.2f})"
    if q.year_low and q.year_high:
        line += f"; 52-week range {q.year_low:,.2f} to {q.year_high:,.2f}"
    if q.is_sample:
        line += "; SAMPLE PRICE, not real market data"
    elif q.is_stale:
        line += f"; from an earlier fetch at {q.fetched_at} because the data source did not respond"
    else:
        line += f"; fetched {q.fetched_at} UTC from Yahoo Finance, may be delayed about 15 minutes"
    return line


class MarketAgent(BaseAgent):
    name = "market"
    rag_k = 1
    label = "Market data"
    system_prompt = (
        "You are the Market data agent of an educational assistant for beginners. You are given current "
        "quotes that were just fetched. Report the numbers clearly (price, today's move, recent trend), "
        "explain in plain language what they mean (for example what a 52-week range or a daily percentage "
        "move tells you), and say when the data is from and that it may be delayed. Use only the numbers "
        "given. Never predict prices, never say whether something is a good buy, and never recommend buying "
        "or selling. Keep it short: a few sentences or bullets per symbol."
    )

    def __init__(self, llm: LLM, retriever: Retriever | None = None, market: MarketDataService | None = None):
        super().__init__(llm, retriever)
        self.market = market

    async def _history(self, symbol: str) -> PriceHistory | None:
        try:
            return await self.market.history(symbol, "1mo")
        except MarketDataError:
            return None

    async def respond(self, history: list[Message], question: str, advice_request: bool = False,
                      level: str = "beginner") -> AgentReply:
        if self.market is None:
            return AgentReply.canned(HELP_TEXT)
        symbols = extract_symbols(question)
        if not symbols and _GENERAL_MARKET.search(question):
            symbols = list(OVERVIEW_SYMBOLS[:3])
        if not symbols:  # follow-up such as "and how has it done this month?": reuse the last ticker discussed
            symbols = next((s for s in map(extract_symbols, recent_user_texts(history)) if s), [])
        if not symbols:
            return AgentReply.canned(HELP_TEXT)

        quotes, errors = await self.market.quotes(symbols)
        if not quotes:
            if errors and all(e.error_code == "symbol_not_found" for e in errors):
                missing = ", ".join(e.symbol for e in errors)
                return AgentReply.canned(f"I could not find market data for **{missing}**. Please check the ticker.")
            return AgentReply.canned("I could not reach the market data service just now. Please try again in a minute.")

        histories = await asyncio.gather(*(self._history(q.symbol) for q in quotes))
        lines = [describe_quote(q, h) for q, h in zip(quotes, histories)]
        if errors:
            lines.append("Could not get data for: " + ", ".join(f"{e.symbol} ({e.error_code})" for e in errors))
        return self.prepare(
            history, question, advice_request, level=level,
            extra_context="\n".join(lines), extra_sources=[yahoo_source(q.symbol) for q in quotes],
        )

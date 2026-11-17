import asyncio

from src.agents.base_agent import AgentReply, BaseAgent, recent_user_texts
from src.core.llm import LLM, Message
from src.core.schemas import NewsItem
from src.market.service import MarketDataError, MarketDataService
from src.market.symbols import extract_symbols
from src.rag.retriever import Retriever

GENERAL_SYMBOL = "SPY"  # broad US market headlines when no ticker is named
MAX_ITEMS = 8


class NewsAgent(BaseAgent):
    name = "news"
    rag_k = 1
    label = "News"
    system_prompt = (
        "You are the News agent of an educational assistant for beginners. You are given recent headlines "
        "(titles only, not full articles), numbered [1], [2], and so on. Summarize the main themes in three to "
        "five bullets, citing headlines like [2]. Only state what the headlines say; do not invent details, "
        "and say so when a headline alone is not enough to know the impact. Briefly explain any finance terms. "
        "Never predict prices or suggest buying or selling."
    )

    def __init__(self, llm: LLM, retriever: Retriever | None = None, market: MarketDataService | None = None):
        super().__init__(llm, retriever)
        self.market = market

    async def _news(self, symbol: str) -> list[NewsItem]:
        try:
            return await self.market.news(symbol, limit=5)
        except MarketDataError:
            return []

    async def respond(self, history: list[Message], question: str, advice_request: bool = False,
                      level: str = "beginner") -> AgentReply:
        if self.market is None:
            return AgentReply.canned("News is not available right now.")
        symbols = extract_symbols(question, limit=3)
        if not symbols and len(question.split()) <= 8:  # short follow-up: keep talking about the same ticker
            symbols = next((s for s in (extract_symbols(t, limit=3) for t in recent_user_texts(history, 1)) if s), [])
        scope = ", ".join(symbols) if symbols else "the broad US stock market"
        batches = await asyncio.gather(*(self._news(s) for s in (symbols or [GENERAL_SYMBOL])))

        items: list[NewsItem] = []
        seen: set[str] = set()
        for batch in batches:
            for item in batch:
                key = item.title.strip().lower()
                if key not in seen:
                    seen.add(key)
                    items.append(item)
        items = items[:MAX_ITEMS]
        if not items:
            return AgentReply.canned(f"I could not find recent headlines for {scope}. Please try again later.")

        context = f"Headlines about {scope}:\n" + "\n".join(
            f"[{i}] {it.title} ({it.publisher or 'unknown source'}, {(it.published_at or '')[:10]})"
            for i, it in enumerate(items, 1)
        )
        sources = [{"title": it.title, "url": it.url, "category": it.publisher or "news", "score": 1.0} for it in items]
        return self.prepare(history, question, advice_request, extra_context=context, extra_sources=sources, level=level)

"""Model Context Protocol server: exposes the assistant's tools to Claude Desktop (or any MCP client).

Run (stdio transport, which is what Claude Desktop launches):

    python -m src.mcp_server.server

Tools:
    ask_finance_assistant   - the full multi-agent workflow (routing, RAG, live data)
    search_knowledge_base   - hybrid search over the financial education articles
    lookup_glossary         - definitions of finance terms
    get_stock_quote         - live quote with freshness fields
    get_price_history       - closing prices for a period
    get_market_news         - recent headlines for a ticker
    analyze_portfolio       - allocation, diversification, risk and cost metrics
    plan_savings_goal       - inflation-adjusted goal projection with three risk scenarios

Everything here reuses the same services as the web API, so the numbers are identical.
stdout is the MCP channel, so all logging goes to stderr.
"""
from __future__ import annotations

import logging
import sys
import uuid
from functools import lru_cache

try:  # mcp >= 2: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from src.core.config import Settings
from src.core.guardrails import DISCLAIMER
from src.core.schemas import GoalRequest, Holding
from src.goals.planner import plan_goal
from src.market.service import MarketDataError, create_market_service
from src.portfolio.analysis import analyze_live
from src.rag.knowledge_base import load_glossary

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

mcp = FastMCP(
    "ai-finance-assistant",
    instructions=(
        "Tools for financial education: explain concepts from a curated knowledge base, look up market "
        "data, analyse a portfolio and project savings goals. Results are educational, not financial advice."
    ),
)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings.load()


@lru_cache(maxsize=1)
def _market():
    return create_market_service(_settings())


@lru_cache(maxsize=1)
def _retriever():
    from src.rag.retriever import create_retriever

    return create_retriever(_settings())


@lru_cache(maxsize=1)
def _graph():
    from langgraph.checkpoint.memory import MemorySaver

    from src.core.llm import create_gateway
    from src.workflow.graph import build_graph

    s = _settings()
    return build_graph(create_gateway(s), _retriever(), MemorySaver(), market=_market(),
                       max_history=s.max_history_messages, max_chars=s.max_message_chars,
                       llm_routing=s.router_llm_fallback and s.llm_provider != "fake")


@mcp.tool()
async def ask_finance_assistant(question: str, session_id: str = "") -> dict:
    """Ask the multi-agent finance assistant a question. It routes to the right specialist
    (Q&A, portfolio, market, goals, news, tax), grounds the answer in its knowledge base and
    live data, and returns the answer with sources. Pass the same session_id to continue a conversation."""
    session_id = session_id or f"mcp-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": session_id}}
    answer, sources, meta = [], [], {}
    async for item in _graph().astream({"messages": [("user", question)]}, config, stream_mode="custom"):
        if item["event"] == "token":
            answer.append(item["data"])
        elif item["event"] == "sources":
            sources = item["data"]
        elif item["event"] == "meta":
            meta = item["data"]
    return {
        "answer": "".join(answer).strip(),
        "agents": meta.get("agents", []),
        "sources": [{"title": s["title"], "url": s.get("url")} for s in sources],
        "disclaimer": meta.get("disclaimer") or DISCLAIMER,
        "session_id": session_id,
    }


@mcp.tool()
def search_knowledge_base(query: str, category: str = "", top_k: int = 4) -> list[dict]:
    """Search the financial education knowledge base. Optional category: basics, stocks, bonds,
    funds, portfolio, planning, tax or markets. Returns passages with source, URL and relevance score."""
    retriever = _retriever()
    results = retriever.search(query, category or None, max(1, min(top_k, 10))) if hasattr(retriever, "stats") \
        else retriever.search(query, category or None)
    return [{k: r[k] for k in ("title", "category", "source", "url", "score", "text")} for r in results]


@mcp.tool()
def lookup_glossary(term: str) -> list[dict]:
    """Look up a finance term (for example 'expense ratio' or 'beta') in the glossary."""
    needle = term.strip().lower()
    return [g for g in load_glossary() if needle in g["term"].lower()][:5]


@mcp.tool()
async def get_stock_quote(symbol: str) -> dict:
    """Current quote for a ticker (for example AAPL or SPY): price, change, 52-week range, and
    freshness fields (fetched_at, source, is_stale, is_sample). Prices may be delayed ~15 minutes."""
    try:
        return (await _market().quote(symbol)).model_dump()
    except MarketDataError as exc:
        return {"error": str(exc), "symbol": symbol}


@mcp.tool()
async def get_price_history(symbol: str, period: str = "1mo") -> dict:
    """Closing prices for a ticker. period is one of: 1mo, 6mo, 1y, 5y."""
    try:
        h = await _market().history(symbol, period)
    except MarketDataError as exc:
        return {"error": str(exc), "symbol": symbol}
    first, last = h.points[0].close, h.points[-1].close
    return {"symbol": h.symbol, "period": h.period, "source": h.source, "is_sample": h.is_sample,
            "change_pct": round((last / first - 1) * 100, 2) if first else None,
            "points": [p.model_dump() for p in h.points]}


@mcp.tool()
async def get_market_news(symbol: str = "SPY", limit: int = 5) -> list[dict]:
    """Recent headlines for a ticker (default SPY for broad US market news). Titles and links only."""
    try:
        return [n.model_dump() for n in await _market().news(symbol, limit)]
    except MarketDataError as exc:
        return [{"error": str(exc), "symbol": symbol}]


@mcp.tool()
async def analyze_portfolio(holdings: list[dict]) -> dict:
    """Analyse a portfolio. holdings is a list like [{"symbol": "AAPL", "shares": 10, "avg_cost": 150}]
    (avg_cost is optional). Returns total value, allocation by asset class and sector, concentration,
    a diversification score, a risk level, fund costs and plain-language flags."""
    parsed = [Holding(**h) for h in holdings]
    result = (await analyze_live(parsed, _market())).model_dump()
    result["disclaimer"] = DISCLAIMER
    return result


@mcp.tool()
def plan_savings_goal(target_amount: float, years: int, current_savings: float = 0.0,
                      monthly_contribution: float = 0.0, risk_profile: str = "balanced") -> dict:
    """Project a savings goal. target_amount is in today's money; risk_profile is conservative,
    balanced or growth. Returns the inflation-adjusted target, projected value, whether the plan is
    on track, the monthly saving required and a comparison of the three risk profiles."""
    req = GoalRequest(target_amount=target_amount, years=years, current_savings=current_savings,
                      monthly_contribution=monthly_contribution, risk_profile=risk_profile)
    result = plan_goal(req).model_dump()
    result["disclaimer"] = DISCLAIMER
    return result


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

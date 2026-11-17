import json
from functools import lru_cache

from fastapi import APIRouter, Depends, Request

from src.core.schemas import PortfolioAnalysis, PortfolioRequest
from src.market.service import MarketDataService
from src.portfolio.analysis import analyze_live
from src.rag.knowledge_base import DATA_DIR
from src.web_app.api.deps import api_rate_limit, get_market, llm_rate_limit, sse_response, stream_reply

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

QUESTION = "Explain this portfolio analysis for me."


@lru_cache(maxsize=1)
def sample_portfolios() -> list[dict]:
    try:
        return json.loads((DATA_DIR / "sample_portfolios" / "portfolios.json").read_text(encoding="utf-8"))["portfolios"]
    except (OSError, ValueError, KeyError):
        return []


@router.get("/samples", dependencies=[Depends(api_rate_limit)])
async def samples():
    """Illustrative portfolios the UI can load with one click."""
    return {"portfolios": sample_portfolios()}


@router.post("/analyze", response_model=PortfolioAnalysis, dependencies=[Depends(api_rate_limit)])
async def analyze(req: PortfolioRequest, market: MarketDataService = Depends(get_market)):
    """Computed metrics only (no LLM), so the tab renders quickly."""
    return await analyze_live(req.holdings, market)


@router.post("/insights/stream", dependencies=[Depends(llm_rate_limit)])
async def insights(req: PortfolioRequest, request: Request, market: MarketDataService = Depends(get_market)):
    """Streams the Portfolio agent's plain-language explanation of the same metrics."""
    agent = request.app.state.agents["portfolio"]
    analysis = await analyze_live(req.holdings, market)
    reply = agent.explain([], QUESTION, analysis)
    return sse_response(stream_reply(request.app.state.gateway, agent.name, reply, QUESTION))

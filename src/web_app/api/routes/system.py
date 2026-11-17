"""Observability endpoints behind the System tab: what the router, cache and fallback chain are doing."""
import time

from fastapi import APIRouter, Depends, Query, Request

from src.core.model_router import ModelRouter
from src.web_app.api.deps import api_rate_limit

router = APIRouter(prefix="/api/system", tags=["system"], dependencies=[Depends(api_rate_limit)])


@router.get("/metrics")
async def metrics(request: Request):
    state = request.app.state
    gateway, retriever = state.gateway, state.retriever
    return {
        "uptime_seconds": round(time.time() - state.started_at),
        "llm": gateway.metrics.snapshot(),
        "gateway": gateway.describe(),
        "response_cache": gateway.cache.stats(),
        "retrieval": retriever.stats() if hasattr(retriever, "stats") else {"chunks": 0, "embedder": "none"},
        "market": state.market.stats(),
    }


@router.get("/route-preview")
async def route_preview(
    request: Request,
    q: str = Query(min_length=1, max_length=2000),
    agent: str = Query(default="finance_qa", max_length=30),
):
    """Explain which model tier a question would use and why, without calling any model."""
    model_router: ModelRouter = request.app.state.gateway.router
    decision = model_router.decide(q, agent)
    chain = [m.name for m in request.app.state.gateway.chain_for(decision.tier)]
    return {**decision.as_dict(), "threshold": model_router.threshold, "model_chain": chain}

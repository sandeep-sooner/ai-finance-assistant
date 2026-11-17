"""Shared API plumbing: per-client rate limits, app-state accessors, SSE responses."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from src.agents.base_agent import AgentReply
from src.core.gateway import CallTrace, LLMGateway
from src.core.guardrails import check_output
from src.core.sse import sse
from src.market.service import MarketDataService

log = logging.getLogger("api")


class ClientRateLimiter:
    """Per-client sliding window, in memory. Fine for one process; use Redis to scale out."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls, self.period = max_calls, period
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] >= self.period:
            q.popleft()
        if len(q) >= self.max_calls:
            raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
        q.append(now)


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def llm_rate_limit(request: Request) -> None:
    """Dependency for endpoints that call the LLM (costly)."""
    request.app.state.llm_limiter.check(_client(request))


def api_rate_limit(request: Request) -> None:
    """Dependency for data endpoints (cheap, but still bounded)."""
    request.app.state.api_limiter.check(_client(request))


def get_market(request: Request) -> MarketDataService:
    return request.app.state.market


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_response(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type="text/event-stream", headers=SSE_HEADERS)


async def stream_reply(gateway: LLMGateway, agent_name: str, reply: AgentReply, question: str) -> AsyncIterator[str]:
    """agent -> token* -> sources -> meta -> done. The gateway never raises, so there is no error branch:
    if every model is down the user gets the deterministic fallback built from the computed numbers."""
    yield sse("agent", agent_name)
    trace = CallTrace()
    parts: list[str] = []
    async for token in gateway.stream(
        reply.messages or [], agent=agent_name, question=question, trace=trace, fallback_text=reply.fallback_text,
    ):
        parts.append(token)
        yield sse("token", token)
    check = check_output("".join(parts))
    yield sse("sources", reply.sources)
    yield sse("meta", {"agents": [agent_name], "calls": [trace.as_dict()], "disclaimer": check.disclaimer,
                       "flags": check.flags, "route": {"plan": [agent_name], "method": "direct"}})
    yield sse("done", {})

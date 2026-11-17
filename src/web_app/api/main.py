"""FastAPI app. Run from the repo root:  uvicorn src.web_app.api.main:app --reload --port 8000"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import MemorySaver
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import ROOT, Settings
from src.core.llm import create_gateway
from src.market.providers import SymbolNotFound
from src.market.service import InvalidRequest, MarketDataUnavailable, create_market_service
from src.rag.retriever import create_retriever
from src.web_app.api.deps import ClientRateLimiter
from src.web_app.api.routes import chat, goals, knowledge, market, portfolio, sessions, system
from src.workflow.graph import build_agents, build_graph

log = logging.getLogger("api")

FRONTEND_DIST = ROOT / "chat" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    gateway, checkpointer = create_gateway(settings), MemorySaver()
    retriever = create_retriever(settings)
    market_service = create_market_service(settings)

    app.state.settings = settings
    app.state.started_at = time.time()
    app.state.llm = app.state.gateway = gateway
    app.state.retriever = retriever
    app.state.market = market_service
    app.state.checkpointer = checkpointer
    app.state.llm_limiter = ClientRateLimiter(settings.chat_requests_per_min)
    app.state.api_limiter = ClientRateLimiter(settings.api_requests_per_min)
    app.state.agents = build_agents(gateway, retriever, market_service)
    app.state.graph = build_graph(
        gateway, retriever, checkpointer, market=market_service,
        max_history=settings.max_history_messages, max_chars=settings.max_message_chars,
        llm_routing=settings.router_llm_fallback and settings.llm_provider != "fake",
    )
    log.info("started with llm=%s models=%s market=%s rag=%s", settings.llm_provider,
             [m.name for m in gateway.models], market_service.source, type(retriever).__name__)
    yield


app = FastAPI(
    title="AI Finance Assistant",
    version="1.0.0",
    description="Multi-agent financial education assistant: LangGraph workflow, RAG, live market data.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def _setup_cors() -> None:
    origins = Settings.load().cors_origins
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(origins), allow_methods=["*"], allow_headers=["*"])


_setup_cors()


# ---------- Errors: always {"error_code", "message"} so the UI can show them ----------

_STATUS_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 429: "rate_limited",
                 500: "internal_error", 503: "service_unavailable"}


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message, **extra})


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, _STATUS_CODES.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(p) for p in first.get("loc", [])[1:])
    message = f"{field}: {first.get('msg')}" if field else str(first.get("msg", "Invalid request."))
    details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in errors]
    return _error(422, "validation_error", message, details=details)


@app.exception_handler(InvalidRequest)
async def invalid_request(request: Request, exc: InvalidRequest):
    return _error(400, "invalid_request", str(exc))


@app.exception_handler(SymbolNotFound)
async def symbol_not_found(request: Request, exc: SymbolNotFound):
    return _error(404, "symbol_not_found", str(exc))


@app.exception_handler(MarketDataUnavailable)
async def market_unavailable(request: Request, exc: MarketDataUnavailable):
    return _error(503, "market_data_unavailable", str(exc))


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "Something went wrong on our side.")


# ---------- Routes ----------

@app.get("/api/health")
async def health(request: Request):
    state = request.app.state
    return {
        "status": "ok",
        "llm_provider": state.settings.llm_provider,
        "models": [m.name for m in state.gateway.models],
        "market_data_provider": state.market.source,
        "rag": type(state.retriever).__name__,
        "uptime_seconds": round(time.time() - state.started_at),
    }


for module in (chat, portfolio, market, goals, sessions, knowledge, system):
    app.include_router(module.router)


# ---------- Built frontend (production / Docker): one process serves API and UI ----------

if (FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith("api/"):
            return _error(404, "not_found", "Not Found")
        candidate = (FRONTEND_DIST / path).resolve()
        if path and candidate.is_file() and Path(FRONTEND_DIST).resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")

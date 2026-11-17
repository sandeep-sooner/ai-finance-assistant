from fastapi import APIRouter, Depends, Request

from src.core.schemas import ChatRequest, ChatResponse, Source
from src.core.sse import sse
from src.web_app.api.deps import llm_rate_limit, log, sse_response

router = APIRouter(prefix="/api", tags=["chat"])

SAFE_REPLY = (
    "Sorry, something went wrong on my side while preparing that answer, so I stopped rather than guess. "
    "Your message is saved in this conversation. Please send it again in a moment."
)


async def _events(graph, req: ChatRequest):
    """Yield (event, data) pairs from the workflow. A crash inside the graph becomes a normal,
    polite assistant message: the user never sees a stack trace or a raw error."""
    config = {
        "configurable": {"thread_id": req.session_id},
        # Shown in LangSmith: the run name, and session_id groups a conversation's turns into one thread.
        "run_name": "finance_chat",
        "tags": ["chat", f"level:{req.level}"],
        "metadata": {"session_id": req.session_id, "level": req.level},
    }
    try:
        async for item in graph.astream(
            {"messages": [("user", req.message)], "level": req.level}, config, stream_mode="custom"
        ):
            yield item["event"], item["data"]
    except Exception:
        log.exception("chat failed session=%s", req.session_id)
        yield "agent", "assistant"
        yield "token", SAFE_REPLY
        yield "sources", []
        yield "meta", {"agents": ["assistant"], "calls": [], "disclaimer": "", "flags": ["workflow_error"],
                       "route": {"plan": [], "method": "error"}}


@router.post("/chat/stream", dependencies=[Depends(llm_rate_limit)])
async def chat_stream(req: ChatRequest, request: Request):
    """Server-sent events: `agent`, `token`*, `sources`, `meta`, `done`."""
    graph = request.app.state.graph

    async def gen():
        async for event, data in _events(graph, req):
            yield sse(event, data)
        yield sse("done", {})

    return sse_response(gen())


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(llm_rate_limit)])
async def chat(req: ChatRequest, request: Request):
    """Same workflow as /chat/stream, returned as one JSON object."""
    agent, tokens, sources, meta = "", [], [], {}
    async for event, data in _events(request.app.state.graph, req):
        if event == "agent":
            agent = ",".join(data) if isinstance(data, list) else data
        elif event == "token":
            tokens.append(data)
        elif event == "sources":
            sources = data
        elif event == "meta":
            meta = data
    return ChatResponse(answer="".join(tokens).strip(), agent_used=agent,
                        sources=[Source(**s) for s in sources], meta=meta)

"""LangSmith helpers: off by default, never raise, and attach our labels when a run is being traced."""
import asyncio
from unittest.mock import MagicMock

from langchain_core.tracers import LangChainTracer
from langgraph.checkpoint.memory import MemorySaver

from evals.suites import EchoLLM, local_retriever
from src.core import tracing
from src.market.providers import SampleProvider
from src.market.service import MarketDataService
from src.workflow.graph import build_graph


def test_tracing_is_off_by_default_and_annotate_is_a_no_op(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert tracing.tracing_enabled() is False
    assert tracing.annotate({"a": 1}, ["t"]) is False
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing.tracing_enabled() is True
    assert tracing.annotate({"a": 1}) is False            # enabled, but nothing is being traced right now


def test_annotate_swallows_errors(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracing, "get_current_run_tree", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert tracing.annotate({"a": 1}) is False


def test_trace_tags_and_document_shape():
    assert tracing.trace_tags({"tier": "fast"}) == ["tier:fast"]
    assert tracing.trace_tags({"tier": "smart", "cache_hit": True, "retries": 1}) == ["tier:smart", "cache_hit", "retried"]
    assert "offline_fallback" in tracing.trace_tags({"tier": "fast", "degraded": True, "fallbacks": ["a"]})
    assert "model_switched" in tracing.trace_tags({"tier": "fast", "fallbacks": ["a"]})
    out = tracing.documents_output([{"title": "T", "passage": "short", "text": "long", "score": 0.5, "url": None}])
    assert out["documents"][0]["page_content"] == "short" and out["documents"][0]["metadata"]["title"] == "T"


def test_workflow_runs_carry_route_and_model_labels(monkeypatch):
    """Record runs with an in-memory tracer (nothing is sent anywhere) and check what was attached."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    ended = []

    class Recorder(LangChainTracer):
        def _update_run_single(self, run):
            ended.append(run)

    tracer = Recorder(client=MagicMock())
    graph = build_graph(EchoLLM(), local_retriever(), MemorySaver(), market=MarketDataService(SampleProvider()))

    async def go():
        cfg = {"configurable": {"thread_id": "trace-0001"}, "callbacks": [tracer]}
        question = "Analyze my portfolio: 10 AAPL, 20 BND and how are capital gains taxed if I sell?"
        return [e async for e in graph.astream({"messages": [("user", question)]}, cfg, stream_mode="custom")]

    asyncio.run(go())
    runs = {r.name: r for r in ended}
    router_meta = runs["router"].extra["metadata"]
    assert router_meta["route_plan"] == ["portfolio", "tax"] and "route:rules" in runs["router"].tags
    tax = runs["tax"]
    assert tax.extra["metadata"]["model_tier"] in {"fast", "smart"} and tax.extra["metadata"]["cache_hit"] is False
    assert any(t.startswith("tier:") for t in tax.tags)
    spans = [c.kwargs for c in tracer.client.create_run.call_args_list if c.kwargs.get("run_type") == "retriever"]
    assert spans and spans[0]["name"] == "knowledge_base_search" and "self" not in spans[0]["inputs"]

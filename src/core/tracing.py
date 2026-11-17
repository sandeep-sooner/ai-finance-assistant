"""Optional LangSmith tracing helpers.

LangGraph and the LangChain chat models trace themselves as soon as LANGSMITH_TRACING=true and
LANGSMITH_API_KEY are set; no code is needed for that. This module adds what LangSmith cannot
see on its own:

  annotate()   - attach our own facts (model tier, cache hit, fallbacks, route) to the current run,
                 so traces can be filtered by them in the LangSmith UI;
  traceable()  - record a plain Python function (the retriever) as its own span.

Both are no-ops when tracing is off or the langsmith package is missing, and neither can raise:
observability must never break a chat.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

log = logging.getLogger(__name__)

try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree

    HAVE_LANGSMITH = True
except ImportError:  # pragma: no cover - langsmith ships with langchain-core
    HAVE_LANGSMITH = False


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return HAVE_LANGSMITH and flag.strip().lower() in {"1", "true", "yes", "on"}


def traceable(**kwargs) -> Callable:
    """langsmith.traceable when available, otherwise a decorator that changes nothing."""
    if not HAVE_LANGSMITH:  # pragma: no cover
        return lambda fn: fn
    return _traceable(**kwargs)


def annotate(metadata: dict[str, Any] | None = None, tags: list[str] | None = None) -> bool:
    """Add metadata and tags to the run that is currently being traced. Returns True if attached."""
    if not tracing_enabled():
        return False
    try:
        run = get_current_run_tree()
        if run is None:
            return False
        if metadata:
            run.add_metadata(metadata)
        if tags:
            run.add_tags(tags)
        return True
    except Exception as exc:
        log.debug("could not annotate trace: %s", exc)
        return False


def trace_tags(trace: dict) -> list[str]:
    """Short, filterable labels for one LLM gateway call."""
    tags = [f"tier:{trace.get('tier') or 'none'}"]
    if trace.get("cache_hit"):
        tags.append("cache_hit")
    if trace.get("degraded"):
        tags.append("offline_fallback")
    elif trace.get("fallbacks") or trace.get("continued"):
        tags.append("model_switched")
    if trace.get("retries"):
        tags.append("retried")
    return tags


def documents_output(results: list[dict]) -> dict:
    """Shape retriever results the way LangSmith renders documents (text plus metadata)."""
    return {"documents": [
        {"page_content": r.get("passage") or r.get("text", ""), "type": "Document",
         "metadata": {k: r.get(k) for k in ("title", "category", "source", "url", "score", "article_id")}}
        for r in results
    ]}

from functools import lru_cache

from fastapi import APIRouter, Depends, Query, Request

from src.rag.knowledge_base import load_articles, load_glossary
from src.web_app.api.deps import api_rate_limit

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"], dependencies=[Depends(api_rate_limit)])


@lru_cache(maxsize=1)
def _articles() -> list[dict]:
    return [{"id": a.id, "title": a.title, "category": a.category, "source": a.source, "url": a.url}
            for a in load_articles()]


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(min_length=2, max_length=300),
    category: str | None = Query(default=None, max_length=30),
    k: int = Query(default=4, ge=1, le=10),
):
    """Hybrid (vector + keyword) search over the knowledge base, with relevance scores."""
    results = request.app.state.retriever.search(q, category, k) if hasattr(request.app.state.retriever, "stats") \
        else request.app.state.retriever.search(q, category)
    return {"query": q, "category": category, "results": results}


@router.get("/articles")
async def articles(category: str | None = Query(default=None, max_length=30)):
    """The article catalogue (titles and sources), optionally filtered by category."""
    rows = [a for a in _articles() if category is None or a["category"] == category]
    return {"count": len(rows), "categories": sorted({a["category"] for a in _articles()}), "articles": rows}


@router.get("/glossary")
async def glossary(q: str | None = Query(default=None, max_length=60)):
    terms = load_glossary()
    if q:
        needle = q.lower()
        terms = [t for t in terms if needle in t["term"].lower() or needle in t["definition"].lower()]
    return {"count": len(terms), "terms": terms}

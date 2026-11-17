"""Retrieval pipeline.

    query -> (cache) -> embed -> FAISS cosine  ┐
                     -> BM25 keyword score     ┴-> hybrid relevance score
          -> category filter -> relevance floor -> best chunk per article -> top k
          -> return the *whole* article for each hit (small-to-big retrieval)

Matching is done on small chunks because a focused chunk embeds more precisely than a
whole article; the LLM is then given the full parent article so it has the surrounding
context, and each article appears once in the citations.

Each result carries title, source, url and score, which the agents pass straight
through to the UI as citations.

Failure handling: if the embedding API is down when the index is built, the
retriever falls back to the local embedder; if it is down at query time, that
query is answered by BM25 alone. Retrieval never raises into a chat request.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from src.core.cache import TTLCache, normalize
from src.core.tracing import documents_output, traceable
from src.rag.embeddings import Embedder, LocalEmbedder, OpenAIEmbedder, tokenize
from src.rag.knowledge_base import Chunk, build_chunks, corpus_fingerprint, load_articles
from src.rag.vector_store import VectorStore

log = logging.getLogger(__name__)


class Retriever(Protocol):
    def search(self, query: str, category: str | None = None) -> list[dict]:
        """Return chunks as {title, url, category, text, score}, best first."""
        ...


class NullRetriever:
    def search(self, query: str, category: str | None = None) -> list[dict]:
        return []


class BM25:
    """Okapi BM25 over the chunk texts (titles are counted twice: a title match is a strong signal)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean()) if docs else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype="float32")
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


class KnowledgeBaseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        top_k: int = 4,
        dense_weight: float = 0.6,
        min_score: float = 0.22,
        relative_cut: float = 0.6,
        index_dir: str | Path | None = None,
        cache_items: int = 500,
        fallback_embedder: Embedder | None = None,
        parents: dict[str, str] | None = None,
    ):
        if not chunks:
            raise ValueError("the knowledge base is empty")
        self.chunks, self.top_k, self.dense_weight, self.min_score = chunks, top_k, dense_weight, min_score
        self.embedder, self.relative_cut = embedder, relative_cut
        self.parents = parents or {}  # article id -> full article text
        self._cache = TTLCache(cache_items)
        self.cache_hits = self.cache_misses = self.dense_failures = 0
        self._bm25 = BM25([tokenize(f"{c.title} {c.title} {c.text}") for c in chunks])
        self.categories = sorted({c.category for c in chunks})

        texts = [f"{c.title}\n{c.text}" for c in chunks]
        try:
            self.store = self._build(texts, embedder, index_dir)
        except Exception as exc:
            if fallback_embedder is None:
                raise
            log.warning("embedder %s failed (%s); falling back to %s", embedder.name, exc, fallback_embedder.name)
            self.embedder = fallback_embedder
            self.store = self._build(texts, fallback_embedder, index_dir)

    def _build(self, texts: list[str], embedder: Embedder, index_dir: str | Path | None) -> VectorStore:
        embedder.fit(texts)
        meta = {"fingerprint": corpus_fingerprint(self.chunks), "embedder": embedder.name, "chunks": len(texts)}
        directory = Path(index_dir) / embedder.name if index_dir else None
        if directory is not None:
            store = VectorStore.load(directory, meta)
            if store is not None and len(store) == len(texts):
                log.info("loaded RAG index from %s (%d chunks, %s)", directory, len(store), store.backend)
                return store
        store = VectorStore(embedder.embed(texts))
        if directory is not None:
            try:
                store.save(directory, meta)
                getattr(embedder, "save", lambda: None)()
            except OSError as exc:  # a read-only disk should not stop the app
                log.warning("could not persist RAG index: %s", exc)
        log.info("built RAG index: %d chunks, embedder=%s, backend=%s", len(store), embedder.name, store.backend)
        return store

    # ---------- search ----------

    @traceable(run_type="retriever", name="knowledge_base_search",
               process_inputs=lambda i: {k: v for k, v in i.items() if k != "self"},
               process_outputs=documents_output)
    def search(self, query: str, category: str | None = None, k: int | None = None) -> list[dict]:
        k = k or self.top_k
        key = (normalize(query), category, k)
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return [dict(r) for r in hit[0]]
        self.cache_misses += 1

        results = self._search(query, category, k)
        if category and not results:  # nothing relevant inside the category: widen to the whole base
            results = self._search(query, None, k)
        self._cache.set(key, results)
        return [dict(r) for r in results]

    def _search(self, query: str, category: str | None, k: int) -> list[dict]:
        tokens = tokenize(query)
        if not tokens:
            return []
        lexical = self._bm25.scores(tokens)
        # Squash BM25 into 0..1. A chunk matching most query terms well scores close to 1.
        lexical = lexical / (lexical + 6.0)
        try:
            dense = np.clip(self.store.scores(self.embedder.embed([query])[0]), 0.0, 1.0)
            score = self.dense_weight * dense + (1 - self.dense_weight) * lexical
        except Exception as exc:
            self.dense_failures += 1
            log.warning("query embedding failed (%s); using keyword search only", exc)
            score = lexical

        order = np.argsort(-score)
        # Absolute floor, plus a relative one: a chunk far weaker than the best match is noise.
        floor = max(self.min_score, self.relative_cut * float(score[order[0]]))
        results: list[dict] = []
        per_article: Counter[str] = Counter()
        for i in order:
            s = float(score[i])
            if s < floor or len(results) >= k:
                break
            c = self.chunks[int(i)]
            if category and c.category != category:
                continue
            if per_article[c.article_id] >= 1:
                continue
            per_article[c.article_id] += 1
            results.append({
                "id": c.id, "article_id": c.article_id, "title": c.title, "category": c.category,
                "source": c.source, "url": c.url, "score": round(s, 4),
                "text": self.parents.get(c.article_id, c.text), "passage": c.text,
            })
        return results

    def stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        return {
            "chunks": len(self.chunks), "articles": len({c.article_id for c in self.chunks}),
            "categories": self.categories, "embedder": self.embedder.name, "backend": self.store.backend,
            "query_cache_hits": self.cache_hits, "query_cache_misses": self.cache_misses,
            "query_cache_hit_rate": round(self.cache_hits / total, 4) if total else 0.0,
            "dense_failures": self.dense_failures,
        }


def create_retriever(settings, chunks: list[Chunk] | None = None) -> Retriever:
    """Build the retriever from settings. Any failure degrades to NullRetriever (answers without citations)."""
    try:
        articles = load_articles()
        chunks = chunks or build_chunks(articles, max_chars=settings.rag_chunk_chars, overlap=settings.rag_chunk_overlap)
        local = LocalEmbedder()
        mode = settings.rag_embedder
        use_openai = mode == "openai" or (mode == "auto" and settings.llm_provider == "openai")
        embedder: Embedder = local
        if use_openai and settings.openai_api_key:
            embedder = OpenAIEmbedder(settings.openai_api_key, settings.rag_embedding_model,
                                      settings.openai_base_url, cache_dir=settings.rag_index_dir)
        return KnowledgeBaseRetriever(
            chunks, embedder, top_k=settings.rag_top_k, dense_weight=settings.rag_dense_weight,
            min_score=settings.rag_min_score, relative_cut=settings.rag_relative_cut,
            index_dir=settings.rag_index_dir,
            cache_items=settings.retrieval_cache_max_items,
            fallback_embedder=local if embedder is not local else None,
            parents={a.id: a.text for a in articles},
        )
    except Exception:
        log.exception("could not build the knowledge base retriever; continuing without RAG")
        return NullRetriever()

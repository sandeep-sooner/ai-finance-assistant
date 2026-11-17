"""Knowledge base loading, chunking, embeddings, vector store and the hybrid retriever."""
import numpy as np
import pytest

from src.rag.embeddings import LocalEmbedder, OpenAIEmbedder, tokenize
from src.rag.knowledge_base import (
    Chunk, build_chunks, corpus_fingerprint, load_articles, load_glossary, parse_articles, split_text,
)
from src.rag.retriever import BM25, KnowledgeBaseRetriever, NullRetriever, create_retriever
from src.rag.vector_store import VectorStore


@pytest.fixture(scope="module")
def articles():
    return load_articles()


@pytest.fixture(scope="module")
def retriever(articles):
    return KnowledgeBaseRetriever(build_chunks(articles), LocalEmbedder(), parents={a.id: a.text for a in articles})


# ---------- knowledge base ----------

def test_knowledge_base_meets_the_brief(articles):
    assert 50 <= len(articles) <= 100
    assert {a.category for a in articles} == {"basics", "stocks", "bonds", "funds", "portfolio", "planning", "tax", "markets"}
    assert len({a.id for a in articles}) == len(articles)                       # ids are unique
    assert all(a.url and a.url.startswith("https://") and a.source != "Knowledge base" for a in articles)
    assert all(len(a.text) > 300 for a in articles)
    assert len(load_glossary()) >= 50


def test_parse_articles_reads_title_metadata_and_body():
    md = "# First One\nsource: SEC\nurl: https://x.test/a\n\nBody line.\n\nMore.\n\n# Second\n\nurl: not metadata here\n"
    first, second = parse_articles(md, "basics")
    assert (first.id, first.source, first.url, first.text) == ("basics/first-one", "SEC", "https://x.test/a", "Body line.\n\nMore.")
    assert second.source == "Knowledge base" and second.url == "not metadata here" or second.text


def test_split_text_respects_size_and_overlaps():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
    chunks = split_text(text, max_chars=500, overlap=80)
    assert len(chunks) > 2 and all(len(c) <= 500 + 80 + 2 for c in chunks)
    assert chunks[1].split("\n\n")[0] in chunks[0]          # the overlap really is the previous tail
    assert split_text("short", 500, 80) == ["short"]
    long_para = "Sentence one is here. " * 60
    assert all(len(c) <= 500 for c in split_text(long_para, 500, 0))


def test_build_chunks_includes_glossary_and_fingerprint_tracks_content(articles):
    chunks = build_chunks(articles)
    assert len(chunks) > len(articles)
    assert any(c.article_id == "glossary/etf" for c in chunks)
    assert corpus_fingerprint(chunks) == corpus_fingerprint(build_chunks(articles))
    edited = [Chunk(**{**chunks[0].as_dict(), "text": "changed"})] + chunks[1:]
    assert corpus_fingerprint(edited) != corpus_fingerprint(chunks)


# ---------- embeddings and store ----------

def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("What are ETFs and bonds?") == ["etf", "bond"]
    assert tokenize("emergencies, taxes; investing") == ["emergency", "tax", "invest"]
    assert tokenize("hi thanks") == []


def test_local_embedder_is_normalised_and_meaningful():
    e = LocalEmbedder(dim=512)
    docs = ["bond prices fall when interest rates rise", "an ETF is a basket of stocks", "emergency fund savings"]
    e.fit(docs)
    m = e.embed(docs + ["why do bonds drop if rates go up"])
    assert m.shape == (4, 512) and np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    sims = m[:3] @ m[3]
    assert sims.argmax() == 0
    assert not e.embed(["hi"]).any()  # nothing but stopwords -> zero vector, not NaN


def test_openai_embedder_batches_and_caches_to_disk(tmp_path):
    class FakeClient:
        def __init__(self):
            self.calls = []
            self.embeddings = self

        def create(self, model, input):
            self.calls.append(list(input))
            return type("R", (), {"data": [type("D", (), {"embedding": [float(len(t)), 1.0, 0.0]}) for t in input]})

    client = FakeClient()
    e = OpenAIEmbedder("k", cache_dir=tmp_path, client=client, batch_size=2)
    out = e.embed(["a", "bb", "ccc", "a"])
    assert out.shape == (4, 3) and len(client.calls) == 2 and e.dim == 3      # 3 unique texts, batches of 2
    e.embed(["a", "bb"])
    assert len(client.calls) == 2                                           # served from memory
    e.save()
    again = OpenAIEmbedder("k", cache_dir=tmp_path, client=FakeClient())
    again.embed(["ccc"])
    assert again._client.calls == []                                        # served from disk


@pytest.mark.parametrize("use_faiss", [True, False])
def test_vector_store_search_and_persistence(tmp_path, use_faiss):
    vecs = np.eye(4, dtype="float32")
    store = VectorStore(vecs, use_faiss=use_faiss)
    scores, idx = store.search(np.array([0, 1, 0, 0], dtype="float32"), k=2)
    assert idx[0] == 1 and scores[0] == pytest.approx(1.0) and len(store) == 4
    assert store.scores(vecs[2]).argmax() == 2
    meta = {"fingerprint": "abc", "embedder": "x"}
    store.save(tmp_path, meta)
    assert len(VectorStore.load(tmp_path, meta)) == 4
    assert VectorStore.load(tmp_path, {"fingerprint": "different"}) is None   # stale index is ignored
    assert VectorStore.load(tmp_path / "missing", meta) is None


# ---------- retriever ----------

def test_bm25_ranks_the_matching_document_first():
    bm = BM25([["bond", "rate"], ["etf", "basket", "stock"], ["tax", "gain"]])
    assert bm.scores(["etf"]).argmax() == 1 and bm.scores(["unknownterm"]).sum() == 0


@pytest.mark.parametrize("query,expected", [
    ("What is an ETF?", {"funds/what-is-an-etf", "glossary/etf"}),
    ("why do bond prices fall when rates rise", {"bonds/bond-prices-and-interest-rates"}),
    ("how are dividends taxed", {"tax/how-dividends-and-interest-are-taxed"}),
])
def test_retriever_finds_the_right_article(retriever, query, expected):
    results = retriever.search(query)
    assert results[0]["article_id"] in expected
    assert {"title", "source", "url", "score", "text", "passage", "category"} <= set(results[0])
    assert results == sorted(results, key=lambda r: -r["score"])


def test_retriever_rejects_off_topic_and_empty_queries(retriever):
    assert retriever.search("write me a poem about cats") == []
    assert retriever.search("hi") == [] and retriever.search("") == []


def test_retriever_category_filter_and_widening(retriever):
    tax = retriever.search("roth ira", category="tax")
    assert tax and all(r["category"] == "tax" for r in tax)
    assert retriever.search("what is an etf", category="tax")[0]["category"] != "tax"  # nothing in tax: widened


def test_retriever_returns_whole_article_once_and_caches(retriever, articles):
    results = retriever.search("difference between etf and mutual fund", k=5)
    ids = [r["article_id"] for r in results]
    assert len(ids) == len(set(ids))                                   # one hit per article
    full = next(a.text for a in articles if a.id == results[0]["article_id"])
    assert results[0]["text"] == full and results[0]["passage"] in full  # small-to-big
    hits = retriever.cache_hits
    assert retriever.search("Difference between ETF and mutual fund?", k=5) == results
    assert retriever.cache_hits == hits + 1
    assert retriever.stats()["chunks"] == len(retriever.chunks) and retriever.stats()["backend"] in {"faiss", "numpy"}


def test_retriever_falls_back_when_the_embedder_is_down(articles):
    class Broken:
        name, dim = "broken", 8

        def fit(self, texts): ...
        def embed(self, texts):
            raise RuntimeError("embedding API down")

    chunks = build_chunks(articles)
    r = KnowledgeBaseRetriever(chunks, Broken(), fallback_embedder=LocalEmbedder())
    assert r.embedder.name == "local-hashed-tfidf" and r.search("what is an etf")
    with pytest.raises(RuntimeError):
        KnowledgeBaseRetriever(chunks, Broken())

    # Embedder dies after the index is built: that query is answered by keyword search alone.
    live = KnowledgeBaseRetriever(chunks, LocalEmbedder())
    live.embedder = Broken()
    assert live.search("what is an etf")[0]["article_id"] in {"funds/what-is-an-etf", "glossary/etf"}
    assert live.dense_failures == 1


def test_index_is_persisted_and_reused(tmp_path, articles):
    chunks = build_chunks(articles)
    KnowledgeBaseRetriever(chunks, LocalEmbedder(), index_dir=tmp_path)
    assert (tmp_path / "local-hashed-tfidf" / "meta.json").exists()

    class CountingEmbedder(LocalEmbedder):
        embedded = 0

        def embed(self, texts):
            CountingEmbedder.embedded += len(texts)
            return super().embed(texts)

    KnowledgeBaseRetriever(chunks, CountingEmbedder(), index_dir=tmp_path)
    assert CountingEmbedder.embedded == 0   # vectors came from disk


def test_create_retriever_degrades_to_null(monkeypatch):
    from src.core.config import Settings

    monkeypatch.setenv("RAG_EMBEDDER", "local")
    monkeypatch.setenv("RAG_INDEX_DIR", "")
    assert isinstance(create_retriever(Settings.load()), KnowledgeBaseRetriever)
    monkeypatch.setattr("src.rag.retriever.load_articles", lambda: (_ for _ in ()).throw(OSError("disk")))
    assert isinstance(create_retriever(Settings.load()), NullRetriever)
    assert NullRetriever().search("anything") == []
    with pytest.raises(ValueError):
        KnowledgeBaseRetriever([], LocalEmbedder())

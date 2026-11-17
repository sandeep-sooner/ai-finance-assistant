"""Embedders. All return L2-normalised float32 vectors, so inner product = cosine similarity.

LocalEmbedder  - no network, no model download: hashed word + bigram TF-IDF. Deterministic,
                 so tests and offline demos behave identically everywhere.
OpenAIEmbedder - text-embedding-3-small via the API, with an on-disk cache so each text is
                 only ever embedded (and paid for) once.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:\([a-z]\))?")
STOPWORDS = frozenset(
    "a about above after again all also am an and any are as at be because been before being below between both "
    "but by can could did do does doing down during each few for from further had has have having he her here "
    "hers him his how i if in into is it its just me more most my no nor not now of off on once only or other "
    "our out over own same she should so some such than that the their them then there these they this those "
    "through to too under until up very was we were what when where which while who whom why will with would "
    "you your hi hello hey thanks thank please tell give know want need like get make much many mean means "
    "work works explain good best really actually okay ok yes".split()
)
# Finance words whose variants should match each other (a tiny domain-specific stemmer).
_SYNONYMS = {"equity": "stock", "equities": "stock", "shares": "stock", "share": "stock", "taxed": "tax",
             "taxation": "tax", "taxes": "tax", "investing": "invest", "investment": "invest",
             "investments": "invest", "investor": "invest", "investors": "invest", "invested": "invest",
             "retire": "retirement", "retiring": "retirement", "retired": "retirement",
             "diversify": "diversification", "diversified": "diversification", "diversifying": "diversification",
             "rebalance": "rebalancing", "savings": "saving", "save": "saving", "saved": "saving",
             "indexes": "index", "indices": "index", "risky": "risk", "riskier": "risk", "compounding": "compound",
             "compounded": "compound", "crypto": "cryptocurrency", "cryptocurrencies": "cryptocurrency",
             "bitcoin": "cryptocurrency", "safe": "risk", "safer": "risk", "safest": "risk", "safety": "risk",
             "volatile": "volatility", "cheaper": "cheap", "pays": "pay", "paying": "pay", "paid": "pay"}


def _stem(tok: str) -> str:
    """Plural -> singular for ordinary words: 'emergencies' -> 'emergency', 'bonds' -> 'bond'."""
    if tok in _SYNONYMS:
        return _SYNONYMS[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Lower-case words with light stemming; stopwords and one-letter tokens removed."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tok = _stem(tok)
        if len(tok) > 1 and tok not in STOPWORDS:
            out.append(tok)
    return out


class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class LocalEmbedder:
    name = "local-hashed-tfidf"

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self._idf: dict[int, float] = {}
        self._default_idf = 1.0

    def _features(self, text: str) -> list[int]:
        toks = tokenize(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return [int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % self.dim for g in grams]

    def fit(self, texts: list[str]) -> None:
        df: dict[int, int] = {}
        for text in texts:
            for f in set(self._features(text)):
                df[f] = df.get(f, 0) + 1
        n = max(len(texts), 1)
        self._idf = {f: math.log((1 + n) / (1 + c)) + 1.0 for f, c in df.items()}
        self._default_idf = math.log(1 + n) + 1.0  # unseen terms are as informative as the rarest

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for f in self._features(text):
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                out[row, f] = (1.0 + math.log(c)) * self._idf.get(f, self._default_idf)
        return _normalize(out)


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None,
                 cache_dir: str | Path | None = None, client=None, batch_size: int = 128):
        self.name, self.dim = f"openai-{model}", 1536
        self._model, self._batch = model, batch_size
        if client is None:
            from openai import OpenAI  # imported lazily

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0, max_retries=2)
        self._client = client
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path = Path(cache_dir) / f"embeddings-{model}.npz" if cache_dir else None
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._cache_path and self._cache_path.exists():
            try:
                with np.load(self._cache_path) as data:
                    self._cache = {k: data[k] for k in data.files}
            except Exception as exc:  # a corrupt cache is not worth failing over
                log.warning("ignoring unreadable embedding cache: %s", exc)

    def save(self) -> None:
        if self._cache_path and self._dirty:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._cache_path, **self._cache)
            self._dirty = False

    def fit(self, texts: list[str]) -> None:  # nothing to learn
        return None

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._cache]
        for i in range(0, len(missing), self._batch):
            batch = missing[i:i + self._batch]
            resp = self._client.embeddings.create(model=self._model, input=[t for _, t in batch])
            for (k, _), item in zip(batch, resp.data):
                self._cache[k] = np.asarray(item.embedding, dtype="float32")
            self._dirty = True
        matrix = np.vstack([self._cache[k] for k in keys])
        self.dim = matrix.shape[1]
        return _normalize(matrix)

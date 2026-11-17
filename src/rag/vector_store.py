"""Vector store: FAISS inner-product index over normalised embeddings, persisted to disk.

If faiss is not installed the same interface is served by a NumPy matrix product,
which is exact and fast enough for a few hundred chunks.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    HAVE_FAISS = True
except ImportError:  # pragma: no cover - exercised only without faiss installed
    faiss = None
    HAVE_FAISS = False


class VectorStore:
    def __init__(self, vectors: np.ndarray, use_faiss: bool = True):
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.backend = "faiss" if (use_faiss and HAVE_FAISS) else "numpy"
        self._index = None
        if self.backend == "faiss":
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) of the k most similar vectors, best first."""
        k = min(k, len(self))
        query = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        if self._index is not None:
            scores, idx = self._index.search(query, k)
            return scores[0], idx[0]
        sims = self.vectors @ query[0]
        order = np.argsort(-sims)[:k]
        return sims[order], order

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Similarity of the query to every vector (used for hybrid scoring)."""
        return self.vectors @ np.asarray(query, dtype="float32").reshape(-1)

    # ---------- persistence ----------

    def save(self, directory: Path, meta: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / "index.faiss"))
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, expected_meta: dict) -> "VectorStore | None":
        """Load a saved index only if it was built from the same corpus with the same embedder."""
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if any(meta.get(k) != v for k, v in expected_meta.items()):
                return None
            return cls(np.load(directory / "vectors.npy"))
        except (OSError, ValueError) as exc:
            log.debug("no reusable index in %s: %s", directory, exc)
            return None

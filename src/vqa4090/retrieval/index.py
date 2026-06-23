from __future__ import annotations

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


class DenseIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.emb: np.ndarray | None = None
        self.use_faiss = faiss is not None
        self.index = faiss.IndexFlatIP(dim) if self.use_faiss else None

    def add(self, vectors: np.ndarray) -> None:
        if self.use_faiss and self.index is not None:
            self.index.add(vectors)
        else:
            self.emb = vectors

    def search(self, q: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.use_faiss and self.index is not None:
            scores, ids = self.index.search(q, top_k)
            return scores, ids

        if self.emb is None:
            raise RuntimeError("Index has no vectors.")

        sims = q @ self.emb.T
        ids = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, ids, axis=1)
        return scores, ids

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # coseno si vectores normalizados

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        scores, ids = self.index.search(query_vec, top_k)
        return scores[0], ids[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path: Path) -> faiss.Index:
        return faiss.read_index(str(path))

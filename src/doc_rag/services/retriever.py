from __future__ import annotations

import json
from typing import Any

import faiss

from doc_rag.core.settings import Settings
from doc_rag.services.embedding import Embedder


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model)
        self.index_path = settings.index_dir / "global.faiss"
        self.chunks_path = settings.index_dir / "chunks.jsonl"

        self._index: faiss.Index | None = None
        self._chunks_by_id: dict[int, dict[str, Any]] = {}

    def load(self) -> None:
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError("Índice no encontrado. Ejecute /documents/reindex primero.")

        self._index = faiss.read_index(str(self.index_path))
        self._chunks_by_id.clear()

        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self._chunks_by_id[int(rec["id"])] = rec

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if self._index is None or not self._chunks_by_id:
            self.load()

        qvec = self.embedder.encode([question])
        scores, ids = self._index.search(qvec, top_k)  # type: ignore[union-attr]

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            rec = self._chunks_by_id.get(int(idx))
            if not rec:
                continue
            results.append({**rec, "score": float(score)})
        return results

from __future__ import annotations

import json
from typing import Any

import faiss

from doc_rag.core.settings import Settings
from doc_rag.services.embedding import Embedder
from doc_rag.services.reranker import Reranker


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model)
        self.index_path = settings.index_dir / "global.faiss"
        self.chunks_path = settings.index_dir / "chunks.jsonl"
        self._reranker: Reranker | None = None

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

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(
                model_name=self.settings.rerank_model,
                device=self.settings.rerank_device,
            )
        return self._reranker

    def search(
        self, question: str, top_k: int, use_rerank: bool | None = None
    ) -> list[dict[str, Any]]:
        if self._index is None or not self._chunks_by_id:
            self.load()

        use_rerank_final = use_rerank if use_rerank is not None else self.settings.use_rerank

        # 1) Recuperación densa (candidatos)
        candidates_k = max(self.settings.retrieve_candidates, top_k * 8)
        qvec = self.embedder.encode([question])
        scores, ids = self._index.search(qvec, candidates_k)  # type: ignore[union-attr]

        candidates: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            rec = self._chunks_by_id.get(int(idx))
            if not rec:
                continue
            candidates.append({**rec, "score_dense": float(score)})

        if not candidates:
            return []

        # 2) Re-rank (CrossEncoder) y selección final
        if use_rerank_final:
            reranker = self._get_reranker()
            passage_texts = [c["text"] for c in candidates]
            rr_scores = reranker.score(question, passage_texts)

            for c, rr in zip(candidates, rr_scores, strict=False):
                c["score_rerank"] = rr
                c["score"] = rr  # score final

            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            for c in candidates:
                c["score"] = c["score_dense"]
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicación mínima por (fichero+ancla)
        seen = set()
        final: list[dict[str, Any]] = []
        for c in candidates:
            key = (c["source_filename"], c["anchor"])
            if key in seen:
                continue
            seen.add(key)
            final.append(c)
            if len(final) >= top_k:
                break

        return final

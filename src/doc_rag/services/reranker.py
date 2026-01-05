from __future__ import annotations

from sentence_transformers import CrossEncoder  # :contentReference[oaicite:2]{index=2}


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

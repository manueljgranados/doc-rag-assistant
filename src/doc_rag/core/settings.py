from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platform


@dataclass(frozen=True)
class Settings:
    # Rutas
    data_dir: Path = Path(os.getenv("DOC_RAG_DATA_DIR", "data"))
    uploads_dir: Path = Path(os.getenv("DOC_RAG_UPLOADS_DIR", "data/uploads"))
    index_dir: Path = Path(os.getenv("DOC_RAG_INDEX_DIR", "data/index"))

    # Límites
    max_upload_mb: int = int(os.getenv("DOC_RAG_MAX_UPLOAD_MB", "20"))

    # RAG
    embedding_model: str = os.getenv(
        "DOC_RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    chunk_size: int = int(os.getenv("DOC_RAG_CHUNK_SIZE", "1100"))
    chunk_overlap: int = int(os.getenv("DOC_RAG_CHUNK_OVERLAP", "180"))
    top_k: int = int(os.getenv("DOC_RAG_TOP_K", "5"))

    # OpenAI (opcional)
    use_openai: bool = os.getenv("RAG_USE_OPENAI", "false").lower() == "true"
    openai_model: str = os.getenv(
        "OPENAI_MODEL", "gpt-4.1"
    )  # prioriza calidad :contentReference[oaicite:1]{index=1}

    # Re-rank (CrossEncoder)
    use_rerank: bool = os.getenv("RAG_USE_RERANK", "true").lower() == "true"
    rerank_model: str = os.getenv(
        "RAG_RERANK_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )  # multilingüe :contentReference[oaicite:1]{index=1}
    rerank_device: str = os.getenv(
        "RAG_RERANK_DEVICE",
        "mps" if platform.system() == "Darwin" else "cpu",
    )
    retrieve_candidates: int = int(os.getenv("RAG_RETRIEVE_CANDIDATES", "40"))

    # Contexto adyacente (para OpenAI)
    adjacent_context: bool = os.getenv("RAG_ADJACENT_CONTEXT", "true").lower() == "true"
    adjacent_n: int = int(os.getenv("RAG_ADJACENT_N", "1"))
    adjacent_same_page: bool = os.getenv("RAG_ADJACENT_SAME_PAGE", "true").lower() == "true"
    adjacent_max_blocks: int = int(os.getenv("RAG_ADJACENT_MAX_BLOCKS", "12"))


SETTINGS = Settings()
SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.uploads_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.index_dir.mkdir(parents=True, exist_ok=True)

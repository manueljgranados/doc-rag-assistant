from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")  # prioriza calidad :contentReference[oaicite:1]{index=1}


SETTINGS = Settings()
SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.uploads_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.index_dir.mkdir(parents=True, exist_ok=True)

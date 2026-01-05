from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from doc_rag.adapters.loaders.md_loader import load_markdown
from doc_rag.adapters.loaders.pdf_loader import load_pdf_pages
from doc_rag.adapters.vectorstore.faiss_store import FaissStore
from doc_rag.core.settings import Settings
from doc_rag.services.chunking import chunk_text
from doc_rag.services.embedding import Embedder


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    doc_id: str
    source_filename: str
    page: int | None
    char_start: int
    char_end: int
    text: str

    def anchor(self) -> str:
        if self.page is None:
            return f"md:c{self.char_start}-{self.char_end}"
        return f"p{self.page}:c{self.char_start}-{self.char_end}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def list_uploads(uploads_dir: Path) -> list[Path]:
    exts = {".pdf", ".md", ".markdown"}
    return sorted([p for p in uploads_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def rebuild_global_index(settings: Settings) -> tuple[int, int]:
    embedder = Embedder(settings.embedding_model)
    store = FaissStore(embedder.dim)

    index_path = settings.index_dir / "global.faiss"
    chunks_path = settings.index_dir / "chunks.jsonl"

    # Reset
    if index_path.exists():
        index_path.unlink()
    if chunks_path.exists():
        chunks_path.unlink()

    chunk_records: list[ChunkRecord] = []
    uploads = list_uploads(settings.uploads_dir)

    next_id = 0
    for file_path in uploads:
        doc_id = sha256_file(file_path)
        source_filename = file_path.name
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            for page in load_pdf_pages(file_path):
                for ch in chunk_text(page.text, settings.chunk_size, settings.chunk_overlap):
                    chunk_records.append(
                        ChunkRecord(
                            id=next_id,
                            doc_id=doc_id,
                            source_filename=source_filename,
                            page=page.page_number,
                            char_start=ch.char_start,
                            char_end=ch.char_end,
                            text=ch.text,
                        )
                    )
                    next_id += 1
        else:
            text = load_markdown(file_path)
            for ch in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
                chunk_records.append(
                    ChunkRecord(
                        id=next_id,
                        doc_id=doc_id,
                        source_filename=source_filename,
                        page=None,
                        char_start=ch.char_start,
                        char_end=ch.char_end,
                        text=ch.text,
                    )
                )
                next_id += 1

    # Embeddings + FAISS
    texts = [c.text for c in chunk_records]
    if texts:
        vecs = embedder.encode(texts)
        store.add(vecs)
        store.save(index_path)

    # JSONL metadata
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunk_records:
            f.write(
                json.dumps(
                    {
                        "id": c.id,
                        "doc_id": c.doc_id,
                        "source_filename": c.source_filename,
                        "page": c.page,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "anchor": c.anchor(),
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return len(uploads), len(chunk_records)

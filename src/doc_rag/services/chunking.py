from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap debe cumplir 0 <= overlap < chunk_size")

    clean = " ".join(text.split())
    n = len(clean)
    if n == 0:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, char_start=start, char_end=end))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

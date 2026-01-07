from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    doc_id: str
    stored_filename: str
    original_filename: str


class ReindexResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None
    use_openai: bool | None = None
    use_rerank: bool | None = None
    doc_id: str | None = None
    source_filename: str | None = None


class Citation(BaseModel):
    doc_id: str
    source_filename: str
    page: int | None
    anchor: str
    section: str | None = None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]

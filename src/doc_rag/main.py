from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from doc_rag.core.models import (
    Citation,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    UploadResponse,
)
from doc_rag.core.settings import SETTINGS
from doc_rag.services.indexer import rebuild_global_index
from doc_rag.services.retriever import Retriever

app = FastAPI(title="Doc RAG Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever(SETTINGS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown"}:
        raise HTTPException(status_code=400, detail="Formato no permitido. Use PDF o Markdown.")

    content = await file.read()
    max_bytes = SETTINGS.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Máximo {SETTINGS.max_upload_mb} MB.")

    doc_id = hashlib.sha256(content).hexdigest()
    safe_name = (Path(file.filename).name if file.filename else f"{doc_id}{suffix}").replace(" ", "_")
    stored_filename = f"{doc_id[:12]}_{safe_name}"
    stored_path = SETTINGS.uploads_dir / stored_filename
    stored_path.write_bytes(content)

    return UploadResponse(doc_id=doc_id, stored_filename=stored_filename, original_filename=safe_name)


@app.post("/documents/reindex", response_model=ReindexResponse)
def reindex():
    docs, chunks = rebuild_global_index(SETTINGS)
    # fuerza recarga en el siguiente query
    retriever._index = None
    retriever._chunks_by_id.clear()
    return ReindexResponse(indexed_documents=docs, indexed_chunks=chunks)


def _build_context_blocks(results: list[dict], max_blocks: int = 5) -> list[str]:
    blocks: list[str] = []
    for r in results[:max_blocks]:
        cite = f"[{r['source_filename']} | {r['anchor']}]"
        blocks.append(f"{cite}\n{r['text']}")
    return blocks


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or SETTINGS.top_k
    use_openai = req.use_openai if req.use_openai is not None else SETTINGS.use_openai

    try:
        results = retriever.search(req.question, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    citations: list[Citation] = []
    for r in results:
        snippet = r["text"][:350] + ("…" if len(r["text"]) > 350 else "")
        citations.append(
            Citation(
                doc_id=r["doc_id"],
                source_filename=r["source_filename"],
                page=r.get("page"),
                anchor=r["anchor"],
                score=float(r["score"]),
                snippet=snippet,
            )
        )

    # Respuesta
    if use_openai:
        try:
            from doc_rag.adapters.llm.openai_client import OpenAIAnswerer

            answerer = OpenAIAnswerer(model=SETTINGS.openai_model)
            context_blocks = _build_context_blocks(results, max_blocks=top_k)
            answer = answerer.answer(req.question, context_blocks)
            return QueryResponse(answer=answer, citations=citations)
        except Exception as e:
            # cae a modo extractivo
            pass

    # Modo extractivo (sin LLM)
    if not results:
        answer = "No se han encontrado fragmentos relevantes en el índice."
    else:
        lines = ["He encontrado estos fragmentos relevantes:"]
        for r in results[: min(3, len(results))]:
            lines.append(f"- [{r['source_filename']} | {r['anchor']}] {r['text']}")
        answer = "\n".join(lines)

    return QueryResponse(answer=answer, citations=citations)


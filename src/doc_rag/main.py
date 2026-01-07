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
from doc_rag.services.indexer import rebuild_global_index, list_uploads, sha256_file
from doc_rag.services.retriever import Retriever
from doc_rag.services.intent import infer_intent

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


@app.get("/documents")
def documents():
    items = []
    for p in list_uploads(SETTINGS.uploads_dir):
        items.append(
            {
                "doc_id": sha256_file(p),
                "source_filename": p.name,
                "size_bytes": p.stat().st_size,
            }
        )
    return items


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
    safe_name = (Path(file.filename).name if file.filename else f"{doc_id}{suffix}").replace(
        " ", "_"
    )
    stored_filename = f"{doc_id[:12]}_{safe_name}"
    stored_path = SETTINGS.uploads_dir / stored_filename
    stored_path.write_bytes(content)

    return UploadResponse(
        doc_id=doc_id, stored_filename=stored_filename, original_filename=safe_name
    )


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


def _build_context_blocks_with_neighbors(
    results: list[dict],
    max_blocks: int = 12,
    neighbor_n: int = 1,
    same_page: bool = True,
) -> list[str]:
    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()

    def add(rec: dict) -> None:
        key = (rec["source_filename"], rec["anchor"])
        if key in seen:
            return
        seen.add(key)
        cite = f"[{rec['source_filename']} | {rec['anchor']}]"
        blocks.append(f"{cite}\n{rec['text']}")

    for r in results:
        # vecinos previos
        for nb in retriever.neighbors(int(r["id"]), n=neighbor_n, same_page=same_page):
            add(nb)
        # chunk principal
        add(r)

        if len(blocks) >= max_blocks:
            break

    return blocks[:max_blocks]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or SETTINGS.top_k
    use_openai = req.use_openai if req.use_openai is not None else SETTINGS.use_openai
    use_rerank = req.use_rerank if req.use_rerank is not None else SETTINGS.use_rerank
    plan = infer_intent(req.question)

    try:
        results = retriever.search(
            req.question,
            top_k=top_k,
            use_rerank=use_rerank,
            doc_id=req.doc_id,
            source_filename=req.source_filename,
            preferred_sections=plan.preferred_sections,
        )
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
                section=r.get("section"),
                snippet=snippet,
            )
        )

    # Respuesta
    if use_openai:
        try:
            from doc_rag.adapters.llm.openai_client import OpenAIAnswerer

            answerer = OpenAIAnswerer(model=SETTINGS.openai_model)
            if SETTINGS.adjacent_context:
                context_blocks = _build_context_blocks_with_neighbors(
                    results,
                    max_blocks=SETTINGS.adjacent_max_blocks,
                    neighbor_n=SETTINGS.adjacent_n,
                    same_page=SETTINGS.adjacent_same_page,
                )
            else:
                context_blocks = _build_context_blocks(results, max_blocks=top_k)

            answer = answerer.answer(req.question, context_blocks, prompt_style=plan.prompt_style)
            return QueryResponse(answer=answer, citations=citations)
        except Exception:
            # pasa a modo extractivo
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

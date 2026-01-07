# Doc RAG Assistant (papers) — índice local con citas

Asistente RAG **local** para consultar **papers en PDF (texto copiable)** y documentos Markdown. Permite subir documentos, reindexar un **índice global** y realizar preguntas con **citas trazables** (fichero + ancla). Incluye **re-rank** (Cross-Encoder) y **contexto adyacente** (chunk anterior/posterior en la misma página) para mejorar la calidad de respuesta en artículos científicos.

## Casos de uso
- “¿Qué dice el paper *X* sobre *Y*?”
- “¿Qué objetivos y conclusiones tiene el paper *X*?”
- Búsqueda y explicación con referencias exactas al documento original.

---

## Características principales
- **Índice global único** (FAISS) sobre todos los documentos cargados.
- **Filtro por paper** en la UI (selección por fichero).
- **Citas completas**: `archivo + ancla` (página y offsets) + snippet.
- **Re-rank multilingüe (ES/EN)** con Cross-Encoder para mejorar precisión.
- **Contexto adyacente (misma página)** para respuestas más coherentes con LLM.
- **OpenAI opcional**: si hay `OPENAI_API_KEY`, genera respuestas con mayor calidad; si no, modo extractivo con citas.
- **Diseñado para uso local**: los documentos se almacenan en `data/uploads`.

---

## Stack tecnológico
- Backend: **FastAPI**
- UI: **Streamlit**
- Embeddings: **SentenceTransformers** (multilingüe)
- Vector store: **FAISS**
- (Opcional) LLM: **OpenAI** (Responses API)

---

## Requisitos
- Python **3.12**
- Recomendado: `uv` para gestión de entorno y dependencias

> Nota (macOS Apple Silicon): si tuviese problemas al instalar `faiss-cpu`, la alternativa habitual es instalar FAISS desde conda-forge y mantener el resto con `uv`.

---

## Instalación y ejecución (local)

### 1) Clonar e instalar dependencias
```bash
git clone <SU_URL_DEL_REPO>
cd doc-rag-assistant

pip install uv
uv sync --all-extras
```
### 2) Ejecutar backend + UI
```bash
./scripts/dev/run_local.sh
```
- Backend: `http//localhost:8000`
- UI: `http://localhost:8501`

---

## Flujo de uso
1. **Subir** PDF/MD desde la UI (máx. 20MB por archivo).
2. Pulsar **"Reindexar todo (global)"**.
3. Seleccionar un **paper** (opcional) en "Filtro por paper".
4. Preguntar y revisar:
     - Respuesta
     - Citas con archivo|ancla y snippet

---

## Configuración (variables de entorno)

### Rutas y límites
```bash
export DOC_RAG_DATA_DIR="data"
export DOC_RAG_UPLOADS_DIR="data/uploads"
export DOC_RAG_INDEX_DIR="data/index"
export DOC_RAG_MAX_UPLOAD_MB=20
```

### RAG (embeddings/chunking)
```bash
export DOC_RAG_EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
export DOC_RAG_CHUNK_SIZE=1100
export DOC_RAG_CHUNK_OVERLAP=180
export DOC_RAG_TOP_K=5
```

### Re-rank (recomendado para papers)
```bash
export RAG_USE_RERANK=true
export RAG_RERANK_MODEL="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
export RAG_RERANK_DEVICE="mps"   # macOS (Apple Silicon) / "cpu" en otros
export RAG_RETRIEVE_CANDIDATES=60
```

### Contexto adyacente
```bash
export RAG_ADJACENT_CONTEXT=true
export RAG_ADJACENT_N=1
export RAG_ADJACENT_SAME_PAGE=true
export RAG_ADJACENT_MAX_BLOCKS=12
```

### OpenAI (opcional)
```bash
export RAG_USE_OPENAI=true
export OPENAI_API_KEY="su_api_key"
export OPENAI_MODEL="gpt-5.1"
```
Nota: Si `RAG_USE_OPENAI=false` o no hay `OPENAI_API_KEY`, el sistema devuelve una respuesta extractiva con citas.

---

## Endpoints (backend)
- `POST /documents/upload` &rarr; subir PDF/MD
- `POST /documents/reindex` &rarr; reconstruir índice global
- `GET /documents` &rarr; listar documentos disponibles (para filtro por paper)
- `POST /query` &rarr; consulta (con opcional `doc_id` / `source_filename`)

---

## Estructura del repositorio (resumen)
- `src/doc_rag/main.py` — API FastAPI
- `src/doc_rag/ui.py` — UI Streamlit
- `src/doc_rag/services/` — chunking, embeddings, indexado, retrieval, rerank, intent
- `src/doc_rag/adapters/` — loaders (PDF/MD), FAISS, OpenAI
- `data/uploads/` — documentos cargados (no versionado)
- `data/index/` — índice FAISS + metadatos (no versionado)

---

## Limitaciones conocidas
- Pensado para **PDF con texto copiable** (no OCR).
- El chunking es heurístico; funciona bien en papers, pero puede requerir ajustes en documentos con maquetación compleja.
- En modo OpenAI, la calidad depende del contexto recuperado y del modelo configurado.

---

## Roadmap (mejoras posibles)
- Indexado incremental (añadir/quitar documentos sin reconstruir todo).
- Mejor limpieza de PDF (cabeceras/pies repetidos, guiones por salto de línea).
- Chunking por secciones más robusto (Abstract/Methods/Results/Conclusion).
- Evaluación con conjunto de preguntas y métricas (precision@k, MRR).

---

## Licencia
Apache-2.0.

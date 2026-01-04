#!/usr/bin/env bash
set -e

# Backend
uv run uvicorn doc_rag.main:app --reload --port 8000 &
BACK_PID=$!

# UI
uv run streamlit run src/doc_rag/ui.py --server.port 8501

kill $BACK_PID

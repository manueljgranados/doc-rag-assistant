#!/usr/bin/env bash
set -e

export PYTHONPATH="src"

uv run uvicorn doc_rag.main:app --reload --port 8000 &
BACK_PID=$!

uv run streamlit run src/doc_rag/ui.py --server.port 8501

kill $BACK_PID

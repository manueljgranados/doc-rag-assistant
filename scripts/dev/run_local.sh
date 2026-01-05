#!/usr/bin/env bash
set -e

export PYTHONPATH="src"

cleanup () {
  if [[ -n "${BACK_PID:-}" ]]; then
    kill "$BACK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

uv run uvicorn doc_rag.main:app --reload --port 8000 &
BACK_PID=$!

uv run streamlit run src/doc_rag/ui.py --server.port 8501

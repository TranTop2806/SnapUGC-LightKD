#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export SNAPUGC_LLM_BACKEND="${SNAPUGC_LLM_BACKEND:-local}"
export SNAPUGC_LOCAL_LLM_MODEL="${SNAPUGC_LOCAL_LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export SNAPUGC_LOCAL_LLM_CACHE="${SNAPUGC_LOCAL_LLM_CACHE:-$HOME/.cache/snapugc-local-llm}"

if [[ "${SNAPUGC_PREPARE_LOCAL_LLM:-0}" == "1" ]]; then
  python scripts/prepare_local_llm.py \
    --model "$SNAPUGC_LOCAL_LLM_MODEL" \
    --cache-dir "$SNAPUGC_LOCAL_LLM_CACHE"
fi

exec python -m uvicorn demo_app.app:app \
  --host "${SNAPUGC_DEMO_HOST:-127.0.0.1}" \
  --port "${SNAPUGC_DEMO_PORT:-7860}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

# Load local, shell-compatible demo configuration without committing secrets.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export SNAPUGC_LLM_BACKEND="${SNAPUGC_LLM_BACKEND:-auto}"
export SNAPUGC_LOCAL_LLM_MODEL="${SNAPUGC_LOCAL_LLM_MODEL:-Qwen/Qwen3.5-4B}"
export SNAPUGC_LOCAL_LLM_CACHE="${SNAPUGC_LOCAL_LLM_CACHE:-$HOME/.cache/snapugc-local-llm}"
export SNAPUGC_LLM_FALLBACK_TO_OPENAI="${SNAPUGC_LLM_FALLBACK_TO_OPENAI:-1}"

if [[ "${SNAPUGC_PREPARE_LOCAL_LLM:-0}" == "1" ]]; then
  "$PYTHON_BIN" scripts/prepare_local_llm.py \
    --model "$SNAPUGC_LOCAL_LLM_MODEL" \
    --cache-dir "$SNAPUGC_LOCAL_LLM_CACHE"
fi

exec "$PYTHON_BIN" -m uvicorn demo_app.app:app \
  --host "${SNAPUGC_DEMO_HOST:-127.0.0.1}" \
  --port "${SNAPUGC_DEMO_PORT:-7860}"

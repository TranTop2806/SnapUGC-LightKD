#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export SNAPUGC_REPORT_JSON="${SNAPUGC_REPORT_JSON:-$ROOT_DIR/results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json}"
export SNAPUGC_STUDENT_CHECKPOINT="${SNAPUGC_STUDENT_CHECKPOINT:-$ROOT_DIR/results/kd_tuning_official_5k/v05_small_cosine_rank/student_kd_best.pth}"
export SNAPUGC_LABELS_CSV="${SNAPUGC_LABELS_CSV:-$ROOT_DIR/data/official_5k_split/split_all_5000.csv}"
export SNAPUGC_STUDENT_INPUT_PRESET="${SNAPUGC_STUDENT_INPUT_PRESET:-clip_mobilenet_text}"
export SNAPUGC_TEXT_ENCODER_MODEL="${SNAPUGC_TEXT_ENCODER_MODEL:-CompVis/stable-diffusion-v1-4}"
export SNAPUGC_DEMO_RUN_DIR="${SNAPUGC_DEMO_RUN_DIR:-$ROOT_DIR/results/demo_runs_proper_kd}"

export SNAPUGC_LLM_BACKEND="${SNAPUGC_LLM_BACKEND:-auto}"
export SNAPUGC_LOCAL_LLM_MODEL="${SNAPUGC_LOCAL_LLM_MODEL:-Qwen/Qwen3.5-4B}"
export SNAPUGC_LOCAL_LLM_CACHE="${SNAPUGC_LOCAL_LLM_CACHE:-$HOME/.cache/snapugc-local-llm}"
export SNAPUGC_LLM_FALLBACK_TO_OPENAI="${SNAPUGC_LLM_FALLBACK_TO_OPENAI:-1}"

if [[ "${SNAPUGC_PREPARE_PROPER_KD:-0}" == "1" ]]; then
  python scripts/prepare_proper_kd_demo.py \
    --report-json "$SNAPUGC_REPORT_JSON" \
    --checkpoint "$SNAPUGC_STUDENT_CHECKPOINT" \
    --text-encoder-model "$SNAPUGC_TEXT_ENCODER_MODEL"
fi

if [[ "${SNAPUGC_PREPARE_LOCAL_LLM:-0}" == "1" ]]; then
  python scripts/prepare_local_llm.py \
    --model "$SNAPUGC_LOCAL_LLM_MODEL" \
    --cache-dir "$SNAPUGC_LOCAL_LLM_CACHE"
fi

exec python -m uvicorn demo_app.app:app \
  --host "${SNAPUGC_DEMO_HOST:-127.0.0.1}" \
  --port "${SNAPUGC_DEMO_PORT:-7861}"

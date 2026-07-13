#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/SnapUGC-LightKD}"
SUBSET_CSV="${SUBSET_CSV:-/workspace/snapugc-data/train_subset_balanced_5000.csv}"
VIDEO_DIR="${VIDEO_DIR:-/workspace/snapugc-data/train_videos_balanced_5000}"
OUT_DIR="${OUT_DIR:-/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32}"
KD_OUT_DIR="${KD_OUT_DIR:-/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/snapugc-checkpoints}"
OFFICIAL_REPO_DIR="${OFFICIAL_REPO_DIR:-${ROOT_DIR}/third_party/SnapUGC_Engagement}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/full_pipeline.log}"
SHUTDOWN_ON_EXIT="${SHUTDOWN_ON_EXIT:-1}"
RUN_OFFICIAL="${RUN_OFFICIAL:-1}"
RUN_STUDENT="${RUN_STUDENT:-1}"
EXPORT_ARTIFACTS="${EXPORT_ARTIFACTS:-1}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUT_DIR}/teacher_artifacts}"
ARTIFACT_SHARD_SIZE="${ARTIFACT_SHARD_SIZE:-25}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR" "$KD_OUT_DIR" "$(dirname "$LOG_FILE")"

shutdown_vm() {
  local status=$?
  echo "FULL_PIPELINE_FINAL_STATUS=${status} $(date -Is)" | tee -a "$LOG_FILE"
  if [[ "$SHUTDOWN_ON_EXIT" == "1" ]]; then
    echo "Shutting down VM now." | tee -a "$LOG_FILE"
    sudo shutdown -h now || true
  fi
  exit "$status"
}
trap shutdown_vm EXIT

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START SNAPUGC FULL GCP PIPELINE $(date -Is) ==="
echo "ROOT_DIR=$ROOT_DIR"
echo "SUBSET_CSV=$SUBSET_CSV"
echo "VIDEO_DIR=$VIDEO_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "KD_OUT_DIR=$KD_OUT_DIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "OFFICIAL_REPO_DIR=$OFFICIAL_REPO_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "RUN_OFFICIAL=$RUN_OFFICIAL"
echo "RUN_STUDENT=$RUN_STUDENT"
echo "EXPORT_ARTIFACTS=$EXPORT_ARTIFACTS"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "ARTIFACT_SHARD_SIZE=$ARTIFACT_SHARD_SIZE"
echo "DEVICE=$DEVICE"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "${ROOT_DIR}/.venv"
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

"$PYTHON_BIN" -m pip install -U pip wheel setuptools
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install -e .

if [[ "$RUN_OFFICIAL" == "1" ]]; then
  echo "=== OFFICIAL TEACHER + ARTIFACT EXPORT $(date -Is) ==="
  official_checkpoint_dir="$OFFICIAL_REPO_DIR/ECR_inference/checkpoints"
  mkdir -p "$official_checkpoint_dir"
  for name in EVQA.pth net_distort6_g_latest.pth r3d18_K_200ep.pth mPLUG2_MSRVTT_Caption.pth ViT-L-14.tar efficientnet_v2_s_21k_ft1k-dbb43f38.pth; do
    if [[ ! -s "$CHECKPOINT_DIR/$name" ]]; then
      echo "Missing checkpoint: $CHECKPOINT_DIR/$name" >&2
      exit 3
    fi
    ln -sfn "$CHECKPOINT_DIR/$name" "$official_checkpoint_dir/$name"
  done

  teacher_args=(
    scripts/run_official_snapugc_evqa.py
    --official-repo-dir "$OFFICIAL_REPO_DIR"
    --videos-dir "$VIDEO_DIR"
    --csv-file "$SUBSET_CSV"
    --out-dir "$OUT_DIR"
    --python "$PYTHON_BIN"
  )
  if [[ "$EXPORT_ARTIFACTS" == "1" ]]; then
    teacher_args+=(
      --export-artifacts
      --artifact-dir "$ARTIFACT_DIR"
      --artifact-shard-size "$ARTIFACT_SHARD_SIZE"
    )
  fi
  "$PYTHON_BIN" "${teacher_args[@]}"
fi

if [[ "$RUN_STUDENT" == "1" ]]; then
  echo "=== STUDENT KD TRAINING $(date -Is) ==="
  "$PYTHON_BIN" scripts/train_official_student_kd.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --labels-csv "$SUBSET_CSV" \
    --save-dir "$KD_OUT_DIR" \
    --input-preset visual_text_sound \
    --epochs 40 \
    --batch 64 \
    --eval-batch 256 \
    --hidden-dim 96 \
    --layers 1 \
    --heads 4 \
    --dropout 0.22 \
    --lr 4e-4 \
    --weight-decay 0.03 \
    --repr-loss cosine \
    --soft-weight 1.1 \
    --clip-weight 0.08 \
    --temporal-weight 0.02 \
    --fusion-weight 0.02 \
    --attention-weight 0.005 \
    --hard-rank-weight 0.02 \
    --teacher-rank-weight 0.12 \
    --device "$DEVICE"
fi

if [[ -n "${EXPLAIN_VIDEO_ID:-}" ]]; then
  echo "=== EXPLANATION INFERENCE ${EXPLAIN_VIDEO_ID} $(date -Is) ==="
  mkdir -p /workspace/results/explanations
  "$PYTHON_BIN" scripts/infer_one_video_with_expl.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --labels-csv "$SUBSET_CSV" \
    --video-id "$EXPLAIN_VIDEO_ID" \
    --report-json "${KD_OUT_DIR}/official_student_kd_report.json" \
    --device "$DEVICE" \
    --topk "${EXPLAIN_TOPK:-3}" \
    --out-json "/workspace/results/explanations/${EXPLAIN_VIDEO_ID}.json"
fi

echo "=== DONE SNAPUGC FULL GCP PIPELINE $(date -Is) ==="

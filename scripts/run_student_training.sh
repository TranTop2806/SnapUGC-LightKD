#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

MODE="${MODE:-semi}"
DEVICE="${DEVICE:-cpu}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts}"
LABELS_CSV="${LABELS_CSV:-$ROOT_DIR/data/train_subset_balanced_5000.csv}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT_DIR/data/official_5k_split_4000_500_500}"
QUALITY_FEATURES="${QUALITY_FEATURES:-$ROOT_DIR/results/clip_vitb32_keyframe_features_5000.npz}"
LITE_ACTION_FEATURES="${LITE_ACTION_FEATURES:-$ROOT_DIR/results/lite_action_features_5000.npz}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/results/student_training/$MODE}"

for path in "$ARTIFACT_DIR" "$LABELS_CSV" "$SPLIT_DIR/train_ids.txt" "$SPLIT_DIR/val_ids.txt" "$QUALITY_FEATURES"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required artifact: $path" >&2
    exit 2
  fi
done

common=(
  --artifact-dir "$ARTIFACT_DIR"
  --labels-csv "$LABELS_CSV"
  --save-dir "$SAVE_DIR"
  --quality-features "$QUALITY_FEATURES"
  --epochs 100 --batch 32 --eval-batch 128
  --lr 5e-4 --weight-decay 0.02
  --seed 42 --split-seed 20260706
  --train-ids "$SPLIT_DIR/train_ids.txt"
  --val-ids "$SPLIT_DIR/val_ids.txt"
  --device "$DEVICE" --run-kind kd --repr-loss cosine
  --hard-weight 1.0 --soft-weight 1.1 --clip-weight 0.08
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005
  --hard-rank-weight 0.04 --teacher-rank-weight 0.18
  --teacher-pearson-weight 0.02 --teacher-spearman-weight 0.015
  --teacher-listwise-weight 0.02 --student-teacher-relation-weight 0.02
  --contrastive-hidden-weight 0.02
)

case "$MODE" in
  semi)
    "$PYTHON_BIN" scripts/train_official_student_kd.py "${common[@]}" \
      --input-preset visual_text_sound --quality-fusion clip_add \
      --hidden-dim 96 --layers 1 --heads 4 --dropout 0.25
    ;;
  proper)
    if [[ ! -f "$LITE_ACTION_FEATURES" ]]; then
      echo "Proper KD also requires: $LITE_ACTION_FEATURES" >&2
      exit 2
    fi
    "$PYTHON_BIN" scripts/train_official_student_kd.py "${common[@]}" \
      --input-preset clip_mobilenet_text --quality-fusion input_concat \
      --lite-action-features "$LITE_ACTION_FEATURES" \
      --hidden-dim 192 --layers 2 --heads 8 --dropout 0.25
    ;;
  *)
    echo "MODE must be 'semi' or 'proper'" >&2
    exit 2
    ;;
esac

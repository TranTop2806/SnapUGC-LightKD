#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
KIND="${KIND:-full}"
OUT_ROOT="${OUT_ROOT:-results/final_4000_500_500_2026}"
ARTIFACT_DIR="results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts"
LABELS="data/train_subset_balanced_5000.csv"
QUALITY="results/clip_vitb32_keyframe_features_5000.npz"
SPLIT_DIR="data/official_5k_split_4000_500_500"

case "$KIND" in
  full|logit_only|hard_transformer|hard_mlp) ;;
  *) echo "KIND must be full, logit_only, hard_transformer, or hard_mlp" >&2; exit 2 ;;
esac

common=(
  --artifact-dir "$ARTIFACT_DIR" --labels-csv "$LABELS"
  --input-preset visual_text_sound --quality-features "$QUALITY" --quality-fusion clip_add
  --hidden-dim 96 --layers 1 --heads 4 --dropout 0.25
  --epochs 60 --batch 32 --eval-batch 128 --lr 5e-4 --weight-decay 0.02
  --split-seed 20260706 --train-ids "$SPLIT_DIR/train_ids.txt" --val-ids "$SPLIT_DIR/val_ids.txt"
  --device "$DEVICE"
)

full_weights=(
  --run-kind kd --hard-weight 1.0 --soft-weight 1.1 --clip-weight 0.08
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005
  --hard-rank-weight 0.04 --teacher-rank-weight 0.18
  --teacher-pearson-weight 0.02 --teacher-spearman-weight 0.015
  --teacher-listwise-weight 0.02 --student-teacher-relation-weight 0.02
  --contrastive-hidden-weight 0.02
)

logit_weights=(
  --run-kind kd --hard-weight 0 --soft-weight 1 --clip-weight 0
  --temporal-weight 0 --fusion-weight 0 --attention-weight 0
  --hard-rank-weight 0 --teacher-rank-weight 0
  --teacher-pearson-weight 0 --teacher-spearman-weight 0
  --teacher-listwise-weight 0 --student-teacher-relation-weight 0
  --contrastive-hidden-weight 0
)

hard_weights=(
  --run-kind baseline --hard-rank-weight 0
)

for seed in 42 43 44 45 46; do
  out="$OUT_ROOT/${KIND}_seed${seed}"
  if [[ -f "$out/official_student_kd_report.json" ]]; then
    echo "skip ${KIND}_seed${seed} (report exists)"
    continue
  fi
  mkdir -p "$out"
  weights=("${full_weights[@]}")
  [[ "$KIND" == "logit_only" ]] && weights=("${logit_weights[@]}")
  [[ "$KIND" == hard_* ]] && weights=("${hard_weights[@]}")
  layers=1
  [[ "$KIND" == "hard_mlp" ]] && layers=0
  "$PYTHON" scripts/train_official_student_kd.py "${common[@]}" \
    --layers "$layers" --save-dir "$out" --seed "$seed" "${weights[@]}" \
    2>&1 | tee "$out/train.log"
done

#!/bin/bash
# run_proper_kd_experiments.sh
#
# Proper KD experiments: student uses its OWN feature extractors.
# Visual: CLIP ViT-B/32 (512-d, pre-extracted, deployable independently)
# Motion: MobileNetV3-Small (1152-d, pre-extracted, deployable independently)
# Text:   CLIP text embeddings from teacher artifacts (sound/title/description)
# KD:     Teacher ECR + hidden/temporal features as supervision signals only.
#
# At inference, student needs NO teacher — only CLIP ViT-B/32 + MobileNetV3.
set -e

ROOT_DIR="/Users/top/Documents/HCMUS/KhoaLuan/SnapUGC-LightKD"
ARTIFACT_DIR="$ROOT_DIR/results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts"
LABELS_CSV="$ROOT_DIR/data/train_subset_balanced_5000.csv"
CLIP_FEAT="$ROOT_DIR/results/clip_vitb32_keyframe_features_5000.npz"
LITE_ACTION_FEAT="$ROOT_DIR/results/lite_action_features_5000.npz"
SAVE_BASE="$ROOT_DIR/results/proper_kd"

cd "$ROOT_DIR"
echo "=== PROPER KD EXPERIMENTS — CLIP + MobileNet Student ==="
echo "Visual input: CLIP ViT-B/32 [T, 512] — student's own backbone"
echo "Motion input: MobileNetV3-Small [T, 1152] — student's own backbone"
echo "Teacher role: KD supervision only (NOT visual input)"
date

# ─── Config 1: Small Student — baseline (no KD) ──────────────────────────────
echo ""
echo "--- [1/4] Small Student Baseline (hidden=96, L=1, no KD) ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/small_baseline_h96_l1" \
  --input-preset clip_mobilenet_text \
  --quality-features "$CLIP_FEAT" \
  --quality-fusion input_concat \
  --lite-action-features "$LITE_ACTION_FEAT" \
  --hidden-dim 96 --layers 1 --heads 4 \
  --dropout 0.22 --epochs 60 --batch 64 --eval-batch 256 \
  --lr 3e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps \
  --run-kind baseline

# ─── Config 2: Small Student — with KD ───────────────────────────────────────
echo ""
echo "--- [2/4] Small Student with KD (hidden=96, L=1) ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/small_kd_h96_l1" \
  --input-preset clip_mobilenet_text \
  --quality-features "$CLIP_FEAT" \
  --quality-fusion input_concat \
  --lite-action-features "$LITE_ACTION_FEAT" \
  --hidden-dim 96 --layers 1 --heads 4 \
  --dropout 0.22 --epochs 60 --batch 64 --eval-batch 256 \
  --lr 3e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps \
  --run-kind kd \
  --kd-curriculum three_phase \
  --soft-weight 1.1 --clip-weight 0.08 \
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 --teacher-listwise-weight 0.02 \
  --student-teacher-relation-weight 0.02 --contrastive-hidden-weight 0.02

# ─── Config 3: Medium Student — baseline (no KD) ─────────────────────────────
echo ""
echo "--- [3/4] Medium Student Baseline (hidden=192, L=2, no KD) ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/medium_baseline_h192_l2" \
  --input-preset clip_mobilenet_text \
  --quality-features "$CLIP_FEAT" \
  --quality-fusion input_concat \
  --lite-action-features "$LITE_ACTION_FEAT" \
  --hidden-dim 192 --layers 2 --heads 8 \
  --dropout 0.25 --epochs 60 --batch 32 --eval-batch 128 \
  --lr 3e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps \
  --run-kind baseline

# ─── Config 4: Medium Student — with KD ──────────────────────────────────────
echo ""
echo "--- [4/4] Medium Student with KD (hidden=192, L=2) ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/medium_kd_h192_l2" \
  --input-preset clip_mobilenet_text \
  --quality-features "$CLIP_FEAT" \
  --quality-fusion input_concat \
  --lite-action-features "$LITE_ACTION_FEAT" \
  --hidden-dim 192 --layers 2 --heads 8 \
  --dropout 0.25 --epochs 60 --batch 32 --eval-batch 128 \
  --lr 3e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps \
  --run-kind kd \
  --kd-curriculum three_phase \
  --soft-weight 1.1 --clip-weight 0.08 \
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 --teacher-listwise-weight 0.02 \
  --student-teacher-relation-weight 0.02 --contrastive-hidden-weight 0.02

echo ""
echo "=== ALL PROPER KD EXPERIMENTS COMPLETED ==="
date

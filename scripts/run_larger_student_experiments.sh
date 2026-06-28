#!/bin/bash
set -e

# Base directories
ROOT_DIR="/Users/top/Documents/HCMUS/KhoaLuan/SnapUGC-LightKD"
ARTIFACT_DIR="$ROOT_DIR/results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts"
LABELS_CSV="$ROOT_DIR/data/train_subset_balanced_5000.csv"
QUALITY_FEAT="$ROOT_DIR/results/clip_vitb32_keyframe_features_5000.npz"
SAVE_BASE="$ROOT_DIR/results/kd_tuning_official_5k"

# Switch to root directory
cd "$ROOT_DIR"

echo "=== STARTING LARGER STUDENT EXPERIMENTS ON MPS ==="
date

# 1. Config A: Medium Student (~3.8M parameters)
# hidden_dim=192, layers=2, heads=8
echo ""
echo "--- Running Config A: Medium Student ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/improve_larger_medium_h192_l2" \
  --input-preset visual_text_sound \
  --quality-features "$QUALITY_FEAT" \
  --quality-fusion clip_add \
  --hidden-dim 192 --layers 2 --heads 8 \
  --dropout 0.25 --epochs 80 --batch 32 --eval-batch 128 \
  --lr 5e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps --run-kind kd \
  --kd-curriculum three_phase \
  --soft-weight 1.1 --clip-weight 0.08 \
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 --teacher-listwise-weight 0.02 \
  --student-teacher-relation-weight 0.02 --contrastive-hidden-weight 0.02

# 2. Config B: Mixture-of-Experts (MoE) Student (~4.1M parameters)
# hidden_dim=192, layers=2, heads=8, fusion_experts=4
echo ""
echo "--- Running Config B: Mixture-of-Experts (MoE) Student ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/improve_larger_moe_h192_l2_e4" \
  --input-preset visual_text_sound \
  --quality-features "$QUALITY_FEAT" \
  --quality-fusion clip_add \
  --hidden-dim 192 --layers 2 --heads 8 --fusion-experts 4 \
  --dropout 0.25 --epochs 80 --batch 32 --eval-batch 128 \
  --lr 5e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps --run-kind kd \
  --kd-curriculum three_phase \
  --soft-weight 1.1 --clip-weight 0.08 \
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 --teacher-listwise-weight 0.02 \
  --student-teacher-relation-weight 0.02 --contrastive-hidden-weight 0.02

# 3. Config C: Large Student (~6.9M parameters, 1/10 of Teacher)
# hidden_dim=256, layers=3, heads=8
echo ""
echo "--- Running Config C: Large Student ---"
python3 scripts/train_official_student_kd.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --labels-csv "$LABELS_CSV" \
  --save-dir "$SAVE_BASE/improve_larger_large_h256_l3" \
  --input-preset visual_text_sound \
  --quality-features "$QUALITY_FEAT" \
  --quality-fusion clip_add \
  --hidden-dim 256 --layers 3 --heads 8 \
  --dropout 0.25 --epochs 80 --batch 32 --eval-batch 128 \
  --lr 5e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps --run-kind kd \
  --kd-curriculum three_phase \
  --soft-weight 1.1 --clip-weight 0.08 \
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 --teacher-listwise-weight 0.02 \
  --student-teacher-relation-weight 0.02 --contrastive-hidden-weight 0.02

echo ""
echo "=== ALL LARGER STUDENT EXPERIMENTS COMPLETED SUCCESSFULLY! ==="
date

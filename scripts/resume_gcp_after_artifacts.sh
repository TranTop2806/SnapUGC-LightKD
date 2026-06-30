#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/workspace/SnapUGC-LightKD}"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUT="${OUT_DIR:-$HOME/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32}"
ART="${ARTIFACT_DIR:-$OUT/teacher_artifacts}"
KD="${KD_OUT_DIR:-$HOME/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank}"
SUBSET="${SUBSET_CSV:-$ROOT/data/official_5k_split/split_all_5000.csv}"
CSV="$OUT/resume_after_existing_artifacts.csv"
VID="${VIDEO_DIR:-$HOME/workspace/snapugc-data/train_videos_balanced_5000}"
REPO="${OFFICIAL_REPO_DIR:-$ROOT/third_party/SnapUGC_Engagement}"
OFFSET_FILE="$OUT/resume_offset.txt"

cd "$ROOT"
mkdir -p "$OUT" "$ART" "$KD"
export SUBSET CSV OFFSET_FILE ART

"$PY" - <<'PY'
import csv
import os
from pathlib import Path

import numpy as np

subset = Path(os.environ["SUBSET"])
out = Path(os.environ["CSV"])
offset_file = Path(os.environ["OFFSET_FILE"])
artifact_dir = Path(os.environ["ART"])

done_ids = set()
max_idx = -1
for path in sorted(artifact_dir.glob("official_teacher_artifacts_*.npz")):
    data = np.load(path, allow_pickle=False)
    ids = [str(value) for value in data["ids"].tolist()]
    idxs = [int(value) for value in data["order_idx"].tolist()]
    done_ids.update(ids)
    if idxs:
        max_idx = max(max_idx, max(idxs))

with subset.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if str(row["Id"]) not in done_ids]

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

offset = max_idx + 1
offset_file.write_text(f"{offset}\n{len(rows)}\n{len(done_ids)}\n", encoding="utf-8")
print(f"resume_state done={len(done_ids)} offset={offset} remaining={len(rows)} csv={out}", flush=True)
PY

OFFSET="$(sed -n '1p' "$OFFSET_FILE")"
REMAINING="$(sed -n '2p' "$OFFSET_FILE")"
DONE="$(sed -n '3p' "$OFFSET_FILE")"

echo "=== RESUME OFFICIAL TEACHER $(date -Is) ==="
echo "done=$DONE offset=$OFFSET remaining=$REMAINING csv=$CSV"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SNAPUGC_DATALOADER_WORKERS="${SNAPUGC_DATALOADER_WORKERS:-0}"
export SNAPUGC_OFFICIAL_FRAME_BATCH="${SNAPUGC_OFFICIAL_FRAME_BATCH:-8}"
export SNAPUGC_MPLUG_CLIP_BATCH="${SNAPUGC_MPLUG_CLIP_BATCH:-1}"
export SNAPUGC_CAPTION_NUM_FRAMES="${SNAPUGC_CAPTION_NUM_FRAMES:-8}"
export SNAPUGC_ARTIFACT_INDEX_OFFSET="$OFFSET"
export ARTIFACT_SHARD_SIZE="${ARTIFACT_SHARD_SIZE:-5}"

if [[ "$REMAINING" != "0" ]]; then
  "$PY" scripts/run_official_snapugc_evqa.py \
    --official-repo-dir "$REPO" \
    --videos-dir "$VID" \
    --csv-file "$CSV" \
    --out-dir "$OUT" \
    --python "$PY" \
    --export-artifacts \
    --artifact-dir "$ART" \
    --artifact-shard-size "$ARTIFACT_SHARD_SIZE"
fi

echo "=== STUDENT KD TRAINING $(date -Is) ==="
"$PY" scripts/train_official_student_kd.py \
  --artifact-dir "$ART" \
  --labels-csv "$SUBSET" \
  --save-dir "$KD" \
  --input-preset visual_text \
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
  --device cuda

echo "=== RESUME PIPELINE DONE $(date -Is) ==="

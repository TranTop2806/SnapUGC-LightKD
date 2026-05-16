#!/bin/bash
# Extract DOVER features in batches to avoid MPS memory leaks
VIDEO_DIR=data/official_balanced_5000_videos
CSV=data/train_subset_balanced_5000.csv
OUT=results/dover_features_5000.npz
BATCH_SIZE=50
DEVICE=mps

python3 -c "
import pandas as pd
df = pd.read_csv('$CSV')
ids = df['Id'].astype(str).tolist()
with open('results/dover_batches/todo.txt', 'w') as f:
    for vid in ids:
        f.write(vid + '\n')
print('Total videos:', len(ids))
"

mkdir -p results/dover_batches

TOTAL=5000
for START in $(seq 0 $BATCH_SIZE $((TOTAL-1))); do
    END=$((START + BATCH_SIZE))
    BATCH_OUT="results/dover_batches/batch_${START}_${END}.npz"
    if [ -f "$BATCH_OUT" ]; then
        echo "Skip batch $START-$END (exists)"
        continue
    fi
    echo "Running batch $START-$END ..."
    PYTHONWARNINGS=ignore python3 scripts/extract_dover_features.py \
        --video-dir "$VIDEO_DIR" \
        --csv "$CSV" \
        --out "$BATCH_OUT" \
        --device "$DEVICE" \
        --max-videos $BATCH_SIZE \
        2>/dev/null
    if [ $? -ne 0 ]; then
        echo "FAILED batch $START-$END"
    fi
    sleep 2
done

echo "Merging batches..."
python3 scripts/merge_dover_batches.py --out "$OUT"

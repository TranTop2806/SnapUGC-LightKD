# Official 5k Student KD Tuning

All runs below use the same official teacher artifacts and the same
deterministic `4000/1000` student split from the fixed balanced 5000-video
subset.

The first KD attempt improved over baseline but was limited by raw hidden-state
MSE. The hidden losses were hundreds of times larger than the scalar ECR losses,
so tuning shifted KD toward:

- stronger soft ECR supervision from the teacher
- pairwise ranking against true ECR and teacher ECR
- light cosine/normalized representation KD
- a smaller, more regularized student

## Comparison

```text
Teacher on validation split final = 0.703849
```

| version | student | repr loss | baseline final | KD final | gain | best epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| original visual_text KD | 128d, 2 layers | raw MSE | 0.505448 | 0.542934 | +0.037486 | 24 |
| v01 soft/rank | 128d, 2 layers | none | 0.505448 | 0.567120 | +0.061672 | 19 |
| v02 norm repr/rank | 128d, 2 layers | normalized MSE | 0.505448 | 0.572278 | +0.066830 | 11 |
| v03 soft/rank norm-lite | 128d, 2 layers | normalized MSE | 0.505448 | 0.575579 | +0.070131 | 41 |
| v04 cosine repr/rank | 128d, 2 layers | cosine | 0.505448 | 0.575857 | +0.070409 | 36 |
| v05 small baseline | 96d, 1 layer | none | 0.512487 | - | - | 9 |
| v05 small cosine/rank | 96d, 1 layer | cosine | 0.512487 | 0.579973 | +0.067486 | 15 |
| v06 small norm/rank | 96d, 1 layer | normalized MSE | 0.512487 | 0.578893 | +0.066405 | 15 |

Best current result:

```text
results/kd_tuning_official_5k/v05_small_cosine_rank/

PLCC  = 0.582792
SRCC  = 0.578094
Final = 0.579973
```

The temporary tuning folder is cleaned to keep only JSON reports for all
versions, the best KD checkpoint, the fair same-architecture baseline
checkpoint, and summary tables.

```text
results/kd_tuning_official_5k/comparison.csv
results/kd_tuning_official_5k/comparison.md
results/kd_tuning_official_5k/best_summary.json
```

## Best Command

```bash
python scripts/train_official_student_kd.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/train_subset_balanced_5000.csv \
  --save-dir results/kd_tuning_official_5k/v05_small_cosine_rank \
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
  --teacher-rank-weight 0.12
```

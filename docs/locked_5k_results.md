# Locked 5000-Video Official Teacher And Student KD Results

This is the current clean baseline for the thesis after the GCloud L4 run.
Generated result artifacts are intentionally ignored by git and kept locally.

## Local Data

```text
data/train_subset_balanced_5000.csv
data/official_balanced_5000_videos/
```

The local video folder has 5000 real `.mp4` files, 5000 unique stems, and no
missing IDs against the subset CSV.

## Official Teacher Output

```text
results/original_snapugc_official_balanced_5000_artifacts_g2_32/
results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts/
```

The official teacher output contains 200 `.npz` teacher artifact shards and the
full 5000-video prediction CSV.

Full 5000-video teacher metric:

```text
PLCC  = 0.7145597704
SRCC  = 0.7075149240
KTau  = 0.5337330027
MSE   = 0.0423051501
MAE   = 0.1465829997
Final = 0.7103328625
```

## Fair Validation Split

The student script uses a deterministic `4000/1000` split from the same 5000
rows. These are the metrics on the same held-out 1000-video validation split.

```text
Teacher on validation split:
PLCC  = 0.7103010124
SRCC  = 0.6995472898
Final = 0.7038487789

Student baseline:
Best epoch = 4
PLCC       = 0.5066140243
SRCC       = 0.5046699378
Final      = 0.5054475724

Student KD:
Best epoch = 24
PLCC       = 0.5447717600
SRCC       = 0.5417080622
Final      = 0.5429335413

KD gain:
Final = +0.0374859689
```

Student result path:

```text
results/official_student_kd_5000_visual_text/
```

The first student run uses the `visual_text` preset:

```text
Student inputs:
- official frame_fusion_feature
- title text pooled embedding
- description text pooled embedding

KD targets:
- teacher_ecr
- teacher clip_ecr
- teacher temporal_hidden
- teacher fusion_hidden
- teacher attention_importance
```

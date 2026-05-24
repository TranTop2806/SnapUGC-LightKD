# Student KD Architecture

This repo now keeps only two official-artifact student runs:

```text
results/kd_tuning_official_5k/v35_concat_source_embed_v22loss/
results/kd_tuning_official_5k/upper_teacher_compressed_tokens_baseline/
```

## 1. Deployable KD Student

This is the model we can reasonably discuss as the thesis student.

```text
Run    = v35_concat_source_embed_v22loss
Input  = visual_text_sound
Params = 383,652
PLCC   = 0.606979
SRCC   = 0.599926
Final  = 0.602748
```

Input:

```text
frame_fusion_feature        T x 1024
sound-label text embedding  1 x 768
title text embedding        1 x 768
description text embedding  1 x 768
```

The sound input is not raw audio. It is the official teacher's YAMNet sound
labels encoded as text and stored in `text_pooled[0]`.

Architecture:

```text
frame_fusion_feature
  -> Linear(1024, hidden_dim)
  -> 1-layer temporal Transformer
  -> attention pooling
  -> video_hidden

sound/title/description embeddings
  -> Linear(768, hidden_dim)
  -> learned source/type embedding
  -> attention pooling
  -> text_hidden

concat(video_hidden, text_hidden)
  -> fusion MLP
  -> sigmoid ECR head
```

Why the source/type embedding matters:

```text
Before v35:
sound, title, description were pooled as anonymous text vectors.

In v35:
the student learns a small embedding that marks each text source, so the model
can treat sound-label text differently from title and description text.
```

KD objective:

```text
hard_ecr:
  MSE(student_ecr, true_ecr)

soft_ecr:
  MSE(student_ecr, teacher_ecr)

clip_ecr:
  MSE(student_clip_ecr, teacher_clip_ecr)

temporal_hidden:
  cosine(project(student_temporal), teacher_temporal_hidden)

fusion_hidden:
  cosine(project(student_hidden), mean(teacher_fusion_hidden))

attention:
  KL(student_temporal_attention, teacher_attention)

hard_rank:
  pairwise_rank(student_ecr, true_ecr)

teacher_rank:
  pairwise_rank(student_ecr, teacher_ecr)

teacher_pearson / teacher_spearman / teacher_listwise:
  score-shape and listwise ranking distillation from teacher scores
```

This is real knowledge distillation: the student learns from the true label and
from teacher scores, teacher ranking structure, teacher hidden states, and
teacher attention.

## 2. Upper-Bound Compressed Student

This is not a deployable final model by itself. It is a diagnostic upper-bound
showing what happens if the student receives compact teacher-rich tokens.

```text
Run    = upper_teacher_compressed_tokens_baseline
Input  = teacher_compressed_tokens
Params = 712,068
PLCC   = 0.700688
SRCC   = 0.690688
Final  = 0.694688
```

Input:

```text
token 1 = mean/std/min/max of teacher_temporal_hidden
token 2 = mean/std/min/max of teacher_fusion_hidden
token 3 = stats of teacher_clip_ecr and teacher_attention
```

This input is powerful because it compresses internal teacher reasoning. It is
not inference-safe unless we can generate similar tokens without running the
official teacher.

## 3. Enriched Input Experiments & Diagnostic

To test whether the bottleneck is student capacity or input information, three
deployable enriched-input runs were added on top of the v35 base.

| Experiment | Added Input | Baseline Final | KD Final | KD Gain |
|---|---|---:|---:|---:|
| **v35 deployable** (best) | — | 0.5243 | **0.6027** | +0.0784 |
| + OpenCV quality features | blur, brightness, noise, PSNR, contrast, saturation (18-d) | 0.5158 | 0.5843 | +0.0685 |
| + Temporal motion difference | L2/cos/pos between consecutive frames (5-d) | 0.5257 | 0.5995 | +0.0738 |
| + YAMNet raw audio logits | 521-class mean-pooled logits per clip (521-d) | 0.5096 | 0.5929 | +0.0833 |
| Large student (capacity check) | same as v35, hidden=192, layers=2 | — | 0.5901 | — |
| **Feature ceiling** (sklearn Ridge) | deployable + quality flat | — | 0.4910 | — |
| **Upper-bound compressed student** | teacher token stats (non-deployable) | — | **0.6947** | — |
| **Teacher** (full stack) | raw video + all modalities | — | **0.7038** | — |

**Key observations**

- **Enriched inputs do not break the ceiling.** All three cheap-feature runs
  scored **lower** than the plain v35 KD (0.6027). Adding raw YAMNet logits
  (521-d) pushed the student clip dimension from 1024 to 1545 yet hurt both
  baseline and KD performance.
- **Capacity is not the limit.** A larger student (hidden=192, 2 layers,
  ~1.5M params) on the same deployable input scored **0.5901**, worse than
  the smaller v35. More parameters without richer signal only leads to
  overfitting.
- **Feature ceiling aligns with neural results.** The best linear model on
  the full deployable feature stack reached only 0.4910, confirming the
  neural KD student (0.6027) has already extracted most learnable signal
  from this input interface.
- **Information is the bottleneck.** The same small student family reaches
  0.6947 when fed compact teacher-internal tokens. The gap between 0.6027
  and 0.6947 is therefore an **information gap**, not an architecture or
  loss gap.

**Conclusion for thesis writing**

> The deployable input stack `frame_fusion + sound/title/desc` does not
> contain enough of the teacher's internal signal for the student to close
> the remaining ~0.10 gap to the teacher. To beat 0.62 while remaining
> deployable, the student must receive higher-fidelity physical cues
> (e.g. distortion, action, raw caption) at inference time. This requires
> either (a) running part of the teacher stack on-device, or (b) training a
> lightweight dedicated frontend extractor. The trade-off between
> deployability and accuracy is the core limitation of the current design.

## Reproduce

Deployable KD student:

```bash
python3 scripts/train_official_student_kd.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/train_subset_balanced_5000.csv \
  --save-dir results/kd_tuning_official_5k/v35_concat_source_embed_v22loss \
  --input-preset visual_text_sound \
  --hidden-dim 96 \
  --layers 1 \
  --heads 4 \
  --dropout 0.22 \
  --epochs 80 \
  --batch 32 \
  --eval-batch 128 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --val-ratio 0.2 \
  --seed 42 \
  --device mps \
  --run-kind kd
```

Upper-bound compressed student:

```bash
python3 scripts/train_official_student_kd.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/train_subset_balanced_5000.csv \
  --save-dir results/kd_tuning_official_5k/upper_teacher_compressed_tokens_baseline \
  --input-preset teacher_compressed_tokens \
  --hidden-dim 128 \
  --layers 1 \
  --heads 4 \
  --dropout 0.22 \
  --epochs 80 \
  --batch 32 \
  --eval-batch 128 \
  --lr 5e-4 \
  --weight-decay 0.02 \
  --val-ratio 0.2 \
  --seed 42 \
  --device mps \
  --run-kind baseline
```

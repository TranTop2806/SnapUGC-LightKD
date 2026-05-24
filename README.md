# SnapUGC-LightKD

Thesis code for a bounded 5000-video SnapUGC knowledge-distillation study.

The current main experiment uses the authors' released SnapUGC EVQA teacher
from `dasongli1/SnapUGC_Engagement` on one fixed, balanced 5000-video subset.
Local code prepares the subset, runs/monitors the official teacher on GCloud,
exports teacher artifacts for KD, and keeps student/baseline experiments
separate from the teacher reproduction.

## Current Pipeline

```text
Balanced 5000-video subset
  -> official SnapUGC EVQA teacher inference on GCloud L4
     - EfficientNetV2-s semantic frame features
     - UVQ-style distortion features
     - ResNet3D-18 action features
     - mPLUG-2 caption and clip features
     - YAMNet sound labels
     - Stable Diffusion text encoder
     - EVQA.pth ECR head
  -> scalar teacher ECR + hidden/attention/artifact shards
  -> lightweight student baseline and KD experiments
```

The official teacher architecture is not reimplemented in
`src/snapugc_lightkd/models.py`. A pinned local copy of the real teacher source
lives in `third_party/SnapUGC_Engagement/ECR_inference/`. Runtime code patches
a working copy only for compatibility/artifact export. See
`docs/original_snapugc_exact_reproduction.md` for file-level architecture
details, and `docs/gcp_official_inference_artifacts_readme.md` for the concrete
Google Cloud runbook used to run official inference and export teacher artifact
shards.

## Dataset And Split

The thesis uses one locked balanced 5000-video subset:

```text
data/train_subset_balanced_5000.csv
data/official_balanced_5000_videos/
data/official_5k_split/
```

The zip archive below contains the 5000 videos, labels, and fixed split files:

```text
data/official_snapugc_5k_locked_dataset.zip
```

Student experiments use a deterministic `4000/1000` split:

```text
seed = 42
val_ratio = 0.2
rows = official teacher artifact rows sorted by order_idx
test/val = first 1000 indices after np.random.default_rng(42).shuffle(...)
train = remaining 4000 rows
```

Split files:

```text
data/official_5k_split/train_4000.csv
data/official_5k_split/test_1000.csv
data/official_5k_split/split_all_5000.csv
data/official_5k_split/manifest.json
```

In the code, the 1000 held-out rows are named `val` because they are used for
model selection. In thesis writing, call them a fixed validation/test split and
state the split protocol clearly.

## Repository Structure

```text
SnapUGC-LightKD/
  data/                    # local 5k subset/videos; ignored by git
  docs/                    # reproduction notes and locked result summaries
  notebooks/               # official Kaggle notebook kept for reproducibility
  results/                 # generated reports/checkpoints; ignored by git
  scripts/
    extract_clip_keyframe_features.py   # extract CLIP ViT-B/32 keyframe embeddings from video tar
    train_official_student_kd.py        # student baseline and KD training
    evaluate_student_ensemble.py        # ensemble evaluation over multiple checkpoints
    run_official_snapugc_evqa.py        # official teacher inference wrapper
    make_subset.py                      # create balanced 5k video subset
  src/snapugc_lightkd/     # student artifact dataset/model helpers
  third_party/             # pinned official teacher source, no checkpoints
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On GCloud/L4, install the CUDA-compatible PyTorch wheel before running the
official teacher wrapper.

## Model Inputs And Outputs

This project has three model roles:

```text
1. Official teacher
   Input: raw video + title + description
   Output: teacher ECR plus artifact shards for KD

2. Student baseline
   Input: reduced teacher-extracted artifacts only
   Output: student ECR
   Loss: ground-truth ECR only

3. Student KD
   Input: exactly the same reduced inputs as student baseline
   Output: student ECR plus auxiliary student artifacts
   Loss: ground-truth ECR + teacher ECR/artifact/ranking distillation
```

The deployable student never sees the full privileged teacher input stack at
inference time. The retained deployable preset is `visual_text_sound` with
an optional CLIP semantic branch (best configuration):

```text
Student input preset: visual_text_sound
- frame_fusion_feature: T x 1024          (EfficientNetV2-s + UVQ, from teacher artifacts)
- YAMNet top sound labels text emb: 1 x 768
- title pooled text embedding:     1 x 768
- description pooled text emb:     1 x 768

[Optional, deployable] CLIP ViT-B/32 keyframe embeddings:
- quality_features: T x 512               (CLIP image encoder on 16 uniform keyframes)
- quality_fusion: clip_add                (added into hidden space after temporal encoder)
```

Learned source/type embeddings distinguish sound labels, title, and description
before text pooling. The `clip_add` fusion adds CLIP embeddings into the
hidden representation after temporal encoding — acting as late semantic gating.

The teacher artifacts available for KD are:

```text
teacher_ecr: scalar
teacher_clip_ecr: T
teacher_temporal_hidden: T x 512
teacher_fusion_hidden: T x 512
teacher_attention_importance: attention_layers x T
```

## Official Teacher Run

The primary GCloud runner is:

```bash
SHUTDOWN_ON_EXIT=1 \
ROOT_DIR=/workspace/SnapUGC-LightKD \
SUBSET_CSV=/workspace/snapugc-data/train_subset_balanced_5000.csv \
VIDEO_DIR=/workspace/snapugc-data/train_videos_balanced_5000 \
OUT_DIR=/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32 \
CHECKPOINT_DIR=/workspace/snapugc-checkpoints \
EXPORT_ARTIFACTS=1 \
ARTIFACT_SHARD_SIZE=25 \
bash scripts/run_gcp_official_balanced_5k_from_links.sh
```

The run writes partial reports every 500 predictions and artifact shards every
25 videos:

```text
official_partial_500_predictions.csv
official_partial_500_report.json
teacher_artifacts/official_teacher_artifacts_0000_0024.npz
teacher_artifacts/official_teacher_artifacts_0000_0024_captions.jsonl
```

Sync outputs back to local:

```bash
PROJECT=snapugc-lightkd \
ZONE=asia-southeast1-a \
INSTANCE=snapugc-l4-artifacts \
REMOTE_OUT_DIR=/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32 \
LOCAL_OUT_DIR=results/original_snapugc_official_balanced_5000_artifacts_g2_32 \
bash scripts/sync_gcp_official_5k_outputs.sh
```

## Official Teacher Architecture

The teacher is the released SnapUGC EVQA inference stack from the original
authors. It is heavyweight because it combines multiple pretrained components:

```mermaid
flowchart TD
    V["Raw SnapUGC video"] --> F1["Sample frames/clips across video"]
    M["Metadata: title + description"] --> T0["Text streams"]

    F1 --> E1["EfficientNetV2-s semantic branch<br/>per-frame 528d"]
    F1 --> D1["UVQ distortion branch<br/>per-frame 256d"]
    F1 --> A1["ResNet3D-18 action branch<br/>per-clip 512d"]
    F1 --> C1["mPLUG-2 video caption branch<br/>caption + 1024d clip feature"]
    V --> S1["YAMNet sound labels<br/>top audio labels as text"]

    M --> T0
    S1 --> T0
    C1 --> T0
    T0 --> SD["Stable Diffusion tokenizer/text encoder<br/>77 x 768 per stream"]

    E1 --> G1["Frame/clip grouping + FC"]
    D1 --> G1
    A1 --> X1["Cross-attention with text streams"]
    SD --> X1
    C1 --> X1

    G1 --> FUS["EVQA multimodal fusion"]
    X1 --> FUS
    FUS --> TR["8 TransformerBlock temporal self-attention"]
    TR --> HEAD["ECR output head"]
    HEAD --> OUT["Teacher ECR"]
    TR --> ART["Hidden/attention/artifact export for KD"]
```

Teacher output files:

```text
official_submission_baseline.csv      # Id, ECR prediction
official_evqa_report.json             # PLCC, SRCC, final score, MSE, MAE
teacher_artifacts/*.npz               # hidden states, clip outputs, attention
teacher_artifacts/*_captions.jsonl    # generated captions
```

The official teacher is inference-only in this thesis: we use the released
checkpoints and do not retrain the original teacher.

## Student Work

Student baseline and KD are implemented separately from the official teacher.
They train from synced teacher artifact shards using reduced student inputs.

```mermaid
flowchart TD
    A["Official teacher artifact shards"] --> B["Allowed student inputs<br/>default: frame fusion + title/description text"]
    A --> T1["Teacher ECR"]
    A --> T2["Teacher clip ECR"]
    A --> T3["Teacher hidden states"]
    A --> T4["Teacher temporal attention"]
    B --> S1["Compact student<br/>projection + small temporal Transformer"]
    S1 --> S2["Student ECR"]
    S1 --> S3["Student clip ECR / hidden / attention"]
    S2 --> L1["Baseline loss<br/>MSE(true ECR)"]
    S2 --> L2["KD loss<br/>true ECR + teacher ECR"]
    S3 --> L2
    T1 --> L2
    T2 --> L2
    T3 --> L2
    T4 --> L2
```

### Student Architecture (Best Single Model)

The best deployable student adds CLIP ViT-B/32 keyframe embeddings via `clip_add`
fusion into the hidden space after temporal encoding:

```mermaid
flowchart TD
    VF["frame_fusion_feature\nT x 1024"] --> VP["Linear + LayerNorm + GELU\n1024 -> hidden_dim=96"]
    VP --> PE["Positional embedding"]
    PE --> TE["Temporal Transformer (1 layer, 4 heads)"]
    TE --> AP["Attention pooling -> video_pooled (96-d)"]

    CL["CLIP ViT-B/32 keyframe embeddings\nT x 512"] --> CP["Linear + LayerNorm -> 96-d"]
    CP --> CAP["Attention pooling -> clip_pooled (96-d)"]
    CAP -- clip_add --> AP

    TX["sound + title + description pooled\n3 x 768"] --> TP["Text projection + source embedding\n768 -> 96-d"]
    TP --> TAP["Text attention pooling -> text_pooled (96-d)"]

    AP --> CAT["Concat video_pooled + text_pooled (192-d)"]
    TAP --> CAT
    CAT --> FM["Fusion MLP (192 -> 96 -> 96)"]
    FM --> HAL["Hallucination heads (train only)\npredict teacher action/caption embeddings"]
    FM --> EH["Sigmoid ECR head"]
    EH --> ECR["Student ECR"]
```

Hyperparameters (tuned compact student):

```text
hidden_dim = 96
Transformer layers = 1
heads = 4
dropout = 0.25
max_clips = 16
quality_fusion = clip_add    # key: add CLIP into hidden, not input concat
```

Baseline training objective:

```text
loss_baseline = MSE(student_ecr, true_ecr)
```

### Student KD Architecture And Loss

The KD student uses the same input and architecture as its baseline counterpart.
It adds auxiliary heads/projections only during training:

```text
student_clip_ecr: per-clip scalar predictions
project(student_temporal): T x 512 teacher-space tokens
project(student_hidden): 512 teacher-space pooled hidden state
student_temporal_attention: T
```

KD objective:

```text
loss_kd =
  hard_ecr      * MSE(student_ecr, true_ecr)
+ soft_ecr      * MSE(student_ecr, teacher_ecr)
+ clip_ecr      * MSE(student_clip_ecr, teacher_clip_ecr)
+ temporal      * repr_loss(project(student_temporal), teacher_temporal_hidden)
+ fusion        * repr_loss(project(student_hidden), mean(teacher_fusion_hidden))
+ attention     * KL(student_temporal_attention, teacher_attention_importance)
+ hard_rank     * pairwise_rank(student_ecr, true_ecr)
+ teacher_rank  * pairwise_rank(student_ecr, teacher_ecr)
```

`repr_loss` supports raw MSE, normalized MSE, and cosine distance. The tuned
model uses cosine representation KD because raw hidden-state MSE was too large
in scale and dominated scalar ECR/ranking losses.

### Best Training Command (CLIP clip_add + Curriculum + Hallucination)

Step 1 — Extract CLIP ViT-B/32 keyframe features from the video archive:

```bash
python scripts/extract_clip_keyframe_features.py \
  --tar results/videos_5000.tar \
  --labels-csv data/train_subset_balanced_5000.csv \
  --out results/clip_vitb32_keyframe_features_5000.npz \
  --model ViT-B-32 --pretrained openai \
  --n-frames 16 --device mps
```

Step 2 — Train the student with CLIP `clip_add` fusion:

```bash
python scripts/train_official_student_kd.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/train_subset_balanced_5000.csv \
  --save-dir results/kd_tuning_official_5k/improve_clip_vitb32_clipadd_curriculum_halluc \
  --input-preset visual_text_sound \
  --quality-features results/clip_vitb32_keyframe_features_5000.npz \
  --quality-fusion clip_add \
  --use-hallucination --hallucination-feedback --feedback-start-epoch 10 \
  --hidden-dim 96 --layers 1 --heads 4 \
  --dropout 0.25 --epochs 100 --batch 32 --eval-batch 128 \
  --lr 5e-4 --weight-decay 0.02 \
  --val-ratio 0.2 --seed 42 --device mps --run-kind kd \
  --kd-curriculum three_phase \
  --soft-weight 1.1 --clip-weight 0.08 \
  --temporal-weight 0.02 --fusion-weight 0.02 --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 --teacher-listwise-weight 0.02 \
  --student-teacher-relation-weight 0.02 --contrastive-hidden-weight 0.02 \
  --action-hallucination-weight 0.03 --caption-hallucination-weight 0.05
```

See `docs/student_kd_architecture.md` for the full student KD diagram,
input presets, loss terms, and ablation results.

The report metric is `final_score = 0.6 * SRCC + 0.4 * PLCC`.

## Retained 5k Results

All retained runs live under `results/kd_tuning_official_5k/`.

```text
Fair 1000-video validation split (seed=42, val_ratio=0.2):
Official teacher (upper bound)          Final = 0.7038
Compressed teacher-token student        Final = 0.6947   (not deployable)
```

| Run | Input | PLCC | SRCC | Final | Note |
|---|---|---:|---:|---:|---|
| `improve_clip_vitb32_clipadd_curriculum_halluc` | `visual_text_sound` + CLIP B/32 | 0.6325 | 0.6259 | **0.6285** | ✅ best deployable single model |
| `clip_clipadd_curriculum_halluc_seed43` | same | 0.6273 | 0.6191 | 0.6223 | stability seed |
| `clip_clipadd_curriculum_halluc_seed44` | same | 0.6243 | 0.6183 | 0.6207 | stability seed |
| `improve_clip_vitb32_clip_add_e100` | `visual_text_sound` + CLIP B/32 | 0.6253 | 0.6175 | 0.6206 | clip_add, no curriculum |
| `improve_strat3_hybrid_curriculum` | `visual_text_sound` + X3D-XS | 0.6079 | 0.6012 | 0.6039 | prev best before CLIP |
| `v35_concat_source_embed_v22loss` | `visual_text_sound` | 0.6070 | 0.5999 | 0.6027 | baseline (no extra features) |
| `v35_teacher_action_caption_clipadd_kd` | `visual_text_sound` + teacher feats | 0.6510 | 0.6432 | 0.6463 | not deployable (uses teacher feats at inference) |
| `upper_teacher_compressed_tokens_baseline` | teacher compressed tokens | 0.7007 | 0.6907 | 0.6947 | not deployable |

### Why CLIP `clip_add` works

`frame_fusion_feature` (EfficientNetV2-s + UVQ) is optimised for visual quality
estimation but lacks **semantic content richness** — what the video is *about*.
CLIP ViT-B/32, trained on 400 M image–text pairs, supplies exactly that missing
dimension (objects, activities, aesthetics, mood) without requiring teacher
features at inference.

Critically, `clip_add` (add CLIP into the hidden state after temporal encoding)
outperforms `input_concat` (prepend CLIP to the raw clip features):

| Fusion | clip_input_dim | Final |
|---|---|---|
| `input_concat` | 1536 (1024+512) | 0.6042 |
| `clip_add` | 1024 (unchanged) | **0.6285** |

With `input_concat` the temporal Transformer must align two distributions
(EfficientNetV2-s vs CLIP) from scratch at input, which is hard. With `clip_add`
the Transformer first learns temporal dynamics on the familiar 1024-d stream,
then CLIP enriches the resulting pooled hidden vector — a much easier learning
problem analogous to a late cross-modal residual.

`visual_text_sound` does not use raw audio embeddings. It uses the same
audio-derived representation as the official teacher: YAMNet top-5 sound labels
encoded by the Stable Diffusion text encoder, stored as `text_pooled[0]`.

See `docs/student_kd_architecture.md` for full details.

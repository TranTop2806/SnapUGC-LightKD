# SnapUGC-LightKD Pipeline And Literature Comparison

Last updated: 2026-04-19

## 1. Current Pipeline

Current repository: `SnapUGC-LightKD`.

High-level flow:

```text
Raw video + title + description + ECR
  -> ECR-stratified subset selection
  -> DOVER-Mobile scoring
  -> feature extraction
       - CLIP ViT-B/16 frame tokens
       - R(2+1)D-18 Kinetics-400 motion tokens
       - DOVER technical/aesthetic/overall scores
       - YAMNet AudioSet audio embeddings/classes
       - Sentence-T5-base metadata/caption/rationale embeddings
       - BLIP-base frame captions
  -> Teacher training
  -> Teacher soft targets / hidden / temporal attention
  -> Student baseline training
  -> Student KD training
  -> report + diagnostic plots + zip packaging without `subset_videos/`
```
## 2. Feature Extraction Stack

| Component | Current model | Output | Role | Alignment / reason |
|---|---|---:|---|---|
| Visual semantics | `openai/clip-vit-base-patch16` | per-frame 512d tokens + mean | Semantic visual representation | Aligns with challenge teams using CLIP-like visual semantic features; lightweight and Kaggle-compatible. |
| Motion/action | `torchvision r2plus1d_18`, Kinetics-400 | per-clip 512d tokens + mean | Temporal/action cue | Aligns with original SnapUGC motivation that temporal/action context matters; lightweight substitute for larger video encoders/LMM video branches. |
| Video quality | DOVER-Mobile precomputed CSV | technical/aesthetic/overall | UGC quality cue | Strongly aligned with VQualA/MMF-QE, where DOVER technical/aesthetic branches are explicitly used. Current uses DOVER-Mobile because full DOVER/LMM branches are expensive. |
| Audio | YAMNet from TensorFlow Hub | 1024d embedding + AudioSet class probs | Background sound/music/speech cue | Aligns with Sun/ECNU result that audio improves engagement prediction; lightweight substitute for VideoLLaMA2 audio branch or BEATs. |
| Metadata text | `sentence-transformers/sentence-t5-base` | 768d | title/description/hashtag semantic cue | Aligns with Original SnapUGC, which uses text/title/description signals and T5-family text encoding. |
| Captioning | `Salesforce/blip-image-captioning-base` | frame captions + 768d Sentence-T5 caption embedding | Semantic visual description | Hardware-driven substitute. Related work uses heavier video captioning/LVLMs/Qwen/VideoLLaMA; BLIP is used because Qwen2.5-VL/VideoLLaMA2 were too heavy/unstable on Kaggle T4 for this pipeline. |
| Rationale text | template from caption + metadata + DOVER + audio top labels | 768d Sentence-T5 embedding | Teacher-only semantic summary | Thesis-design addition; not a direct paper component, but tries to preserve an interpretable multimodal signal without running a large reasoning VLM. |

## 3. Teacher Architecture

Current teacher configuration in notebook:

```text
teacher_hidden = 512
teacher_blocks = 2
teacher_heads = 8
max_frames = 16
max_motion_clips = 4
dropout = 0.1
```

Inputs:

- CLIP frame sequence: `16 x 512`
- R(2+1)D motion clip sequence: `4 x 512`
- YAMNet audio embedding: `1024`
- metadata text embedding: `768`
- BLIP caption embedding: `768`
- rationale embedding: `768`
- DOVER quality scores: `3`

Structure:

```text
CLIP frames -> TemporalEncoder -> visual token
Motion clips -> TemporalEncoder -> motion token
Audio -> projector -> audio token
Metadata text -> projector -> text token
Caption -> projector -> caption token
Rationale -> projector -> rationale token
DOVER scores -> projector -> quality token

7 modality tokens + modality embeddings
  -> Transformer fusion encoder
  -> attention pooling
  -> residual MLP blocks
  -> ECR head
  -> aesthetic auxiliary head
  -> technical auxiliary head
```

Parameter count from current reports:

| Model | Params |
|---|---:|
| Teacher | `20,779,526` |

## 4. Student Architecture

Current student configuration in notebook:

```text
student_hidden = 256
student_heads = 4
max_frames = 16
dropout = 0.1
```

Inputs:

- CLIP frame sequence: `16 x 512`
- YAMNet audio embedding: `1024`
- metadata text embedding: `768`

Teacher-only signals not given directly to the student:

- motion tokens
- DOVER quality scores
- BLIP caption embedding
- rationale embedding

Structure:

```text
CLIP frames -> TemporalEncoder -> visual token
Audio -> projector -> audio token
Metadata text -> projector -> text token

3 modality tokens + modality embeddings
  -> Transformer fusion encoder
  -> attention pooling
  -> residual MLP blocks
  -> ECR head
  -> aesthetic auxiliary head
  -> technical auxiliary head
  -> KD projector to teacher hidden size
```

Parameter count from current reports:

| Model | Params |
|---|---:|
| Student baseline | `4,548,613` |
| Student KD | `4,548,613` |

Compression ratio:

```text
20.78M / 4.55M ~= 4.6x
```

## 5. KD Objective

Student KD uses a mixed objective:

```text
L = 1.0 * MSE(student_ecr, true_ecr)
  + 0.5 * MSE(student_ecr, teacher_ecr)
  + 0.3 * cosine_hidden_distillation(student_hidden, teacher_hidden)
  + 0.1 * KL(student_temporal_attention, teacher_temporal_attention)
  + 0.2 * MSE(student_aesthetic, DOVER_aesthetic)
  + 0.2 * MSE(student_technical, DOVER_technical)
```

Current KD weights from report:

| Term | Weight |
|---|---:|
| hard ECR | `1.0` |
| soft teacher ECR | `0.5` |
| hidden representation KD | `0.3` |
| temporal attention KD | `0.1` |
| aesthetic auxiliary | `0.2` |
| technical auxiliary | `0.2` |


## 6. Current Results

| System | Scale / setup | PLCC | SRCC/SROCC | Final |
|---|---|---:|---:|---:|
| Current SnapUGC-LightKD 2k balanced, Teacher | 1600/400 split | `0.409` | `0.426` | `0.419` |
| Current SnapUGC-LightKD 2k balanced, Student baseline | 1600/400 split | `0.444` | `0.456` | `0.451` |
| Current SnapUGC-LightKD 2k balanced, Student KD | 1600/400 split | `0.474` | `0.477` | `0.476` |
| Current 2k balanced, tabular Ridge on teacher features | 1600/400 split | `0.483` | `0.473` | `0.477` |
| Old Distil-ShortVU 5k, Student KD | 4000/1000 split | `0.551` | `0.545` | `~0.548` |
| Old Distil-ShortVU 100k, Student KD | 84011/21003 split | `0.709` | `0.690` | `~0.697` |
| Original SnapUGC paper, Ours ECR | full paper setting | `0.688` | `0.675` | `~0.680` |
| VQualA challenge baseline | challenge test | `0.665` | `0.657` | `0.660` |
| Sun/ECNU Qwen2.5-VL | challenge-style setting | `0.662` | `0.665` | `0.664` |
| Sun/ECNU VideoLLaMA2 | challenge-style setting | `0.701` | `0.691` | `0.695` |
| Sun/ECNU ensemble top-1 | challenge test | `0.714` | `0.707` | `0.710` |
| Guan/MMF-QE top co-first | challenge test | `0.702` | `0.696` | `0.698` |

Notes:

- Final score uses `0.6 * SROCC/SRCC + 0.4 * PLCC`.

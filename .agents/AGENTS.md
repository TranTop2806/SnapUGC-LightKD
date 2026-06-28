# AI Instructions for SnapUGC-LightKD Project

This file serves as a system prompt addition or context guide for AI agents working on the SnapUGC-LightKD codebase.

---

## 1. Project Overview & Objective

* **Project Name**: SnapUGC-LightKD
* **Goal**: Research and implement Knowledge Distillation (KD) to compress a heavyweight Video Quality/Engagement Assessment model (Teacher) into a lightweight, edge-deployable model (Student) using a bounded subset of 5,000 Snapchat User-Generated Content (UGC) videos.
* **Task Type**: Subjective video quality regression (predicting Engagement Continuation Ratio - ECR).

---

## 2. Core Architecture Roles

### A. Teacher Model (EVQA)
* **Status**: Frozen during student training (inference-only).
* **Size**: `~1,801.7M` (1.80 Billion) parameters.
* **Components**: 
  - Visual: `EfficientNetV2-S` (semantic frames) + `UVQ` (distortion) + `ResNet3D-18` (action clips).
  - Language: `mPLUG-2` (video captions) + `Stable Diffusion v1.4` (text encoder).
  - Audio: `YAMNet` (sound class labels as text).
  - Temporal Fusion: Multimodal fusion layer + 8 Transformer Blocks.
* **Output**: Scalar teacher ECR, intermediate hidden states, attention matrices, and caption shards.

### B. Student Model (Compact Transformer)
* **Status**: Trainable, compact, and edge-deployable.
* **Size**: Tiny head (`378K` parameters pruned / `433K` total for baseline; `1.15M` pruned for large KD).
* **Backbone**: 1-layer, 4-heads, 96-d hidden dimension temporal Transformer.
* **Inputs**: Restrictive/lightweight features, bypassing mPLUG-2 and ResNet3D-18.

---

## 3. Two Deployment Configurations (Presets)

AI models must strictly distinguish between these two modes when coding, setting presets, or interpreting results:

### A. Semi-independent / Head Distillation (Preset: `visual_text_sound`)
* **Features**: Frame-level EfficientNetV2-S + UVQ features (pre-extracted by the teacher) + CLIP keyframe features + Metadata Text.
* **Target Scores**: PLCC/SRCC ≈ `0.62`.
* **Inference Footprint**: ~2.5GB VRAM, ~2.2s per video.
* **Inference Dependency**: Still requires pre-extracted feature tensors from the teacher's backbones. Best when a shared feature pipeline already exists.

### B. Proper / Full Pipeline KD (Preset: `clip_mobilenet_text`)
* **Features**: CLIP ViT-B/32 + MobileNetV3-Small (running directly on raw video frames) + Metadata Text.
* **Target Scores**: PLCC/SRCC ≈ `0.57`.
* **Inference Footprint**: Ultra-lightweight, runs standalone on edge devices from raw video.
* **Inference Dependency**: Fully independent of the teacher's pipeline.

---

## 4. Dataset & Split Protocol

* **Total Samples**: Bounded balanced 5000-video subset.
* **Split**: Deterministic `4000` train and `1000` validation/test split.
* **Split Seed**: `42` (must not be altered to ensure reproduction consistency).
* **Metrics**:
  - `PLCC` (Pearson Linear Correlation Coefficient)
  - `SRCC` (Spearman Rank Correlation Coefficient)
  - `Final Score` = `0.6 * SRCC + 0.4 * PLCC`

---

## 5. Knowledge Distillation Loss Functions

The KD training objective optimizes a combination of several loss terms:
1. **Hinton Logit KD (`soft_ecr`, `clip_ecr`)**: Mimicking the teacher's continuous quality ratings.
2. **FitNets Feature Alignment (`temporal`, `fusion`)**: Cosine representation alignment of intermediate layers via projection heads.
3. **Attention Transfer (`attention`)**: KL Divergence matching of temporal attention weights.
4. **Pairwise Ranking Loss (`hard_rank`, `teacher_rank`)**: Optimizing relative order.
5. **Privileged Information / Hallucination Loss (`action_hallucination`, `caption_hallucination`)**: Student predicts action and caption features during training only (auxiliary heads are discarded at inference).

---

## 6. Repository Layout & Main Files

When modifying code or analyzing results, focus on:
* **`scripts/train_official_student_kd.py`**: Main entrypoint for student training (baseline and KD runs).
* **`src/snapugc_lightkd/`**: Core codebase containing dataset, model, losses, and training loops.
  - `src/snapugc_lightkd/dataset.py`: Multi-modal feature dataset loader.
  - `src/snapugc_lightkd/official_student.py`: The student model architecture definition.
  - `src/snapugc_lightkd/losses.py`: Distillation and task loss implementations.
* **`results/`**: Training reports, checkpoints, and evaluation metrics (`official_student_kd_report.json`).
* **`data/`**: Subsets and metadata.

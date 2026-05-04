# Exact Paper Reproduction Choice: LMM-EVQA VideoLLaMA2

Date: 2026-05-04

## Decision

Use the VideoLLaMA2.1-7B-AV branch from Sun et al., *Engagement Prediction of Short Videos with Large Multimodal Models*, as the exact paper architecture to reproduce on the 5,000-video subset.

This is the fairest next experiment because the data scale differs from the papers, so the architecture should be held fixed. Among the four references, this is the strongest option with official code that can be run without relying on the broken original SnapUGC EfficientNetV2 download.

## Why This Paper

| Reference | Exact reproduction status | Reason |
|---|---|---|
| Original SnapUGC / Li et al. | Blocked | Official code requires `EVQA.pth`, `net_distort6_g_latest.pth`, `r3d18_K_200ep.pth`, `mPLUG2_MSRVTT_Caption.pth`, `ViT-L-14.tar`, and an additional EfficientNetV2 weight from a dead OneDrive URL. Replacing that weight is no longer exact. |
| VQualA challenge overview | Not an architecture | It summarizes methods and baseline results, but the baseline is still the original SnapUGC system. |
| Guan et al. MMF-QE | Not fully reproducible from paper alone | It combines the original SnapUGC branch, DOVER fine-tuning, and Qwen2.5-VL SFT. The original branch inherits the blocked artifact. |
| Sun et al. LMM-EVQA | Selected | Official GitHub provides VideoLLaMA2 training code, data format, and train command. It avoids the broken EfficientNetV2 artifact and is a strong teacher candidate. |

## Exact Architecture To Run

Selected branch: `VideoLLaMA2-audio_visual`.

Official code commit used by our runner:

```text
sunwei925/LMM-EVQA@b3434ee576ad42d5141be8d6c5c45734a9313794
```

Core settings preserved from the official `train.sh`:

```text
model_type: videollama2_qwen2
base model path: VideoLLaMA2.1-7B-AV weights
vision tower: google/siglip-so400m-patch14-384
audio tower: audio_tower.bin
audio projector: mm_projector_a.bin
mm_projector_type: stc_connector_v35
mm_projector_a_type: mlp2x_gelu
modalities: video + audio + title/description text
num_frames: 8
video sampling: first_n_seconds in official train_EVQA.py
loss: MSE on ECR scaled to 0-100
epochs: 1
learning_rate: 5e-5
bf16: true
tf32: true
gradient_checkpointing: true
per_device_train_batch_size: 6
gradient_accumulation_steps: 4
```

The official prompt is preserved:

```text
<video>
How would you judge the engagement continuation rate of the given content, where engagement continuation rate represents the probability of watch time exceeding 5 seconds. The title of the video is {title}, and the description of the video is {description}
```

## Files Added

```text
scripts/prepare_lmm_evqa_videollama2_data.py
scripts/run_lmm_evqa_videollama2_5000.sh
scripts/train_student_from_prediction_teacher.py
```

## Cloud Run

Prepare data:

```bash
python scripts/prepare_lmm_evqa_videollama2_data.py \
  --csv /path/to/train_data.csv \
  --video-root /path/to/train_videos \
  --out-dir /workspace/snapugc_lmm_evqa_5000 \
  --max-samples 5000 \
  --val-ratio 0.2 \
  --seed 42
```

Run exact teacher:

```bash
REPO_DIR=/workspace/LMM-EVQA \
DATA_DIR=/workspace/snapugc_lmm_evqa_5000 \
MODEL_ROOT=/workspace/videollama2weights \
OUTPUT_DIR=/workspace/videollama2_evqa_snapugc_5000_mse \
bash scripts/run_lmm_evqa_videollama2_5000.sh
```

The runner saves:

```text
${OUTPUT_DIR}/eval/val_submission.csv
${OUTPUT_DIR}/eval/all_submission.csv
```

Use `val_submission.csv` for the teacher metric row and `all_submission.csv` for soft-label KD.

Recommended hardware for paper-faithful execution:

```text
2 x A100 80GB or 2 x A800
```

The paper used 2 A800 GPUs for the VideoLLaMA2 branch. A single A100 80GB may run only after reducing batch settings, but that changes the training recipe, so it should be treated as a cost-saving ablation rather than the exact run.

## KD Plan

After the teacher finishes:

1. Keep the official validation metrics as the teacher result.
2. Use `${OUTPUT_DIR}/eval/all_submission.csv` as soft ECR predictions for all 5,000 samples.
3. Train the lightweight student using the same train/val split and soft-label KD from the LMM teacher predictions.
4. Report three rows:
   - exact LMM-EVQA teacher on 5k
   - lightweight student baseline on the same 5k split
   - lightweight student + KD from exact LMM-EVQA teacher

This makes the thesis fair: the teacher is a reproduced paper architecture, while the KD student is our compression contribution.

KD command after copying `all_submission.csv` back to this repo:

```bash
python scripts/train_student_from_prediction_teacher.py \
  --features results/opening5_teacher_5000/features_opening5_5000.json \
  --teacher-preds /path/to/all_submission.csv \
  --split-csv /path/to/split.csv \
  --save-dir results/lmm_evqa_videollama2_teacher_kd_5000 \
  --teacher-scale 100 \
  --student-hidden 64 \
  --student-heads 2 \
  --max-frames 4 \
  --no-student-audio \
  --no-student-text \
  --epochs 60 \
  --device cuda
```

## Sources

- Sun et al., *Engagement Prediction of Short Videos with Large Multimodal Models*: https://openaccess.thecvf.com/content/ICCV2025W/VQualA/papers/Sun_Engagement_Prediction_of_Short_Videos_with_Large_Multimodal_Models_ICCVW_2025_paper.pdf
- Official LMM-EVQA repository: https://github.com/sunwei925/LMM-EVQA
- Original SnapUGC repository: https://github.com/dasongli1/SnapUGC_Engagement
- VQualA challenge overview: https://arxiv.org/abs/2509.02969

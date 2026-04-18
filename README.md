# SnapUGC-LightKD

Literature-aligned lightweight knowledge distillation for SnapUGC short-video engagement prediction.

This is the clean thesis repo. It intentionally contains only the final pipeline, not the earlier exploratory/v1 code.

## Pipeline

```text
Video + metadata
  |-- CLIP ViT-B/16 frame tokens
  |-- R(2+1)D-18 Kinetics-400 motion tokens
  |-- DOVER technical/aesthetic/overall quality scores
  |-- YAMNet AudioSet background-sound embedding/classes
  |-- Sentence-T5-base metadata/caption/rationale embeddings
  |-- BLIP-base lightweight frame captions
  |
Teacher: visual + motion + quality + audio + metadata + caption + rationale
Student: visual + audio + metadata
KD: prediction + hidden representation + temporal attention + quality auxiliary heads
```

## Repository Structure

```text
SnapUGC-LightKD/
  configs/                 # documented run configs
  data/                    # local symlinks or mounted dataset; ignored by git
  docs/                    # paper/model justification
  notebooks/               # Kaggle notebook for 1000-video run
  results/                 # generated reports/checkpoints; ignored by git
  scripts/                 # CLI wrappers
  src/snapugc_lightkd/     # package code
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[extraction,notebook]"
```

If installing PyTorch manually for Kaggle/CUDA, install the matching wheel first, then run `pip install -e ".[extraction,notebook]"`.

## Feature Extraction

First run, 1000 videos:

```bash
python scripts/extract_features.py \
  --csv data/train_data.csv \
  --videos data/train_videos \
  --out features/features_final_1000.json \
  --max 1000 \
  --dover-csv path/to/dover_scores.csv
```

For CPU smoke test:

```bash
python scripts/extract_features.py \
  --csv data/train_data.csv \
  --videos data/train_videos \
  --out features/features_smoke.json \
  --max 20 \
  --skip-caption --skip-motion --skip-audio
```

## Training

```bash
python scripts/train.py \
  --data features/features_final_1000.json \
  --save-dir results/final_1000 \
  --device cuda
```

Quick smoke test:

```bash
python scripts/make_synthetic.py --out features/synthetic_v2.json --n 64
python scripts/train.py \
  --data features/synthetic_v2.json \
  --save-dir results/smoke \
  --device cpu \
  --quick \
  --teacher-hidden 128 \
  --student-hidden 64 \
  --teacher-blocks 1
```

## Main Outputs

```text
features/features_final_1000.json
results/final_1000/final_teacher_best.pth
results/final_1000/final_student_baseline_best.pth
results/final_1000/final_student_kd_best.pth
results/final_1000/final_experiment_report.json
```

The report includes PLCC, SRCC, KRCC, MSE, MAE, and `final_score = 0.6 * SRCC + 0.4 * PLCC`.

## Run on Kaggle from GitHub

Create a GitHub repo named `snapugc-lightkd`, push this repository, then run `notebooks/kaggle_final_1000.ipynb`. The first notebook cell clones:

```bash
git clone https://github.com/TranTop2806/SnapUGC-LightKD.git /kaggle/working/SnapUGC-LightKD
```

Update `GITHUB_REPO` in the notebook if the GitHub URL is different.

## Kaggle

Use `notebooks/kaggle_final_1000.ipynb`. Despite the legacy filename, it now defaults to a bounded 300-video run with DOVER-Mobile and BLIP-base captioning, then trains Teacher/Student/KD and zips the outputs. Increase `MAX_VIDEOS` only after the 300-video run finishes successfully.

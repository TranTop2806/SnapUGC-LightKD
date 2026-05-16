# Google Cloud README: Official SnapUGC Inference And Teacher Artifact Export

This README documents the exact Google Cloud workflow used in this thesis repo
to run the original SnapUGC paper code and export teacher artifacts for KD.

Terminology note: this document says **inference**. If older notes say
`interface`, they mean running the official teacher forward pass.

## Goal

Run the original paper teacher on the locked 5000-video subset:

```text
official SnapUGC paper code
  -> official EVQA teacher predictions
  -> PLCC/SRCC evaluation
  -> hidden/attention/artifact shards for student KD
```

The important repo scripts are:

```text
scripts/run_gcp_official_balanced_5k_from_links.sh
scripts/run_official_snapugc_evqa.py
scripts/monitor_gcp_official_partial.py
scripts/sync_gcp_official_5k_outputs.sh
```

## What Gets Exported

When `EXPORT_ARTIFACTS=1`, the wrapper patches the official EVQA code at
runtime and writes shard files under:

```text
teacher_artifacts/
  official_teacher_artifacts_0000_0024.npz
  official_teacher_artifacts_0000_0024_captions.jsonl
  official_teacher_artifacts_0025_0049.npz
  ...
```

Each `.npz` shard stores:

```text
ids
order_idx
teacher_ecr
clip_ecr
fusion_hidden
temporal_hidden
caption_feature
action_feature
frame_fusion_feature
text_tokens
text_pooled
attention_mean
attention_importance
```

These artifacts are later consumed by:

```bash
python3 scripts/train_official_student_kd.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/train_subset_balanced_5000.csv \
  --input-preset visual_text_sound \
  ...
```

## Local Requirements

Install and authenticate Google Cloud CLI locally:

```bash
gcloud auth login
gcloud config set project snapugc-lightkd
gcloud auth application-default login
```

Use your actual project id if different:

```bash
export PROJECT=snapugc-lightkd
export ZONE=us-central1-a
export INSTANCE=snapugc-l4-artifacts
```

## Create The L4 VM

Recommended machine:

```text
GPU: 1 x NVIDIA L4 24GB
Machine: g2-standard-16
Disk: 400GB pd-ssd
OS: Deep Learning VM CUDA 12 / Ubuntu 22.04
```

Create it:

```bash
gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-16 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=400GB \
  --boot-disk-type=pd-ssd \
  --image-family=common-cu121-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release
```

SSH into the VM:

```bash
gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE"
```

Verify GPU:

```bash
nvidia-smi
```

## Prepare The VM

Inside the VM:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg unzip htop curl

cd /workspace
git clone <YOUR_REPO_URL> SnapUGC-LightKD
cd /workspace/SnapUGC-LightKD

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision torchaudio

pip install \
  numpy pandas scipy matplotlib ipython \
  tensorflow tensorflow_hub \
  imageio oss2 decord ruamel.yaml timm opencv-python \
  einops transformers diffusers torchmetrics ftfy \
  huggingface_hub gdown accelerate sentencepiece
```

Why not use the official `environment.yaml` directly? The official environment
is old and does not fit L4/CUDA 12 cleanly. The repo wrapper patches runtime
compatibility while keeping the original architecture/checkpoints unchanged.

## Prepare Input Data

The managed GCP runner expects:

```text
/workspace/snapugc-data/train_subset_balanced_5000.csv
/workspace/snapugc-data/train_videos_balanced_5000/
```

The CSV must contain:

```text
Id
Title
Description
Download_link
ECR
```

From local machine, copy the CSV:

```bash
gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="mkdir -p /workspace/snapugc-data"

gcloud compute scp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  data/train_subset_balanced_5000.csv \
  "$INSTANCE:/workspace/snapugc-data/train_subset_balanced_5000.csv"
```

The runner downloads videos automatically:

1. First from `Download_link` in the CSV.
2. Then fallback from Kaggle dataset API for missing videos.

For Kaggle fallback, create `/workspace/kaggle.netrc` on the VM:

```bash
cat > /workspace/kaggle.netrc <<'EOF'
machine www.kaggle.com
login <YOUR_KAGGLE_USERNAME>
password <YOUR_KAGGLE_API_KEY>
EOF
chmod 600 /workspace/kaggle.netrc
```

## Prepare Official Checkpoints

The wrapper expects checkpoints here:

```text
/workspace/snapugc-checkpoints/
  EVQA.pth
  net_distort6_g_latest.pth
  r3d18_K_200ep.pth
  mPLUG2_MSRVTT_Caption.pth
  ViT-L-14.tar
  efficientnet_v2_s_21k_ft1k-dbb43f38.pth
```

The first five are the official SnapUGC checkpoint files. The
`efficientnet_v2_s_21k_ft1k-dbb43f38.pth` file is also required because the
official EfficientNetV2 code tries to fetch it from an old URL that now fails.

Copy checkpoint files from local:

```bash
gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="mkdir -p /workspace/snapugc-checkpoints"

gcloud compute scp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  /path/to/checkpoints/* \
  "$INSTANCE:/workspace/snapugc-checkpoints/"
```

## Run Official Inference With Artifact Export

Inside the VM:

```bash
cd /workspace/SnapUGC-LightKD
source .venv/bin/activate

export ROOT_DIR=/workspace/SnapUGC-LightKD
export SUBSET_CSV=/workspace/snapugc-data/train_subset_balanced_5000.csv
export VIDEO_DIR=/workspace/snapugc-data/train_videos_balanced_5000
export CHECKPOINT_DIR=/workspace/snapugc-checkpoints
export OFFICIAL_REPO_DIR=/workspace/SnapUGC_Engagement
export OUT_DIR=/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32
export LOG_FILE="$OUT_DIR/run_from_links.log"

export EXPORT_ARTIFACTS=1
export ARTIFACT_DIR="$OUT_DIR/teacher_artifacts"
export ARTIFACT_SHARD_SIZE=25

export SNAPUGC_LINK_WORKERS=16
export SNAPUGC_KAGGLE_WORKERS=4
export KAGGLE_NETRC=/workspace/kaggle.netrc

export SNAPUGC_DATALOADER_WORKERS=1
export SNAPUGC_MPLUG_CLIP_BATCH=4
export SNAPUGC_OFFICIAL_FRAME_BATCH=12

export SHUTDOWN_ON_EXIT=1
export RESET_LOG=1

nohup bash scripts/run_gcp_official_balanced_5k_from_links.sh \
  > "$OUT_DIR/nohup_runner.log" 2>&1 &
```

Notes:

- `EXPORT_ARTIFACTS=1` is the key switch for hidden/artifact export.
- `ARTIFACT_SHARD_SIZE=25` is safer than huge shards because partial progress
  is saved frequently.
- `SHUTDOWN_ON_EXIT=1` shuts down the VM after success or failure to avoid
  cloud cost surprises.
- `SNAPUGC_OFFICIAL_FRAME_BATCH=12` is the safer L4 setting. If you have more
  VRAM, you may try `24`.

## Monitor Progress

Inside the VM:

```bash
tail -f /workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32/run_from_links.log
```

Partial monitor output:

```bash
tail -f /workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32/monitor_every_500.log
```

Useful checks:

```bash
find /workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  -name 'official_teacher_artifacts_*.npz' | wc -l

ls -lh /workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32
```

Expected scalar output files:

```text
official_input.csv
official_submission_baseline.csv
official_evqa_report.json
official_partial_500_predictions.csv
official_partial_500_report.json
...
official_partial_5000_predictions.csv
official_partial_5000_report.json
```

Expected artifact output:

```text
teacher_artifacts/official_teacher_artifacts_0000_0024.npz
teacher_artifacts/official_teacher_artifacts_0000_0024_captions.jsonl
...
```

## Resume Rules

Scalar prediction resume is supported with:

```bash
export RESUME_PREDICTIONS=/workspace/results/.../official_partial_2500_predictions.csv
```

But artifact resume is deliberately restricted:

```text
RESUME_PREDICTIONS + EXPORT_ARTIFACTS=1
```

is refused by default, because partial prediction CSV files contain only scalar
ECR predictions and do not contain hidden artifacts.

For artifact export, the cleanest rule is:

```text
Run all 5000 videos from the beginning with EXPORT_ARTIFACTS=1.
```

Only use:

```bash
export ALLOW_PARTIAL_ARTIFACT_RESUME=1
```

if you fully understand that seed rows will not have complete hidden artifacts.
For KD training, prefer complete artifact shards.

## Sync Results Back To Local

From local machine:

```bash
PROJECT=snapugc-lightkd \
ZONE=us-central1-a \
INSTANCE=snapugc-l4-artifacts \
REMOTE_OUT_DIR=/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32 \
LOCAL_OUT_DIR=results/original_snapugc_official_balanced_5000_artifacts_g2_32 \
SYNC_VIDEOS=0 \
bash scripts/sync_gcp_official_5k_outputs.sh
```

This copies predictions, reports, logs, and `teacher_artifacts`.

Usually do not sync videos back unless needed:

```bash
SYNC_VIDEOS=1 FORCE_VIDEO_SYNC=1 bash scripts/sync_gcp_official_5k_outputs.sh
```

## Verify Synced Artifacts Locally

From repo root on local machine:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np

artifact_dir = Path("results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts")
paths = sorted(artifact_dir.glob("official_teacher_artifacts_*.npz"))
print("n_shards =", len(paths))
print("first =", paths[0] if paths else None)

total = 0
with np.load(paths[0]) as z:
    print("keys =", z.files)
for path in paths:
    with np.load(path) as z:
        total += len(z["ids"])
print("n_rows =", total)
PY
```

Expected:

```text
n_rows = 5000
keys include frame_fusion_feature, text_pooled, clip_ecr, fusion_hidden, ...
```

Check official teacher score:

```bash
cat results/original_snapugc_official_balanced_5000_artifacts_g2_32/official_evqa_report.json
```

For the locked 5000 run used in this repo, the official teacher is expected to
be around:

```text
PLCC  ~= 0.7146
SRCC  ~= 0.7075
Final ~= 0.7103
```

Small differences can happen if the subset differs, videos are missing, or the
run is resumed incorrectly.

## Train Student From Synced Artifacts

Example deployable KD student:

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
  --split-seed 42 \
  --device mps \
  --run-kind kd \
  --repr-loss cosine \
  --soft-weight 1.1 \
  --clip-weight 0.08 \
  --temporal-weight 0.02 \
  --fusion-weight 0.02 \
  --attention-weight 0.005 \
  --hard-rank-weight 0.04 \
  --teacher-rank-weight 0.18 \
  --teacher-pearson-weight 0.02 \
  --teacher-spearman-weight 0.015 \
  --teacher-listwise-weight 0.02
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` fails | GPU driver not installed | Use Deep Learning VM image or install Google GPU driver. |
| Missing checkpoint error | Checkpoint folder incomplete | Put all six files under `/workspace/snapugc-checkpoints`. |
| EfficientNet downloads from dead URL | Missing external EfficientNet weight | Add `efficientnet_v2_s_21k_ft1k-dbb43f38.pth`. |
| CUDA OOM | Frame batch too high | Set `SNAPUGC_OFFICIAL_FRAME_BATCH=12`. |
| Artifact shards missing | `EXPORT_ARTIFACTS` not set | Run with `EXPORT_ARTIFACTS=1`. |
| Scalar predictions exist but artifacts incomplete | Resumed from scalar CSV | Rerun complete 5000 with artifact export enabled. |
| VM keeps running after job | `SHUTDOWN_ON_EXIT=0` or script crashed before trap | Stop manually with `gcloud compute instances stop`. |

## Stop Or Delete VM

Stop:

```bash
gcloud compute instances stop "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE"
```

Delete:

```bash
gcloud compute instances delete "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE"
```

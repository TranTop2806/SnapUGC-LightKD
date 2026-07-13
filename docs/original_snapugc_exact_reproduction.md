# Exact Original SnapUGC Reproduction

This note documents the official SnapUGC paper/teacher code path that is used
for the current bounded 5000-video thesis run.

## Paper Feature Stack

The original paper and official repository use:

| Paper component | Official implementation | Shape in official EVQA code |
|---|---|---:|
| Per-frame semantic features | `EfficientNetV2-s`, ImageNet pretrained | `528` per frame |
| Per-frame distortion features | UVQ-style `Distortion`, trained on KADIS-700K/KADID-10K | `256` per frame |
| Per-clip action features | `ResNet3D-18`, Kinetics pretrained | `512` per 16-frame clip |
| Video caption + mid-layer features | `mPLUG-2` MSRVTT caption checkpoint | caption text + `1024` per clip |
| Background sound | `YAMNet`, top-5 AudioSet labels converted to text | text |
| Title/description/caption/sound text | Stable Diffusion v1.4 tokenizer + text encoder in official code | `77 x 768` each |

The official checkpoints expected by `ECR_inference` are:

- `EVQA.pth`
- `net_distort6_g_latest.pth`
- `r3d18_K_200ep.pth`
- `mPLUG2_MSRVTT_Caption.pth`
- `ViT-L-14.tar`

Download source: <https://drive.google.com/drive/folders/19_s6Z4R-iTaQHkRWFRn2Aby1FOy2cHes?usp=share_link>

The released EfficientNetV2 implementation also tries to download an ImageNet
pretrained semantic backbone file at runtime:

- `efficientnet_v2_s_21k_ft1k-dbb43f38.pth`

This file is not one of the five SnapUGC Drive checkpoints above. The original
OneDrive URL embedded in `modules/efficientnet_v2.py` now returns HTTP 404, so
exact reproduction requires placing this file in the Kaggle input/checkpoint
dataset as well. The Kaggle notebook caches it under
`~/.cache/torch/hub/checkpoints/` before running the official script.

## Official EVQA Architecture

The paper teacher architecture is not implemented in this repository's
`src/snapugc_lightkd/models.py`. The clean local reference copy is vendored at:

```text
third_party/SnapUGC_Engagement/ECR_inference/
```

On GCloud, `scripts/run_official_snapugc_evqa.py` uses the same authors'
released source and patches a working copy at runtime. The relevant files are:

```text
third_party/SnapUGC_Engagement/ECR_inference/modules/EVQA.py
third_party/SnapUGC_Engagement/ECR_inference/test_SnapUGC_baseline.py
third_party/SnapUGC_Engagement/ECR_inference/modules/efficientnet_v2.py
third_party/SnapUGC_Engagement/ECR_inference/modules/distort.py
third_party/SnapUGC_Engagement/ECR_inference/modules/resnet3d.py
third_party/SnapUGC_Engagement/ECR_inference/mPLUG_2/models/model_video_caption_mplug2.py
third_party/SnapUGC_Engagement/ECR_inference/mPLUG_2/models/visual_transformers.py
```

The upstream source is:

<https://github.com/dasongli1/SnapUGC_Engagement/blob/main/ECR_inference/modules/EVQA.py>

Main fusion path:

```text
EfficientNet semantic frames (T x 528)
  -> fc1 + frame grouping over 16 frames

UVQ distortion frames (T x 256)
  -> fc3 + frame grouping over 16 frames

ResNet3D action clips (C x 512)
  -> feat3_preprocess + fc30
  -> query for cross-attention into each text stream

mPLUG-2 clip features (C x 1024)
  -> fc4

Text streams:
  sound top-5 text
  title
  description
  generated caption
  -> Stable Diffusion text encoder
  -> multiple CrossAttention blocks

concat(
  semantic+distortion fused feature,
  action feature,
  mPLUG-2 feature,
  sound/title/description/caption cross-attended features
)
  -> fc_merge123
  -> 8 TransformerBlock temporal self-attention
  -> output MLP
  -> ECR
```

Important paper alignment detail: the paper extracts clip/frame features across
the whole video. ECR is then defined by the ECR output head averaged over clips
within the first 5 seconds. Do not crop the input video to only 5 seconds when
claiming an exact official/paper-style run; a first-5s-only run is an efficiency
ablation, not the main reproduction.

## Running The Exact Official Path On The 5k Subset

Use the wrapper below after the 5k videos have been downloaded and the official
checkpoints have been placed under `ECR_inference/checkpoints/` or supplied by
the GCloud runner's `CHECKPOINT_DIR`.

```bash
python scripts/run_official_snapugc_evqa.py \
  --official-repo-dir third_party/SnapUGC_Engagement \
  --videos-dir data/official_balanced_5000_videos \
  --csv-file data/train_subset_balanced_5000.csv \
  --out-dir results/original_snapugc_official_balanced_5000
```

Outputs:

- `official_input.csv`: CSV converted to official repo format.
- `official_submission_baseline.csv`: official ECR predictions.
- `official_evqa_report.json`: PLCC/SRCC/final score against the 5k ECR labels.

## GCP L4 Local Run Guide

Use an L4 VM for official inference, not for training the original model from
scratch. The wrapper runs the released EVQA checkpoint on the 5k videos and
evaluates against local ECR labels.

Recommended VM:

```text
GPU: 1 x NVIDIA L4 24GB
Disk: 300-500GB balanced/pd-ssd
Image: Ubuntu 22.04 with CUDA 12 / Deep Learning VM
Python: 3.10
```

Create a VM from local terminal:

```bash
gcloud compute instances create snapugc-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-16 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=400GB \
  --boot-disk-type=pd-ssd \
  --image-family=common-cu121-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release
```

If that image family is unavailable, create any Ubuntu 22.04 L4 VM and install
the NVIDIA driver/CUDA from Google Cloud's GPU driver instructions. Verify:

```bash
nvidia-smi
```

Set up the project:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg unzip htop

git clone <YOUR_REPO_URL_OR_COPY_THIS_REPO> SnapUGC-LightKD
cd SnapUGC-LightKD

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

The original `environment.yaml` uses old PyTorch/CUDA versions. Do not use it
on L4; L4 needs a modern CUDA/PyTorch build. The wrapper patches compatibility
issues without changing the official model architecture.

Expected data layout on the VM:

```text
/workspace/snapugc-data/train_data.csv
/workspace/snapugc-data/train_videos/<Id>.mp4
/workspace/snapugc-checkpoints/EVQA.pth
/workspace/snapugc-checkpoints/net_distort6_g_latest.pth
/workspace/snapugc-checkpoints/r3d18_K_200ep.pth
/workspace/snapugc-checkpoints/mPLUG2_MSRVTT_Caption.pth
/workspace/snapugc-checkpoints/ViT-L-14.tar
/workspace/snapugc-checkpoints/efficientnet_v2_s_21k_ft1k-dbb43f38.pth
```

Copy the checkpoints into the official repo after the wrapper clones it:

```bash
python scripts/run_official_snapugc_evqa.py \
  --official-repo-dir /workspace/SnapUGC_Engagement \
  --videos-dir /workspace/snapugc-data/train_videos \
  --csv-file /workspace/snapugc-data/train_data.csv \
  --out-dir /workspace/results/original_snapugc_official_evqa_5000 \
  --max-samples 5000 \
  --skip-run

mkdir -p /workspace/SnapUGC_Engagement/ECR_inference/checkpoints
cp /workspace/snapugc-checkpoints/* \
  /workspace/SnapUGC_Engagement/ECR_inference/checkpoints/
```

Run official inference:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SNAPUGC_OFFICIAL_FRAME_BATCH=24

python scripts/run_official_snapugc_evqa.py \
  --official-repo-dir /workspace/SnapUGC_Engagement \
  --videos-dir /workspace/snapugc-data/train_videos \
  --csv-file /workspace/snapugc-data/train_data.csv \
  --out-dir /workspace/results/original_snapugc_official_evqa_5000 \
  --max-samples 5000
```

For the full balanced 5k GCP run, first upload the complete `<Id>.mp4` video
directory and the six checkpoints. The runner symlinks the checkpoints, runs
official inference, and saves teacher artifacts:

```bash
SHUTDOWN_ON_EXIT=1 \
ROOT_DIR=/workspace/SnapUGC-LightKD \
SUBSET_CSV=/workspace/snapugc-data/train_subset_balanced_5000.csv \
VIDEO_DIR=/workspace/snapugc-data/official_balanced_5000_videos \
OUT_DIR=/workspace/results/original_snapugc_official_balanced_5000 \
CHECKPOINT_DIR=/workspace/snapugc-checkpoints \
OFFICIAL_REPO_DIR=/workspace/SnapUGC-LightKD/third_party/SnapUGC_Engagement \
EXPORT_ARTIFACTS=1 \
RUN_STUDENT=0 \
bash scripts/run_gcp_full_pipeline.sh
```

Partial outputs are written as:

```text
official_partial_500_predictions.csv
official_partial_500_report.json
official_partial_1000_predictions.csv
official_partial_1000_report.json
...
```

If L4 OOMs, rerun with:

```bash
export SNAPUGC_OFFICIAL_FRAME_BATCH=12
```

This only changes frame feature extraction batch size, not the paper model.

Outputs:

```text
/workspace/results/original_snapugc_official_evqa_5000/official_submission_baseline.csv
/workspace/results/original_snapugc_official_evqa_5000/official_evqa_report.json
```

Common failures:

| Error | Cause | Fix |
|---|---|---|
| `HTTP Error 404` in `efficientnet_v2.py` | Missing external EfficientNetV2-S ImageNet weight | Add `efficientnet_v2_s_21k_ft1k-dbb43f38.pth` to checkpoints. |
| `ViT-L-14.tar does not exist` | Missing mPLUG CLIP video backbone | Add `ViT-L-14.tar`. |
| CUDA OOM | L4 memory pressure during frame feature extraction | Set `SNAPUGC_OFFICIAL_FRAME_BATCH=12`. |
| `No mp4 videos found` | Wrong video folder nesting | Point `--videos-dir` to the folder directly containing `<Id>.mp4`. |

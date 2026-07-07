# SnapUGC-LightKD

Graduation-thesis code for building a lightweight knowledge-distilled model to
predict and explain social-video engagement. The project uses the released
SnapUGC EVQA model as a heavyweight teacher, trains a compact student on a fixed
5000-video subset, and adds an NLA-inspired semantic explanation layer for
predictions.

## What This Project Does

The thesis objective is:

```text
Build a distilled student model that predicts video engagement and explains why
the video is expected to be low/medium/high engagement.
```

There are two separate phases:

```text
1. Offline training / distillation
   Raw SnapUGC videos
     -> official SnapUGC EVQA teacher
     -> teacher predictions + hidden states + attention/artifacts
     -> compact student KD training

2. Inference / demo
   New video + title + description
     -> student-only feature extraction
     -> compact student prediction
     -> ablation/attention evidence
     -> semantic labeling
     -> optional LLM explanation + top clips + recommendations
```

The important deployment constraint is that **the teacher is not required at
new-video inference time**. Teacher artifacts are used for offline supervision
only.

## Current Status

Completed:

- Official SnapUGC EVQA teacher reproduced on a locked 5000-video subset.
- Teacher artifact export completed for all 5000 videos.
- Compact student baseline and KD model trained.
- KD improves validation score over the baseline.
- Artifact-based explanation script for already-exported videos.
- Student-only new-video inference script.
- Demo UI for uploading a completely new video and viewing prediction,
  evidence, semantic labels, thumbnails, and split recommendations.
- Deterministic auto-edit loop for feasible improvements: the demo can adjust
  weak non-top clips with brightness/contrast/sharpness/saturation, rerun
  student inference, and compare ECR before/after.
- Google Cloud setup documented for both GPU training and CPU demo fallback.

Current best KD run:

```text
save_dir: results/kd_tuning_official_5k/v05_small_cosine_rank
input_preset: visual_text
repr_loss: cosine
baseline_final: 0.5264651028399077
kd_best_epoch: 9
kd_final: 0.535382287857137
kd_gain_final_score: 0.008917185017229379
```

Older locked-summary docs may contain earlier experimental scores. Treat the
report JSON in the run directory as the source of truth for the latest run.

## Repository Layout

```text
SnapUGC-LightKD/
  data/
    official_5k_split/          # tracked split CSVs and IDs
  demo_app/                     # FastAPI + static UI for video demo
  docs/                         # architecture and result notes
  notebooks/                    # original Kaggle reproduction notebook
  references/                   # papers and challenge references
  scripts/
    run_official_snapugc_evqa.py
    run_gcp_official_balanced_5k_from_links.sh
    run_gcp_full_pipeline.sh
    resume_gcp_after_artifacts.sh
    train_official_student_kd.py
    infer_one_video_with_expl.py
    infer_new_video_with_student_expl.py
  src/snapugc_lightkd/
    official_artifacts.py       # teacher artifact dataset utilities
    official_student.py         # compact student model and KD losses
    explanations.py             # NLA-inspired verbalization + ablation
    llm_explainer.py            # semantic evidence package -> LLM/template text
    student_native.py           # teacher-free new-video feature path
  third_party/
    SnapUGC_Engagement/         # pinned official teacher source
```

Generated data, videos, checkpoints, artifacts, and outputs are intentionally
not committed.

## Data And Checkpoints

The locked split is tracked:

```text
data/official_5k_split/train_4000.csv
data/official_5k_split/test_1000.csv
data/official_5k_split/split_all_5000.csv
data/official_5k_split/train_ids_4000.txt
data/official_5k_split/test_ids_1000.txt
data/official_5k_split/manifest.json
```

Large local/VM-only paths:

```text
~/workspace/snapugc-data/train_videos_balanced_5000
~/workspace/snapugc-checkpoints
~/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32
~/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank
```

Expected teacher/checkpoint files:

```text
EVQA.pth
mPLUG2_MSRVTT_Caption.pth
net_distort6_g_latest.pth
r3d18_K_200ep.pth
ViT-L-14.tar
efficientnet_v2_s_21k_ft1k-dbb43f38.pth
```

On the VM these live in:

```text
~/workspace/snapugc-checkpoints
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For the demo UI:

```bash
pip install -r requirements-demo.txt
```

Run the lightweight regression and smoke suite after changing the teacher,
artifact format, student model, or demo wiring:

```bash
pip install -e '.[dev]'
pytest -q
python -m uvicorn demo_app.app:app --host 127.0.0.1 --port 7860
curl http://127.0.0.1:7860/health
```

Optional notebook dependencies:

```bash
pip install -e ".[notebook]"
```

## Model Roles

### 1. Official Teacher

Input:

```text
raw video + title + description
```

Output:

```text
teacher ECR
teacher clip ECR
temporal hidden states
fusion hidden states
temporal attention
caption/text artifacts
```

The teacher is the released SnapUGC EVQA stack in
`third_party/SnapUGC_Engagement/ECR_inference`. It combines EfficientNetV2,
UVQ-style distortion features, ResNet3D, mPLUG-2, YAMNet sound labels, Stable
Diffusion text encoder, and EVQA fusion/head.

### 2. Student Baseline

Input preset for the main thesis setting:

```text
visual_text
- frame_fusion_feature: T x 1024
- title pooled text embedding: 1 x 768
- description pooled text embedding: 1 x 768
```

Objective:

```text
MSE(student_ecr, true_ecr)
```

### 3. Student KD

Same inference inputs as baseline. Additional training losses mimic the teacher:

```text
loss_kd =
  hard_ecr      * MSE(student_ecr, true_ecr)
+ soft_ecr      * MSE(student_ecr, teacher_ecr)
+ clip_ecr      * MSE(student_clip_ecr, teacher_clip_ecr)
+ temporal      * repr_loss(project(student_temporal), teacher_temporal_hidden)
+ fusion        * repr_loss(project(student_hidden), mean(teacher_fusion_hidden))
+ attention     * KL(student_attention, teacher_attention)
+ hard_rank     * pairwise_rank(student_ecr, true_ecr)
+ teacher_rank  * pairwise_rank(student_ecr, teacher_ecr)
```

The tuned compact student uses:

```text
hidden_dim = 96
Transformer layers = 1
heads = 4
dropout = 0.22
max_clips = 16
repr_loss = cosine
```

## GCloud GPU Training Setup

The original GPU VM:

```text
name: evqa-training
zone: asia-southeast1-b
machine: g2-standard-12
gpu: 1 x NVIDIA L4
boot disk: 200GB pd-balanced
project path: ~/workspace/SnapUGC-LightKD
checkpoint path: ~/workspace/snapugc-checkpoints
video path: ~/workspace/snapugc-data/train_videos_balanced_5000
```

Install/update code on the VM:

```bash
cd ~/workspace/SnapUGC-LightKD
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install CUDA-compatible PyTorch for the current driver. On the original L4 VM
with NVIDIA driver 550 / CUDA 12.4, a working PyTorch pair was:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
```

Useful GPU check:

```bash
nvidia-smi
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Run the full GCloud pipeline:

```bash
cd ~/workspace/SnapUGC-LightKD

nohup env \
  ROOT_DIR=$HOME/workspace/SnapUGC-LightKD \
  SUBSET_CSV=$HOME/workspace/SnapUGC-LightKD/data/official_5k_split/split_all_5000.csv \
  VIDEO_DIR=$HOME/workspace/snapugc-data/train_videos_balanced_5000 \
  OUT_DIR=$HOME/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32 \
  KD_OUT_DIR=$HOME/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank \
  CHECKPOINT_DIR=$HOME/workspace/snapugc-checkpoints \
  OFFICIAL_REPO_DIR=$HOME/workspace/SnapUGC_Engagement \
  KAGGLE_NETRC=$HOME/workspace/kaggle.netrc \
  SHUTDOWN_ON_EXIT=0 \
  bash scripts/run_gcp_full_pipeline.sh \
  > ~/workspace/snapugc_full_pipeline.nohup.log 2>&1 &
```

Check progress:

```bash
tail -f ~/workspace/snapugc_full_pipeline.nohup.log
ps aux | grep -E 'run_gcp_full_pipeline|run_official|train_official|python' | grep -v grep
nvidia-smi
```

If teacher artifact export finished but KD needs resume:

```bash
nohup bash scripts/resume_gcp_after_artifacts.sh \
  > ~/workspace/snapugc_resume_pipeline.nohup.log 2>&1 &

tail -f ~/workspace/snapugc_resume_pipeline.nohup.log
```

Known memory-safe settings for the official teacher export:

```text
SNAPUGC_DATALOADER_WORKERS=0
SNAPUGC_OFFICIAL_FRAME_BATCH=8
SNAPUGC_MPLUG_CLIP_BATCH=1
SNAPUGC_CAPTION_NUM_FRAMES=8
```

The project patches the vendored official inference at runtime to use these
environment variables, `torch.inference_mode()`, and `torch.cuda.empty_cache()`.

## Student KD Training Command

```bash
python scripts/train_official_student_kd.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/official_5k_split/split_all_5000.csv \
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

Output:

```text
official_student_kd_report.json
student_baseline_best.pth
student_kd_best.pth
```

## Explanation Modes

### Artifact-Based Explanation

Use this for a video already present in teacher artifacts:

```bash
python scripts/infer_one_video_with_expl.py \
  --artifact-dir results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts \
  --labels-csv data/official_5k_split/split_all_5000.csv \
  --video-id VIDEO_ID \
  --report-json results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json \
  --device cpu \
  --topk 3 \
  --out-json results/explanations/VIDEO_ID.json
```

This mode uses teacher artifacts to inspect what the student learned. It is
useful for offline analysis, but it is not the final deployment path.

### Student-Only New-Video Explanation

Use this for a completely new video:

```bash
python scripts/infer_new_video_with_student_expl.py \
  --video /path/to/new_video.mp4 \
  --title "Short, specific title" \
  --description "Optional context" \
  --report-json results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json \
  --checkpoint results/kd_tuning_official_5k/v05_small_cosine_rank/student_kd_best.pth \
  --labels-csv data/official_5k_split/split_all_5000.csv \
  --out-json results/demo_runs/example/result.json \
  --assets-dir results/demo_runs/example/assets
```

This path does **not** call the teacher. It builds native inputs from sampled
frames, lightweight visual metrics / EfficientNet features when available, and
metadata text embeddings.

The new-video explanation flow is:

```text
student prediction
  -> clip/text ablation and attention evidence
  -> semantic labels for important clips/text
  -> optional LLM explanation, or deterministic template fallback
  -> split recommendations for post-production vs content changes
```

The explanation returns:

```text
- predicted ECR
- low/medium/high band
- top temporal clips
- thumbnail images for top clips
- text stream contribution
- semantic attributes
- semantic_explanation generated by LLM or fallback template
- actionable recommendations split into post-production/metadata and content/direction changes
```

Semantic attribute fields:

```text
hook_strength
motion_action
visual_clarity
lighting_quality
text_specificity
pacing_variety
```

These attributes are not a separately trained concept bottleneck model. The
current thesis framing is **semantic labeling -> LLM/template explanation**:
selected student evidence is verbalized for normal readers, then ablations check
whether the mentioned evidence affects the student prediction.

Optional remote API LLM settings:

```bash
export SNAPUGC_LLM_BACKEND="openai"
export SNAPUGC_LLM_API_KEY="..."
export SNAPUGC_LLM_BASE_URL="https://api.openai.com/v1"   # optional
export SNAPUGC_LLM_MODEL="gpt-4o-mini"                    # optional
```

Optional local/offline LLM settings:

```bash
python scripts/prepare_local_llm.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --cache-dir ~/.cache/snapugc-local-llm

export SNAPUGC_LLM_BACKEND="local"
export SNAPUGC_LOCAL_LLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
export SNAPUGC_LOCAL_LLM_CACHE="$HOME/.cache/snapugc-local-llm"
```

Use `Qwen/Qwen2.5-3B-Instruct` for a stronger offline demo on Apple Silicon
machines with enough memory, such as an M1 Max with 64GB RAM. Use
`Qwen/Qwen2.5-0.5B-Instruct` only for very small machines, or try
`Qwen/Qwen2.5-7B-Instruct` if you can tolerate slower generation. For
defense/demo without a VM, pre-download the local model on your laptop, keep the
checkpoint/report files local, and run the UI locally. If no API key or local
model is configured, inference still works and uses a deterministic template
over the same semantic evidence package. Use `--disable-llm` to force template
mode.

Convenience local-LLM demo command:

```bash
SNAPUGC_PREPARE_LOCAL_LLM=1 \
SNAPUGC_REPORT_JSON=results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json \
SNAPUGC_STUDENT_CHECKPOINT=results/kd_tuning_official_5k/v05_small_cosine_rank/student_kd_best.pth \
SNAPUGC_LABELS_CSV=data/official_5k_split/split_all_5000.csv \
scripts/run_demo_local_llm.sh
```

Set `SNAPUGC_PREPARE_LOCAL_LLM=1` only the first time to download/cache the
model. Later runs can omit it and work offline from the local cache.

## Demo UI

Install:

```bash
pip install -r requirements-demo.txt
```

Run locally:

```bash
uvicorn demo_app.app:app --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

The UI lets you upload a video, enter title/description, and returns the full
student-only prediction + explanation report. After the first analysis, use
`Chỉnh video` to run the deterministic auto-edit loop:

```text
original video -> explain weak/non-top clips -> feasible OpenCV/ffmpeg edits
-> edited video -> rerun student inference -> before/after ECR comparison
```

The auto-edit module intentionally does **not** invent new scenes, actions, or
objects. It only applies feasible local edits such as brightness, contrast,
sharpness, and saturation to selected clips while preserving the original video
timeline.

## GCloud Demo VM

When the original L4 VM cannot start because of GCP capacity stockout, a CPU
clone can run the demo UI. The current clone setup was:

```text
original GPU VM: evqa-training
zone: asia-southeast1-b
machine: g2-standard-12
status during stockout: TERMINATED

CPU demo clone: evqa-training-demo
zone: asia-southeast1-a
machine: n2-standard-8
disk: evqa-training-demo-disk
source snapshot: evqa-training-boot-20260525-224211
```

The clone is for demo/inference only. It has no GPU and should not be used to
rerun the full teacher pipeline.

Start demo server on the VM:

```bash
cd ~/workspace/SnapUGC-LightKD
source .venv/bin/activate
pip install -r requirements-demo.txt

nohup env \
  SNAPUGC_REPORT_JSON=$HOME/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json \
  SNAPUGC_STUDENT_CHECKPOINT=$HOME/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank/student_kd_best.pth \
  SNAPUGC_LABELS_CSV=$HOME/workspace/SnapUGC-LightKD/data/official_5k_split/split_all_5000.csv \
  SNAPUGC_EFFICIENTNET_WEIGHTS=$HOME/workspace/snapugc-checkpoints/efficientnet_v2_s_21k_ft1k-dbb43f38.pth \
  .venv/bin/python -m uvicorn demo_app.app:app --host 127.0.0.1 --port 7860 \
  > ~/workspace/snapugc_demo_ui.nohup.log 2>&1 &
```

Check:

```bash
curl http://127.0.0.1:7860/health
tail -f ~/workspace/snapugc_demo_ui.nohup.log
```

Tunnel from your local machine:

```bash
gcloud compute ssh aothuat0424@evqa-training-demo \
  --zone asia-southeast1-a \
  -- -L 7860:localhost:7860
```

Open locally:

```text
http://localhost:7860
```

Stop the demo VM from local/cloud shell:

```bash
gcloud compute instances stop evqa-training-demo --zone asia-southeast1-a
```

Do not run that stop command from inside the VM unless the VM service account
has Compute API scopes and IAM permissions. From inside the VM you may see:

```text
ACCESS_TOKEN_SCOPE_INSUFFICIENT
```

because the active account is the instance service account, not your local
Google account.

## GCloud Stockout Notes

If `evqa-training` cannot start:

```text
ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS
g2-standard-12 with nvidia-l4 currently unavailable
```

Options:

1. Try later in the same zone.
2. Try another zone with L4/G2 capacity.
3. Create a CPU clone from a boot-disk snapshot for UI demo only.

Safe CPU clone flow:

```bash
SNAP=evqa-training-boot-$(date +%Y%m%d-%H%M%S)
gcloud compute snapshots create "$SNAP" \
  --source-disk=evqa-training \
  --source-disk-zone=asia-southeast1-b \
  --storage-location=asia-southeast1

gcloud compute disks create evqa-training-demo-disk \
  --zone asia-southeast1-a \
  --source-snapshot "$SNAP" \
  --type pd-balanced \
  --size 200GB

gcloud compute instances create evqa-training-demo \
  --zone asia-southeast1-a \
  --machine-type n2-standard-8 \
  --disk name=evqa-training-demo-disk,boot=yes,auto-delete=no \
  --network default \
  --subnet default \
  --maintenance-policy=MIGRATE \
  --provisioning-model=STANDARD
```

This does not delete or modify the original GPU VM/disk.

## Useful Commands

Check active training/inference:

```bash
ps aux | grep -E 'run_gcp_full_pipeline|run_official|train_official|uvicorn|python' | grep -v grep
```

Check GPU:

```bash
nvidia-smi
```

Read demo output:

```bash
jq . results/demo_runs/example/result.json
```

Read explanation sections:

```bash
jq '.scores' result.json
jq '.nla_style_explanation.summary' result.json
jq '.semantic_explanation.llm' result.json
jq '.evidence.top_clips' result.json
jq '.semantic_attributes.attributes' result.json
jq '.recommendations' result.json
```

## Thesis Interpretation

The clean thesis statement is:

```text
The teacher model is used offline to generate soft labels and explanation
proxies. After distillation, the compact student predicts and explains new
videos without querying the teacher at inference time.
```

This protects the main distillation claim: the deployed/demo path is smaller
and teacher-free, while the teacher remains the source of supervision during
training.

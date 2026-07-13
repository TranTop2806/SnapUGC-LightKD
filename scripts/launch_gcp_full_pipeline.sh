#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-$(gcloud config get-value compute/zone 2>/dev/null)}"
INSTANCE="${INSTANCE:-snapugc-l4-artifacts}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-16}"
ACCELERATOR="${ACCELERATOR:-type=nvidia-l4,count=1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-350GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/SnapUGC-LightKD}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/workspace/snapugc-data}"
REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-/workspace/snapugc-checkpoints}"
LOCAL_SUBSET_CSV="${LOCAL_SUBSET_CSV:-data/official_5k_split/split_all_5000.csv}"
LOCAL_VIDEO_DIR="${LOCAL_VIDEO_DIR:-data/official_balanced_5000_videos}"
REMOTE_SUBSET_CSV="${REMOTE_SUBSET_CSV:-${REMOTE_DATA_DIR}/train_subset_balanced_5000.csv}"
REMOTE_VIDEO_DIR="${REMOTE_VIDEO_DIR:-${REMOTE_DATA_DIR}/official_balanced_5000_videos}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-third_party/SnapUGC_Engagement/ECR_inference/checkpoints}"
REMOTE_OUT_DIR="${REMOTE_OUT_DIR:-/workspace/results/original_snapugc_official_balanced_5000_artifacts_g2_32}"
REMOTE_KD_OUT_DIR="${REMOTE_KD_OUT_DIR:-/workspace/results/kd_tuning_official_5k/v05_small_cosine_rank}"
START_PIPELINE="${START_PIPELINE:-1}"
UPLOAD_CHECKPOINTS="${UPLOAD_CHECKPOINTS:-1}"
UPLOAD_SOURCE="${UPLOAD_SOURCE:-1}"
UPLOAD_VIDEOS="${UPLOAD_VIDEOS:-1}"

if [[ -z "$PROJECT" || -z "$ZONE" ]]; then
  echo "PROJECT and ZONE must be set, or configured in gcloud."
  exit 2
fi

required_checkpoints=(
  EVQA.pth
  net_distort6_g_latest.pth
  r3d18_K_200ep.pth
  mPLUG2_MSRVTT_Caption.pth
  ViT-L-14.tar
  efficientnet_v2_s_21k_ft1k-dbb43f38.pth
)

for name in "${required_checkpoints[@]}"; do
  if [[ ! -s "${LOCAL_CHECKPOINT_DIR}/${name}" ]]; then
    echo "Missing checkpoint: ${LOCAL_CHECKPOINT_DIR}/${name}"
    exit 3
  fi
done

if [[ ! -s "$LOCAL_SUBSET_CSV" ]]; then
  echo "Missing subset CSV: $LOCAL_SUBSET_CSV"
  exit 4
fi

if [[ "$UPLOAD_VIDEOS" == "1" && ! -d "$LOCAL_VIDEO_DIR" ]]; then
  echo "Missing video directory: $LOCAL_VIDEO_DIR"
  exit 5
fi

if ! gcloud compute instances describe "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" >/dev/null 2>&1; then
  echo "Creating GCP instance ${INSTANCE} in ${ZONE}..."
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="$ACCELERATOR" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --metadata=install-nvidia-driver=True
else
  echo "Instance ${INSTANCE} already exists."
fi

status="$(gcloud compute instances describe "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --format='value(status)')"
if [[ "$status" == "TERMINATED" ]]; then
  gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$ZONE"
fi

echo "Waiting for SSH..."
gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="mkdir -p '$REMOTE_ROOT' '$REMOTE_DATA_DIR' '$REMOTE_CHECKPOINT_DIR' /workspace/results"

if [[ "$UPLOAD_SOURCE" == "1" ]]; then
  archive="/tmp/snapugc-lightkd-gcp-source.tgz"
  tar \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='results' \
    --exclude='data/official_5k_split/train_videos' \
    --exclude='data/official_5k_split/test_videos' \
    --exclude='data/official_balanced_5000_videos' \
    --exclude='third_party/SnapUGC_Engagement/ECR_inference/checkpoints' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czf "$archive" .
  gcloud compute scp \
    --project="$PROJECT" \
    --zone="$ZONE" \
    "$archive" \
    "${INSTANCE}:/tmp/snapugc-lightkd-gcp-source.tgz"
  gcloud compute ssh "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --command="rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT' && tar -xzf /tmp/snapugc-lightkd-gcp-source.tgz -C '$REMOTE_ROOT'"
fi

gcloud compute scp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  "$LOCAL_SUBSET_CSV" \
  "${INSTANCE}:${REMOTE_SUBSET_CSV}"

if [[ "$UPLOAD_VIDEOS" == "1" ]]; then
  gcloud compute scp \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --recurse \
    "$LOCAL_VIDEO_DIR" \
    "${INSTANCE}:${REMOTE_DATA_DIR}/"
fi

if [[ "$UPLOAD_CHECKPOINTS" == "1" ]]; then
  for name in "${required_checkpoints[@]}"; do
    gcloud compute scp \
      --project="$PROJECT" \
      --zone="$ZONE" \
      "${LOCAL_CHECKPOINT_DIR}/${name}" \
      "${INSTANCE}:${REMOTE_CHECKPOINT_DIR}/${name}"
  done
fi

gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="cd '$REMOTE_ROOT' && python3 -m venv .venv && .venv/bin/python -m pip install -U pip wheel setuptools && .venv/bin/python -m pip install -e ."

if [[ "$START_PIPELINE" == "1" ]]; then
  echo "Starting remote full pipeline under nohup..."
  gcloud compute ssh "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --command="cd '$REMOTE_ROOT' && nohup env ROOT_DIR='$REMOTE_ROOT' SUBSET_CSV='$REMOTE_SUBSET_CSV' VIDEO_DIR='$REMOTE_VIDEO_DIR' CHECKPOINT_DIR='$REMOTE_CHECKPOINT_DIR' OUT_DIR='$REMOTE_OUT_DIR' KD_OUT_DIR='$REMOTE_KD_OUT_DIR' OFFICIAL_REPO_DIR='$REMOTE_ROOT/third_party/SnapUGC_Engagement' SHUTDOWN_ON_EXIT=1 bash scripts/run_gcp_full_pipeline.sh > /workspace/snapugc_full_pipeline.nohup.log 2>&1 < /dev/null &"
fi

cat <<EOF
Launched.

Monitor:
  gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE --command='tail -f /workspace/snapugc_full_pipeline.nohup.log'

Sync outputs after the VM stops:
  gcloud compute scp --project=$PROJECT --zone=$ZONE --recurse "$INSTANCE:$REMOTE_OUT_DIR/" results/original_snapugc_official_balanced_5000_artifacts_g2_32/
  gcloud compute scp --project=$PROJECT --zone=$ZONE --recurse "$INSTANCE:$REMOTE_KD_OUT_DIR/" results/kd_tuning_official_5k/v05_small_cosine_rank/
EOF

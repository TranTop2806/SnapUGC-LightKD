#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-snapugc-lightkd}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-snapugc-l4}"
REMOTE_OUT_DIR="${REMOTE_OUT_DIR:-/workspace/results/original_snapugc_official_balanced_5000}"
REMOTE_VIDEO_DIR="${REMOTE_VIDEO_DIR:-/workspace/snapugc-data/train_videos_balanced_5000}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-results/original_snapugc_official_balanced_5000}"
LOCAL_VIDEO_DIR="${LOCAL_VIDEO_DIR:-data/official_balanced_5000_videos}"
SYNC_VIDEOS="${SYNC_VIDEOS:-0}"
FORCE_VIDEO_SYNC="${FORCE_VIDEO_SYNC:-0}"

mkdir -p "$LOCAL_OUT_DIR"

echo "Syncing official EVQA outputs..."
gcloud compute scp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --recurse \
  "${INSTANCE}:${REMOTE_OUT_DIR}/" \
  "$LOCAL_OUT_DIR/"

if [[ "$SYNC_VIDEOS" != "1" ]]; then
  echo "SYNC_VIDEOS=0; skipping video sync."
  exit 0
fi

remote_kb="$(gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="du -sk '$REMOTE_VIDEO_DIR' | cut -f1")"
available_kb="$(df -k "$(dirname "$LOCAL_VIDEO_DIR")" | awk 'NR==2 {print $4}')"
echo "Remote video size: $((remote_kb / 1024)) MiB"
echo "Local available:   $((available_kb / 1024)) MiB"

if (( available_kb < remote_kb )) && [[ "$FORCE_VIDEO_SYNC" != "1" ]]; then
  echo "Not enough local space for videos. Set FORCE_VIDEO_SYNC=1 to try anyway, or change LOCAL_VIDEO_DIR to an external disk."
  exit 6
fi

mkdir -p "$LOCAL_VIDEO_DIR"
echo "Syncing videos..."
gcloud compute scp \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --recurse \
  "${INSTANCE}:${REMOTE_VIDEO_DIR}/" \
  "$LOCAL_VIDEO_DIR/"

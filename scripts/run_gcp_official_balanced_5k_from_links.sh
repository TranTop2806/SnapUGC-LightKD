#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/SnapUGC-LightKD}"
SUBSET_CSV="${SUBSET_CSV:-/workspace/snapugc-data/train_subset_balanced_5000.csv}"
VIDEO_DIR="${VIDEO_DIR:-/workspace/snapugc-data/train_videos_balanced_5000}"
OUT_DIR="${OUT_DIR:-/workspace/results/original_snapugc_official_balanced_5000}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run_from_links.log}"
RESET_LOG="${RESET_LOG:-0}"
RESUME_PREDICTIONS="${RESUME_PREDICTIONS:-}"
EXPORT_ARTIFACTS="${EXPORT_ARTIFACTS:-0}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${OUT_DIR}/teacher_artifacts}"
ARTIFACT_SHARD_SIZE="${ARTIFACT_SHARD_SIZE:-500}"
ALLOW_PARTIAL_ARTIFACT_RESUME="${ALLOW_PARTIAL_ARTIFACT_RESUME:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/snapugc-checkpoints}"
OFFICIAL_REPO_DIR="${OFFICIAL_REPO_DIR:-/workspace/SnapUGC_Engagement}"
WORKERS="${SNAPUGC_LINK_WORKERS:-16}"
KAGGLE_WORKERS="${SNAPUGC_KAGGLE_WORKERS:-4}"
KAGGLE_NETRC="${KAGGLE_NETRC:-/workspace/kaggle.netrc}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SHUTDOWN_ON_EXIT="${SHUTDOWN_ON_EXIT:-1}"
export ROOT_DIR SUBSET_CSV VIDEO_DIR OUT_DIR LOG_FILE RESET_LOG RESUME_PREDICTIONS EXPORT_ARTIFACTS ARTIFACT_DIR ARTIFACT_SHARD_SIZE ALLOW_PARTIAL_ARTIFACT_RESUME CHECKPOINT_DIR OFFICIAL_REPO_DIR WORKERS KAGGLE_WORKERS KAGGLE_NETRC
export SNAPUGC_LINK_WORKERS="$WORKERS"
export SNAPUGC_KAGGLE_WORKERS="$KAGGLE_WORKERS"

mkdir -p "$VIDEO_DIR" "$OUT_DIR" "$(dirname "$LOG_FILE")"
if [[ "$RESET_LOG" == "1" ]]; then
  rm -f "$LOG_FILE"
fi

shutdown_vm() {
  local status=$?
  echo "FINAL_STATUS=${status} $(date -Is)" | tee -a "$LOG_FILE"
  if [[ "$SHUTDOWN_ON_EXIT" == "1" ]]; then
    echo "Shutting down VM now." | tee -a "$LOG_FILE"
    sudo shutdown -h now || true
  fi
  exit "$status"
}
trap shutdown_vm EXIT

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START OFFICIAL BALANCED 5000 FROM LINKS $(date -Is) ==="
echo "ROOT_DIR=$ROOT_DIR"
echo "SUBSET_CSV=$SUBSET_CSV"
echo "VIDEO_DIR=$VIDEO_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "RESUME_PREDICTIONS=${RESUME_PREDICTIONS:-<none>}"
echo "EXPORT_ARTIFACTS=$EXPORT_ARTIFACTS"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "ARTIFACT_SHARD_SIZE=$ARTIFACT_SHARD_SIZE"
echo "ALLOW_PARTIAL_ARTIFACT_RESUME=$ALLOW_PARTIAL_ARTIFACT_RESUME"
echo "SNAPUGC_DATALOADER_WORKERS=${SNAPUGC_DATALOADER_WORKERS:-1}"
echo "SNAPUGC_MPLUG_CLIP_BATCH=${SNAPUGC_MPLUG_CLIP_BATCH:-4}"
echo "SNAPUGC_OFFICIAL_FRAME_BATCH=${SNAPUGC_OFFICIAL_FRAME_BATCH:-24}"
echo "LINK_WORKERS=$WORKERS"
echo "KAGGLE_WORKERS=$KAGGLE_WORKERS"

"$PYTHON_BIN" - <<'PY'
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv_path = Path(os.environ["SUBSET_CSV"])
video_dir = Path(os.environ["VIDEO_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
workers = int(os.environ.get("SNAPUGC_LINK_WORKERS", os.environ.get("WORKERS", "16")))
failed_path = out_dir / "failed_link_downloads.txt"

rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = str(row["Id"])
        url = row.get("Download_link", "")
        if not url:
            raise ValueError(f"Missing Download_link for {vid}")
        rows.append((vid, url))

def exists_ok(vid: str) -> bool:
    p = video_dir / f"{vid}.mp4"
    return p.exists() and p.stat().st_size > 0

missing = [(vid, url) for vid, url in rows if not exists_ok(vid)]
print(f"rows={len(rows)} existing={len(rows)-len(missing)} missing={len(missing)}", flush=True)

def download_one(item):
    vid, url = item
    target = video_dir / f"{vid}.mp4"
    part = video_dir / f"{vid}.mp4.part"
    if exists_ok(vid):
        return vid, True, "exists"
    if part.exists():
        part.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "20",
        "--max-time",
        "240",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--retry-connrefused",
        "-A",
        "Mozilla/5.0",
        url,
        "-o",
        str(part),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and part.exists() and part.stat().st_size > 0:
        part.replace(target)
        return vid, True, "downloaded"
    if part.exists():
        part.unlink()
    msg = result.stderr.strip().replace("\n", " ")[:300] or f"curl_exit_{result.returncode}"
    return vid, False, msg

ok = 0
failed = []
start = time.time()
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(download_one, item) for item in missing]
    for i, fut in enumerate(as_completed(futures), 1):
        vid, success, msg = fut.result()
        if success:
            ok += 1
        else:
            failed.append((vid, msg))
            print(f"FAILED_VIDEO {vid} {msg}", flush=True)
        if i % 50 == 0 or i == len(futures):
            elapsed = max(time.time() - start, 1e-6)
            rate = i / elapsed
            final_existing = sum(1 for vid, _ in rows if exists_ok(vid))
            remaining = len(futures) - i
            eta_min = remaining / rate / 60 if rate > 0 else 0.0
            print(
                f"progress done={i}/{len(futures)} ok={ok} failed={len(failed)} "
                f"final_existing={final_existing}/{len(rows)} rate={rate:.2f}/s eta_min={eta_min:.1f}",
                flush=True,
            )

final_missing = [vid for vid, _ in rows if not exists_ok(vid)]
if failed or final_missing:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with failed_path.open("w", encoding="utf-8") as f:
        for vid, msg in failed:
            if vid not in seen:
                seen.add(vid)
                f.write(f"{vid}\t{msg}\n")
        for vid in final_missing:
            if vid not in seen:
                f.write(f"{vid}\tmissing_after_download\n")
    print(f"link_download_incomplete failed={len(failed)} final_missing={len(final_missing)} report={failed_path}", flush=True)
else:
    print(f"link_download_complete final_existing={len(rows)}", flush=True)

print("link_phase_done", flush=True)
PY

"$PYTHON_BIN" - <<'PY'
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

csv_path = Path(os.environ["SUBSET_CSV"])
video_dir = Path(os.environ["VIDEO_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
workers = int(os.environ.get("SNAPUGC_KAGGLE_WORKERS", "4"))
netrc = os.environ.get("KAGGLE_NETRC", "/workspace/kaggle.netrc")
failed_path = out_dir / "failed_kaggle_downloads.txt"
dataset = "nguyntuncng/snapugc-dataset"

rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        rows.append(str(row["Id"]))

def exists_ok(vid: str) -> bool:
    p = video_dir / f"{vid}.mp4"
    return p.exists() and p.stat().st_size > 0

missing = [vid for vid in rows if not exists_ok(vid)]
print(f"kaggle_fallback rows={len(rows)} existing={len(rows)-len(missing)} missing={len(missing)} workers={workers}", flush=True)
if not missing:
    print("download_complete final_existing=5000", flush=True)
    raise SystemExit(0)

def kaggle_url(vid: str) -> str:
    rel = f"train_videos/train_videos/{vid}.mp4"
    encoded = quote(rel, safe="")
    return (
        f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{encoded}"
        f"?filename={encoded}&raw=false"
    )

def download_one(vid: str):
    target = video_dir / f"{vid}.mp4"
    part = video_dir / f"{vid}.mp4.part"
    if exists_ok(vid):
        return vid, True, "exists"
    if part.exists():
        part.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--max-time",
        "900",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--retry-connrefused",
        "--netrc-file",
        netrc,
        kaggle_url(vid),
        "-o",
        str(part),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and part.exists() and part.stat().st_size > 0:
        part.replace(target)
        return vid, True, "downloaded"
    if part.exists():
        part.unlink()
    msg = result.stderr.strip().replace("\n", " ")[:300] or f"curl_exit_{result.returncode}"
    return vid, False, msg

ok = 0
failed = []
start = time.time()
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(download_one, vid) for vid in missing]
    for i, fut in enumerate(as_completed(futures), 1):
        vid, success, msg = fut.result()
        if success:
            ok += 1
        else:
            failed.append((vid, msg))
            print(f"FAILED_KAGGLE_VIDEO {vid} {msg}", flush=True)
        if i % 25 == 0 or i == len(futures):
            elapsed = max(time.time() - start, 1e-6)
            rate = i / elapsed
            final_existing = sum(1 for vid in rows if exists_ok(vid))
            remaining = len(futures) - i
            eta_min = remaining / rate / 60 if rate > 0 else 0.0
            print(
                f"kaggle_progress done={i}/{len(futures)} ok={ok} failed={len(failed)} "
                f"final_existing={final_existing}/{len(rows)} rate={rate:.2f}/s eta_min={eta_min:.1f}",
                flush=True,
            )

final_missing = [vid for vid in rows if not exists_ok(vid)]
if failed or final_missing:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with failed_path.open("w", encoding="utf-8") as f:
        for vid, msg in failed:
            if vid not in seen:
                seen.add(vid)
                f.write(f"{vid}\t{msg}\n")
        for vid in final_missing:
            if vid not in seen:
                f.write(f"{vid}\tmissing_after_kaggle_download\n")
    print(f"kaggle_download_failed failed={len(failed)} final_missing={len(final_missing)} report={failed_path}", flush=True)
    sys.exit(2)

print(f"download_complete final_existing={len(rows)}", flush=True)
PY

echo "=== PREPARE CHECKPOINT LINKS $(date -Is) ==="
mkdir -p "${OFFICIAL_REPO_DIR}/ECR_inference/checkpoints"
for name in EVQA.pth net_distort6_g_latest.pth r3d18_K_200ep.pth mPLUG2_MSRVTT_Caption.pth ViT-L-14.tar efficientnet_v2_s_21k_ft1k-dbb43f38.pth; do
  if [[ ! -s "${CHECKPOINT_DIR}/${name}" ]]; then
    echo "Missing checkpoint ${CHECKPOINT_DIR}/${name}"
    exit 3
  fi
  ln -sf "${CHECKPOINT_DIR}/${name}" "${OFFICIAL_REPO_DIR}/ECR_inference/checkpoints/${name}"
done
ls -lh "${OFFICIAL_REPO_DIR}/ECR_inference/checkpoints"

echo "=== RUN OFFICIAL BALANCED 5000 $(date -Is) ==="
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export SNAPUGC_OFFICIAL_FRAME_BATCH="${SNAPUGC_OFFICIAL_FRAME_BATCH:-12}"

RUN_CSV="$SUBSET_CSV"
MONITOR_SEED_ARGS=()
if [[ -n "$RESUME_PREDICTIONS" ]]; then
  if [[ "$EXPORT_ARTIFACTS" == "1" && "$ALLOW_PARTIAL_ARTIFACT_RESUME" != "1" ]]; then
    echo "Refusing artifact export with RESUME_PREDICTIONS because seed rows do not contain hidden/attention artifacts."
    echo "Unset RESUME_PREDICTIONS to run all 5000 from the beginning, or set ALLOW_PARTIAL_ARTIFACT_RESUME=1 for scalar-only seed rows."
    exit 5
  fi
  if [[ ! -s "$RESUME_PREDICTIONS" ]]; then
    echo "RESUME_PREDICTIONS file does not exist or is empty: $RESUME_PREDICTIONS"
    exit 4
  fi
  RUN_CSV="${OUT_DIR}/resume_remaining_after_seed.csv"
  export RUN_CSV
  "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

subset_csv = Path(os.environ["SUBSET_CSV"])
seed_csv = Path(os.environ["RESUME_PREDICTIONS"])
run_csv = Path(os.environ["RUN_CSV"])

with seed_csv.open("r", encoding="utf-8", newline="") as f:
    seed_ids = {str(row["Id"]) for row in csv.DictReader(f)}

with subset_csv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"Empty CSV: {subset_csv}")
    remaining = [row for row in reader if str(row["Id"]) not in seed_ids]

run_csv.parent.mkdir(parents=True, exist_ok=True)
with run_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(remaining)

print(
    f"resume_seed={len(seed_ids)} remaining_rows={len(remaining)} run_csv={run_csv}",
    flush=True,
)
PY
  MONITOR_SEED_ARGS=(--seed-predictions "$RESUME_PREDICTIONS")
fi

"$PYTHON_BIN" scripts/monitor_gcp_official_partial.py \
  --log "$LOG_FILE" \
  --labels-csv "$SUBSET_CSV" \
  --out-dir "$OUT_DIR" \
  --target-n 5000 \
  --every-n 500 \
  --poll-seconds 60 \
  "${MONITOR_SEED_ARGS[@]}" \
  --no-stop > "${OUT_DIR}/monitor_every_500.log" 2>&1 &
MONITOR_PID=$!
echo "Started partial monitor pid=${MONITOR_PID}"

RUN_ARGS=()
if [[ "$EXPORT_ARTIFACTS" == "1" ]]; then
  RUN_ARGS=(--export-artifacts --artifact-dir "$ARTIFACT_DIR" --artifact-shard-size "$ARTIFACT_SHARD_SIZE")
fi

"$PYTHON_BIN" scripts/run_official_snapugc_evqa.py \
  --official-repo-dir "$OFFICIAL_REPO_DIR" \
  --videos-dir "$VIDEO_DIR" \
  --csv-file "$RUN_CSV" \
  --out-dir "$OUT_DIR" \
  --python "$PYTHON_BIN" \
  "${RUN_ARGS[@]}"

echo "=== DONE $(date -Is) ==="

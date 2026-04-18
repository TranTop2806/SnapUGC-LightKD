"""Robust Qwen caption extraction for bounded Kaggle runs.

Qwen runs in child processes so a CUDA device-side assert cannot poison the
main feature extraction process. The parent requires captions for every selected
video and never silently skips failed samples.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from typing import Dict, List, Sequence

from tqdm.auto import tqdm

from .extraction import QWEN_DEVICE, QwenCaptioner, VideoItem, VideoReaderHelper, build_work


def load_caption_map(path: str) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.values()
    else:
        rows = payload
    result: Dict[str, Dict] = {}
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if video_id:
            result[video_id] = row
    return result


def save_caption_map(rows: Dict[str, Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def read_batch(path: str) -> List[VideoItem]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [VideoItem(**row) for row in rows]


def worker_main(args) -> int:
    items = read_batch(args.batch_json)
    existing = load_caption_map(args.out)
    reader = VideoReaderHelper()
    qwen = QwenCaptioner(args.qwen_model, device=args.qwen_device, enabled=True, strict=True)

    for item in tqdm(items, desc="Qwen worker", unit="video"):
        if item.video_id in existing and str(existing[item.video_id].get("qwen_caption", "")).strip():
            continue
        frames, _ = reader.read_frames(item.video_path, args.num_frames)
        if not frames:
            raise RuntimeError(f"empty video: {item.video_id}")
        caption, rationale, events = qwen.caption_and_rationale(frames, item.title, item.description)
        if not caption.strip():
            raise RuntimeError(f"Qwen produced empty caption for video_id={item.video_id}")
        existing[item.video_id] = {
            "video_id": item.video_id,
            "video_path": item.video_path,
            "qwen_caption": caption,
            "qwen_engagement_rationale": rationale,
            "qwen_visual_events": events,
        }
        save_caption_map(existing, args.out)
    return 0


def run_worker(items: Sequence[VideoItem], args, device: str) -> Dict[str, Dict]:
    with tempfile.TemporaryDirectory(prefix="snapugc_qwen_") as tmpdir:
        batch_path = os.path.join(tmpdir, "batch.json")
        out_path = os.path.join(tmpdir, "captions.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in items], f, ensure_ascii=False)

        cmd = [
            sys.executable,
            os.path.abspath(sys.argv[0]),
            "--worker",
            "--batch-json",
            batch_path,
            "--out",
            out_path,
            "--qwen-model",
            args.qwen_model,
            "--qwen-device",
            device,
            "--num-frames",
            str(args.num_frames),
        ]
        completed = subprocess.run(cmd)
        partial = load_caption_map(out_path)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Qwen worker failed on {len(items)} video(s) using {device}. "
                f"Partial successes: {len(partial)}"
            )
        return partial


def process_items(items: Sequence[VideoItem], args, done: Dict[str, Dict], device: str):
    missing = [item for item in items if item.video_id not in done]
    if not missing:
        return
    if len(missing) > args.batch_size:
        for start in range(0, len(missing), args.batch_size):
            process_items(missing[start:start + args.batch_size], args, done, device)
        return

    try:
        done.update(run_worker(missing, args, device))
        save_caption_map(done, args.out)
        return
    except Exception as exc:
        print(f"[Qwen parent] batch failed: {exc}", flush=True)
        done.update(load_caption_map(args.out))
        save_caption_map(done, args.out)

    still_missing = [item for item in missing if item.video_id not in done]
    if not still_missing:
        return
    if len(still_missing) == 1:
        item = still_missing[0]
        if args.cpu_fallback and device != "cpu":
            print(f"[Qwen parent] retrying {item.video_id} on CPU fallback", flush=True)
            done.update(run_worker([item], args, "cpu"))
            save_caption_map(done, args.out)
            return
        raise RuntimeError(f"Qwen failed for single video_id={item.video_id}; not skipping.")

    mid = max(1, len(still_missing) // 2)
    process_items(still_missing[:mid], args, done, device)
    process_items(still_missing[mid:], args, done, device)


def parent_main(args) -> int:
    existing = load_caption_map(args.out)
    items = build_work(args.csv, args.videos, args.max, set())
    if len(items) != args.max:
        raise RuntimeError(f"Expected exactly {args.max} matched videos, got {len(items)}.")

    print("=" * 80)
    print("Qwen Caption Extraction | SnapUGC-LightKD")
    print(f"Device: {args.qwen_device} | Videos: {len(items)} | Existing captions: {len(existing)}")
    print("=" * 80)

    t0 = time.time()
    process_items(items, args, existing, args.qwen_device)

    final = load_caption_map(args.out)
    missing = [item.video_id for item in items if item.video_id not in final]
    empty = [video_id for video_id, row in final.items() if not str(row.get("qwen_caption", "")).strip()]
    if missing or empty:
        raise RuntimeError(
            f"Qwen extraction incomplete. Missing={len(missing)} Empty={len(empty)}. "
            "This run does not skip samples."
        )
    elapsed = (time.time() - t0) / 60.0
    print(f"Qwen captions complete: {len(final)}/{len(items)} | Time: {elapsed:.1f} min")
    print(f"Output: {args.out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Robust Qwen caption extraction")
    parser.add_argument("--csv")
    parser.add_argument("--videos")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=300)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--qwen-device", default=QWEN_DEVICE)
    parser.add_argument("--cpu-fallback", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch-json")
    args = parser.parse_args()

    if args.worker:
        if not args.batch_json:
            raise ValueError("--batch-json is required in worker mode")
        raise SystemExit(worker_main(args))

    if not args.csv or not args.videos:
        raise ValueError("--csv and --videos are required")
    raise SystemExit(parent_main(args))


if __name__ == "__main__":
    main()

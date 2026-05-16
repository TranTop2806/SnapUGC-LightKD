#!/usr/bin/env python3
"""Extract cheap deployable video quality and motion features with OpenCV."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "dark_fraction",
    "bright_fraction",
    "contrast_std",
    "laplacian_var_log",
    "noise_mean",
    "noise_std",
    "edge_density",
    "colorfulness",
    "motion_mean",
    "motion_std",
    "motion_p95",
    "frame_position",
    "aspect_ratio_half",
    "megapixels_half",
)


def sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return list(range(max_frames))
    if frame_count <= max_frames:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, max_frames).round().astype(int).tolist()


def colorfulness_score(frame: np.ndarray) -> float:
    b, g, r = cv2.split(frame.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def frame_features(
    frame: np.ndarray,
    *,
    prev_gray: np.ndarray | None,
    frame_position: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(np.float32) - smooth.astype(np.float32))
    edges = cv2.Canny(gray, 80, 160)
    if prev_gray is None:
        motion = np.zeros_like(gray, dtype=np.float32)
    else:
        if prev_gray.shape != gray.shape:
            prev_gray = cv2.resize(prev_gray, (width, height), interpolation=cv2.INTER_AREA)
        motion = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))

    features = np.asarray(
        [
            value.mean() / 255.0,
            value.std() / 128.0,
            saturation.mean() / 255.0,
            saturation.std() / 128.0,
            (value < 30).mean(),
            (value > 225).mean(),
            gray.std() / 128.0,
            np.log1p(lap_var) / 12.0,
            noise.mean() / 255.0,
            noise.std() / 128.0,
            (edges > 0).mean(),
            colorfulness_score(frame) / 255.0,
            motion.mean() / 255.0,
            motion.std() / 128.0,
            np.percentile(motion, 95) / 255.0,
            frame_position,
            (width / max(height, 1)) / 2.0,
            (width * height / 1_000_000.0) / 2.0,
        ],
        dtype=np.float32,
    )
    return features, gray


def extract_one(args: tuple[str, str, int]) -> tuple[str, np.ndarray, np.ndarray, str]:
    video_id, video_path, max_frames = args
    features = np.zeros((max_frames, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((max_frames,), dtype=np.bool_)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_id, features, mask, "open_failed"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(frame_count, max_frames)
    prev_gray = None
    status = "ok"
    for out_idx, frame_idx in enumerate(indices[:max_frames]):
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            status = "partial_read_failed"
            continue
        position = frame_idx / max(frame_count - 1, 1) if frame_count > 1 else out_idx / max(max_frames - 1, 1)
        features[out_idx], prev_gray = frame_features(
            frame,
            prev_gray=prev_gray,
            frame_position=float(position),
        )
        mask[out_idx] = True
    cap.release()
    if not mask.any():
        status = "no_frames"
    return video_id, features, mask, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)
    ids = [str(value) for value in labels["Id"].tolist()]
    if args.limit:
        ids = ids[: args.limit]
    video_dir = Path(args.video_dir)
    jobs = [(video_id, str(video_dir / f"{video_id}.mp4"), args.max_frames) for video_id in ids]

    results = []
    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            results.append(extract_one(job))
            if idx % 250 == 0:
                print(f"processed {idx}/{len(jobs)}", flush=True)
    else:
        with Pool(processes=args.workers) as pool:
            for idx, result in enumerate(pool.imap(extract_one, jobs, chunksize=16), start=1):
                results.append(result)
                if idx % 250 == 0:
                    print(f"processed {idx}/{len(jobs)}", flush=True)

    out_ids = [item[0] for item in results]
    quality = np.stack([item[1] for item in results]).astype(np.float32)
    mask = np.stack([item[2] for item in results]).astype(np.bool_)
    statuses = [item[3] for item in results]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.asarray(out_ids),
        quality_features=quality,
        quality_mask=mask,
        feature_names=np.asarray(FEATURE_NAMES),
        statuses=np.asarray(statuses),
    )
    summary = {
        "out": str(out_path),
        "n": len(out_ids),
        "feature_dim": len(FEATURE_NAMES),
        "max_frames": args.max_frames,
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "valid_video_fraction": float(mask.any(axis=1).mean()) if len(mask) else 0.0,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract YAMNet audio logits from SnapUGC videos."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def extract_audio_to_wav(video_path: str, sr: int = 16000) -> np.ndarray:
    """Extract mono audio from video using ffmpeg, resample to target rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(sr),
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Read wav file
        import wave
        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return audio
    finally:
        os.unlink(tmp_path)


def run_yamnet(model, audio: np.ndarray, sr: int = 16000):
    """Run YAMNet on audio waveform. Returns (scores, embeddings, spectrogram)."""
    scores, embeddings, spectrogram = model(audio)
    return scores.numpy(), embeddings.numpy(), spectrogram.numpy()


def pool_scores(scores: np.ndarray) -> dict[str, np.ndarray]:
    """Pool YAMNet scores over time."""
    return {
        "mean": scores.mean(axis=0).astype(np.float32),
        "max": scores.max(axis=0).astype(np.float32),
        "std": scores.std(axis=0).astype(np.float32),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)
    ids = [str(v) for v in labels["Id"].tolist()]
    if args.limit:
        ids = ids[: args.limit]
    
    video_dir = Path(args.video_dir)
    
    # Load YAMNet model
    print("Loading YAMNet model from TF-Hub...", flush=True)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet loaded.", flush=True)
    
    results = []
    for idx, video_id in enumerate(ids, start=1):
        video_path = str(video_dir / f"{video_id}.mp4")
        try:
            audio = extract_audio_to_wav(video_path, sr=16000)
            if len(audio) == 0:
                scores = np.zeros((1, 521), dtype=np.float32)
            else:
                scores, _, _ = run_yamnet(yamnet_model, audio, sr=16000)
            pooled = pool_scores(scores)
            results.append((video_id, pooled, "ok"))
        except Exception as e:
            print(f"Error processing {video_id}: {e}", flush=True)
            pooled = {
                "mean": np.zeros(521, dtype=np.float32),
                "max": np.zeros(521, dtype=np.float32),
                "std": np.zeros(521, dtype=np.float32),
            }
            results.append((video_id, pooled, str(e)))
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(ids)}", flush=True)
    
    # Save as npz
    out_ids = [r[0] for r in results]
    mean_scores = np.stack([r[1]["mean"] for r in results]).astype(np.float32)
    max_scores = np.stack([r[1]["max"] for r in results]).astype(np.float32)
    std_scores = np.stack([r[1]["std"] for r in results]).astype(np.float32)
    statuses = [r[2] for r in results]
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.asarray(out_ids),
        quality_features=mean_scores,  # use mean as primary
        quality_max=max_scores,
        quality_std=std_scores,
        feature_names=np.asarray([f"yamnet_cls_{i}" for i in range(521)]),
        statuses=np.asarray(statuses),
    )
    
    summary = {
        "out": str(out_path),
        "n": len(out_ids),
        "feature_dim": 521,
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

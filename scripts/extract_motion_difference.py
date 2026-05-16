#!/usr/bin/env python3
"""Extract temporal motion difference features from frame_fusion_feature artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _unpack_ragged(npz, key, row_idx):
    flat_key = f"{key}_flat"
    offsets_key = f"{key}_offsets"
    shapes_key = f"{key}_shapes"
    if flat_key not in npz or offsets_key not in npz or shapes_key not in npz:
        return np.zeros((0,), dtype=np.float32)
    offsets = npz[offsets_key]
    shapes = npz[shapes_key]
    start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
    shape = tuple(int(x) for x in shapes[row_idx] if int(x) > 0)
    values = npz[flat_key][start:end]
    if not shape or values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return values.reshape(shape).astype(np.float32, copy=False)


def compute_motion_features(frame_fusion: np.ndarray) -> np.ndarray:
    """Compute temporal motion/difference features from frame fusion vectors.
    
    Input: (T, D) frame_fusion_feature
    Output: (T, K) motion features per frame/clip
    """
    T = frame_fusion.shape[0]
    if T <= 1:
        # Only one clip: return zeros
        return np.zeros((max(1, T), 5), dtype=np.float32)
    
    # L2 distance between consecutive frames
    l2_diff = np.linalg.norm(np.diff(frame_fusion, axis=0), axis=1, keepdims=True)
    
    # Cosine distance between consecutive frames
    normed = frame_fusion / (np.linalg.norm(frame_fusion, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(normed[:-1] * normed[1:], axis=1, keepdims=True)
    cos_dist = 1.0 - cos_sim
    
    # Frame-to-first and frame-to-last distances
    dist_to_first = np.linalg.norm(frame_fusion - frame_fusion[0], axis=1, keepdims=True)
    dist_to_last = np.linalg.norm(frame_fusion - frame_fusion[-1], axis=1, keepdims=True)
    
    # Position in sequence (normalized)
    position = np.arange(T).reshape(-1, 1) / max(T - 1, 1)
    
    # Pad l2_diff, cos_dist to length T by prepending zeros
    l2_padded = np.concatenate([np.zeros((1, 1), dtype=np.float32), l2_diff], axis=0)
    cos_padded = np.concatenate([np.zeros((1, 1), dtype=np.float32), cos_dist], axis=0)
    
    features = np.concatenate([l2_padded, cos_padded, dist_to_first, dist_to_last, position], axis=1)
    return features.astype(np.float32, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    shard_paths = sorted(artifact_dir.glob("official_teacher_artifacts_*.npz"))
    
    all_ids = []
    all_features = []
    all_masks = []
    
    for shard_path in shard_paths:
        with np.load(shard_path) as npz:
            ids = [str(x) for x in npz["ids"]]
            for i, vid in enumerate(ids):
                frame_fusion = _unpack_ragged(npz, "frame_fusion_feature", i)
                motion = compute_motion_features(frame_fusion)
                all_ids.append(vid)
                all_features.append(motion)
                all_masks.append(np.ones(motion.shape[0], dtype=bool))
    
    # Pad to max length
    max_len = max(f.shape[0] for f in all_features)
    feature_dim = all_features[0].shape[1]
    padded = np.zeros((len(all_features), max_len, feature_dim), dtype=np.float32)
    mask = np.zeros((len(all_features), max_len), dtype=bool)
    
    for i, feat in enumerate(all_features):
        n = feat.shape[0]
        padded[i, :n] = feat
        mask[i, :n] = True
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.asarray(all_ids),
        quality_features=padded,  # reuse key for compatibility
        quality_mask=mask,
        feature_names=np.asarray(["l2_diff", "cos_dist", "dist_to_first", "dist_to_last", "position"]),
    )
    
    summary = {
        "out": str(out_path),
        "n": len(all_ids),
        "feature_dim": feature_dim,
        "max_len": max_len,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a tiny v2-compatible synthetic feature file for smoke tests."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="features/synthetic_v2.json")
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    data = []
    for i in range(args.n):
        clip = rng.normal(size=(16, 512)).astype("float32")
        motion = rng.normal(size=(4, 512)).astype("float32")
        audio = rng.normal(size=1024).astype("float32")
        text = rng.normal(size=768).astype("float32")
        caption = rng.normal(size=768).astype("float32")
        rationale = rng.normal(size=768).astype("float32")
        technical = float(rng.uniform(0.2, 0.9))
        aesthetic = float(rng.uniform(0.2, 0.9))
        ecr = float(np.clip(
            0.20 * clip[:, 0].mean()
            + 0.15 * audio[:8].mean()
            + 0.20 * text[:8].mean()
            + 0.25 * technical
            + 0.20 * aesthetic,
            0.0,
            1.0,
        ))
        data.append({
            "video_id": f"synthetic_{i:04d}",
            "ecr": ecr,
            "clip_frame_embeddings": clip.tolist(),
            "motion_clip_embeddings": motion.tolist(),
            "yamnet_embedding_mean": audio.tolist(),
            "metadata_text_embedding": text.tolist(),
            "caption_embedding": caption.tolist(),
            "rationale_embedding": rationale.tolist(),
            "dover_scores": {
                "technical": technical,
                "aesthetic": aesthetic,
                "overall": (technical + aesthetic) / 2.0,
            },
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"Wrote {len(data)} samples to {out}")


if __name__ == "__main__":
    main()

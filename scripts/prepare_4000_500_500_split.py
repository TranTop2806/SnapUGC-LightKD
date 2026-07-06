#!/usr/bin/env python3
"""Create one deterministic 4000/500/500 split from the 5000-video pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{video_id}\n" for video_id in ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)
    ids = labels["Id"].astype(str).tolist()
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError(f"Expected 5000 unique IDs, found {len(ids)} rows/{len(set(ids))} unique")

    shuffled = np.asarray(ids, dtype=object)
    np.random.default_rng(args.seed).shuffle(shuffled)
    test_ids = shuffled[:500].tolist()
    val_ids = shuffled[500:1000].tolist()
    train_ids = shuffled[1000:].tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ids(out_dir / "train_ids.txt", train_ids)
    write_ids(out_dir / "val_ids.txt", val_ids)
    write_ids(out_dir / "test_ids.txt", test_ids)
    manifest = {
        "seed": args.seed,
        "source_labels": str(Path(args.labels_csv).resolve()),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "scope": "Internal student split; teacher artifacts were precomputed for all 5000 videos.",
        "limitation": "Not teacher-held-out; the released teacher was trained on the source pool.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

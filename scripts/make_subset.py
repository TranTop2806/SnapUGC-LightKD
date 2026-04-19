#!/usr/bin/env python3
"""Build an ECR-balanced CSV/video subset for bounded runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.subset import make_ecr_balanced_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Input train CSV")
    parser.add_argument("--videos", required=True, help="Input video directory")
    parser.add_argument("--out-csv", required=True, help="Output subset CSV")
    parser.add_argument("--out-videos", required=True, help="Output subset video directory")
    parser.add_argument("--max", type=int, required=True, help="Maximum selected videos")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--bins", type=int, default=10, help="Number of ECR quantile bins")
    parser.add_argument("--reset", action="store_true", help="Clear output video directory first")
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlinking videos")
    args = parser.parse_args()

    _, summary = make_ecr_balanced_subset(
        csv_path=args.csv,
        video_dir=args.videos,
        out_csv=args.out_csv,
        out_video_dir=args.out_videos,
        max_videos=args.max,
        seed=args.seed,
        bins=args.bins,
        reset=args.reset,
        copy=args.copy,
    )

    print("ECR-balanced subset created")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

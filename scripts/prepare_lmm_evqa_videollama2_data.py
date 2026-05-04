#!/usr/bin/env python3
"""Prepare SnapUGC JSON files for the official LMM-EVQA VideoLLaMA2 branch.

This keeps the data schema and prompt used by Sun et al.'s official
`VideoLLaMA2-audio_visual/prepare_dataset.py`, while making paths configurable
for Kaggle or cloud machines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROMPT = (
    "<video>\n"
    "How would you judge the engagement continuation rate of the given content, "
    "where engagement continuation rate represents the probability of watch time "
    "exceeding 5 seconds. The title of the video is {title}, and the description "
    "of the video is {description}"
)


def norm_text(value) -> str:
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    return text if text else "None"


def resolve_video(video_root: Path, video_id: str) -> Path | None:
    candidates = [
        video_root / f"{video_id}.mp4",
        video_root / "train_videos" / f"{video_id}.mp4",
        video_root / "videos" / f"{video_id}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def make_entry(row: pd.Series, video_path: Path) -> dict:
    title = norm_text(row.get("Title"))
    description = norm_text(row.get("Description"))
    ecr = float(row["ECR"])
    return {
        "id": str(row["Id"]),
        "ECR": ecr * 100.0,
        "video": str(video_path),
        "conversations": [
            {"from": "human", "value": PROMPT.format(title=title, description=description)},
            {"from": "gpt", "value": "The engagement continuation rate of the video."},
        ],
    }


def dump_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="SnapUGC train_data.csv with Id/Title/Description/ECR")
    parser.add_argument("--video-root", required=True, help="Directory containing mp4 videos")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-null-metadata",
        action="store_true",
        help="Optional ablation only. Leave off for paper-faithful reproduction.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(csv_path)
    required = {"Id", "Title", "Description", "ECR"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    if args.drop_null_metadata:
        df = df[df["Title"].notna() & df["Description"].notna()].copy()

    entries: list[dict] = []
    skipped: list[str] = []
    for _, row in df.iterrows():
        video_id = str(row["Id"])
        path = resolve_video(video_root, video_id)
        if path is None:
            skipped.append(video_id)
            continue
        entries.append(make_entry(row, path))
        if args.max_samples and len(entries) >= args.max_samples:
            break

    if not entries:
        raise RuntimeError(f"No usable videos found under {video_root}")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(entries))
    split = int(len(indices) * (1.0 - args.val_ratio))
    train_entries = [entries[i] for i in indices[:split]]
    val_entries = [entries[i] for i in indices[split:]]
    all_entries = train_entries + val_entries

    dump_json(out_dir / "train.json", train_entries)
    dump_json(out_dir / "val.json", val_entries)
    dump_json(out_dir / "all.json", all_entries)

    pd.DataFrame(
        {
            "Id": [row["id"] for row in all_entries],
            "split": ["train"] * len(train_entries) + ["val"] * len(val_entries),
            "ECR": [row["ECR"] / 100.0 for row in all_entries],
            "video": [row["video"] for row in all_entries],
        }
    ).to_csv(out_dir / "split.csv", index=False)

    summary = {
        "source_csv": str(csv_path),
        "video_root": str(video_root),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "val_ratio": args.val_ratio,
        "drop_null_metadata": bool(args.drop_null_metadata),
        "n_total": len(all_entries),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "n_skipped_missing_video": len(skipped),
        "skipped_missing_video_preview": skipped[:20],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

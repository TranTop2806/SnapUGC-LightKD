"""Utilities for building ECR-balanced video subsets."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching column, case-insensitive."""
    lower_to_actual = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        found = lower_to_actual.get(candidate.lower())
        if found is not None:
            return found
    return None


def build_video_index(video_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Index videos by stem and filename so CSV ids can resolve robustly."""
    root = Path(video_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {root}")

    index: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        index.setdefault(path.stem, path)
        index.setdefault(path.name, path)
    return index


def attach_video_paths(
    df: pd.DataFrame,
    video_dir: str | os.PathLike[str],
    id_col: str,
    video_path_col: str = "_video_path",
) -> pd.DataFrame:
    """Keep rows whose id has a matching video file and attach source paths."""
    index = build_video_index(video_dir)
    rows = []
    for _, row in df.iterrows():
        video_id = str(row[id_col])
        path = index.get(video_id) or index.get(f"{video_id}.mp4")
        if path is None:
            continue
        item = row.copy()
        item[video_path_col] = str(path)
        rows.append(item)

    if not rows:
        raise RuntimeError(f"No CSV rows matched videos in {video_dir}")
    return pd.DataFrame(rows)


def _allocate_quotas(group_sizes: dict[int, int], total: int) -> dict[int, int]:
    """Allocate near-equal quotas, then fill spare capacity deterministically."""
    if not group_sizes:
        return {}

    group_ids = sorted(group_sizes)
    base = total // len(group_ids)
    quotas = {gid: min(size, base) for gid, size in group_sizes.items()}

    remaining = total - sum(quotas.values())
    while remaining > 0:
        progressed = False
        candidates = sorted(
            group_ids,
            key=lambda gid: (group_sizes[gid] - quotas[gid], group_sizes[gid]),
            reverse=True,
        )
        for gid in candidates:
            if quotas[gid] >= group_sizes[gid]:
                continue
            quotas[gid] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def stratified_sample_by_ecr(
    df: pd.DataFrame,
    ecr_col: str,
    max_rows: int,
    bins: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample rows across ECR quantiles instead of taking the first N rows."""
    if max_rows <= 0:
        raise ValueError("--max must be positive")
    if bins <= 0:
        raise ValueError("--bins must be positive")

    work = df.copy()
    work[ecr_col] = pd.to_numeric(work[ecr_col], errors="coerce")
    work = work.dropna(subset=[ecr_col]).reset_index(drop=True)
    if work.empty:
        raise RuntimeError(f"No rows have numeric ECR values in column {ecr_col!r}")

    if len(work) <= max_rows:
        return work.sample(frac=1, random_state=seed).reset_index(drop=True)

    unique_ecr = int(work[ecr_col].nunique())
    n_bins = min(int(bins), unique_ecr, max_rows)
    if n_bins < 2:
        return work.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    work["_ecr_bin"] = pd.qcut(
        work[ecr_col],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    work = work.dropna(subset=["_ecr_bin"]).copy()
    work["_ecr_bin"] = work["_ecr_bin"].astype(int)

    groups = {int(gid): group for gid, group in work.groupby("_ecr_bin", sort=True)}
    quotas = _allocate_quotas({gid: len(group) for gid, group in groups.items()}, max_rows)

    sampled_parts = []
    for gid, group in groups.items():
        take = quotas.get(gid, 0)
        if take <= 0:
            continue
        sampled_parts.append(group.sample(n=take, random_state=seed + gid))

    if not sampled_parts:
        raise RuntimeError("Stratified sampling produced no rows")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled.drop(columns=["_ecr_bin"], errors="ignore")


def materialize_subset_videos(
    df: pd.DataFrame,
    out_video_dir: str | os.PathLike[str],
    id_col: str,
    video_path_col: str = "_video_path",
    *,
    reset: bool = False,
    copy: bool = False,
) -> int:
    """Symlink or copy selected videos into a bounded subset directory."""
    out_dir = Path(out_video_dir)
    if reset and out_dir.exists():
        for path in out_dir.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for _, row in df.iterrows():
        src = Path(str(row[video_path_col]))
        if not src.exists():
            raise FileNotFoundError(f"Selected source video is missing: {src}")
        dst = out_dir / f"{row[id_col]}{src.suffix.lower()}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        count += 1
    return count


def ecr_summary(df: pd.DataFrame, ecr_col: str, bins: int = 10) -> dict[str, object]:
    """Return concise ECR distribution diagnostics for logging."""
    values = pd.to_numeric(df[ecr_col], errors="coerce").dropna()
    stats = {
        "count": int(values.shape[0]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }
    n_bins = min(int(bins), int(values.nunique()), int(values.shape[0]))
    if n_bins >= 2:
        counts = pd.qcut(values, q=n_bins, duplicates="drop").value_counts().sort_index()
        stats["quantile_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        stats["quantile_counts"] = {}
    return stats


def make_ecr_balanced_subset(
    *,
    csv_path: str | os.PathLike[str],
    video_dir: str | os.PathLike[str],
    out_csv: str | os.PathLike[str],
    out_video_dir: str | os.PathLike[str],
    max_videos: int,
    seed: int = 42,
    bins: int = 10,
    reset: bool = False,
    copy: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create an ECR-balanced subset CSV and matching subset video directory."""
    df = pd.read_csv(csv_path)
    id_col = find_first_column(df, ["Id", "id", "video_id", "videoid"])
    ecr_col = find_first_column(df, ["ECR", "engagement", "label", "target"])
    if id_col is None:
        raise ValueError("Could not find video id column in CSV.")
    if ecr_col is None:
        raise ValueError("Could not find ECR/target column in CSV.")

    available = attach_video_paths(df, video_dir, id_col)
    sampled = stratified_sample_by_ecr(
        available,
        ecr_col=ecr_col,
        max_rows=max_videos,
        bins=bins,
        seed=seed,
    )

    materialized = materialize_subset_videos(
        sampled,
        out_video_dir,
        id_col=id_col,
        reset=reset,
        copy=copy,
    )
    if materialized != len(sampled):
        raise RuntimeError(f"Materialized {materialized} videos for {len(sampled)} rows")

    output_df = sampled.drop(columns=["_video_path"], errors="ignore")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_csv, index=False)

    summary = ecr_summary(output_df, ecr_col=ecr_col, bins=bins)
    summary.update(
        {
            "available_rows": int(len(available)),
            "selected_rows": int(len(output_df)),
            "id_col": id_col,
            "ecr_col": ecr_col,
            "seed": int(seed),
            "bins": int(bins),
        }
    )
    return output_df, summary

#!/usr/bin/env python3
"""Analyze Proper KD auto-edit batch results and export charts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORE_COLS = ["True ECR", "Predicted ECR", "After-edit ECR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "results/proper_kd_auto_edit_100_normal/"
            "proper_kd_auto_edit_100_normal_results.csv"
        ),
    )
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path("data/official_5k_split"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/proper_kd_auto_edit_100_normal/analysis"),
    )
    parser.add_argument("--new-count", type=int, default=300)
    parser.add_argument(
        "--skip-new-vs-existing",
        action="store_true",
        help="Skip the existing-vs-new distribution chart when analyzing a filtered subset.",
    )
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [col for col in ["Id", *SCORE_COLS] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df["Id"] = df["Id"].astype(str)
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_official_ids(official_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not official_dir.exists():
        return ids

    for path in official_dir.glob("*"):
        if path.suffix.lower() == ".csv":
            try:
                sample = pd.read_csv(path, nrows=5)
                id_cols = [c for c in sample.columns if c.lower() == "id"]
                if not id_cols:
                    continue
                series = pd.read_csv(path, usecols=[id_cols[0]])[id_cols[0]]
                ids.update(series.dropna().astype(str).tolist())
            except Exception:
                continue
        elif path.suffix.lower() == ".txt":
            try:
                ids.update(line.strip() for line in path.read_text().splitlines() if line.strip())
            except Exception:
                continue
    return ids


def finite_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr_a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    return arr_a[mask], arr_b[mask]


def safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float | None:
    left, right = finite_pair(a, b)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    if method == "spearman":
        return float(pd.Series(left).corr(pd.Series(right), method="spearman"))
    return float(np.corrcoef(left, right)[0, 1])


def rmse(a: pd.Series, b: pd.Series) -> float | None:
    left, right = finite_pair(a, b)
    if len(left) == 0:
        return None
    return float(np.sqrt(np.mean((right - left) ** 2)))


def mae(a: pd.Series, b: pd.Series) -> float | None:
    left, right = finite_pair(a, b)
    if len(left) == 0:
        return None
    return float(np.mean(np.abs(right - left)))


def series_stats(s: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "q25": None,
            "q75": None,
            "max": None,
        }
    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.shape[0] > 1 else 0.0,
        "median": float(clean.median()),
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.edgecolor": "#1F2937",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 10,
            "legend.frameon": False,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
        }
    )


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def chart_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    colors = {
        "True ECR": "#2563EB",
        "Predicted ECR": "#DC2626",
        "After-edit ECR": "#16A34A",
    }
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(9.5, 5.5))
    for col in SCORE_COLS:
        plt.hist(
            df[col].dropna(),
            bins=bins,
            alpha=0.35,
            density=True,
            label=col,
            color=colors[col],
            edgecolor="white",
        )
    plt.title("ECR Score Distributions")
    plt.xlabel("ECR score")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.grid(axis="y")
    plt.legend()
    save_fig(out_dir / "ecr_distribution_overlay.png")


def chart_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    colors = ["#2563EB", "#DC2626", "#16A34A"]
    data = [df[col].dropna().to_numpy() for col in SCORE_COLS]
    plt.figure(figsize=(8.5, 5.0))
    box = plt.boxplot(data, tick_labels=SCORE_COLS, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.25, edgecolor=color)
    for median in box["medians"]:
        median.set(color="#111827", linewidth=1.6)
    plt.title("ECR Spread by Score Type")
    plt.ylabel("ECR score")
    plt.ylim(-0.02, 1.02)
    plt.grid(axis="y")
    save_fig(out_dir / "ecr_boxplot.png")


def chart_true_pred_scatter(df: pd.DataFrame, out_dir: Path, metrics: dict[str, Any]) -> None:
    true, pred = finite_pair(df["True ECR"], df["Predicted ECR"])
    err = np.abs(pred - true)
    plt.figure(figsize=(6.5, 6.2))
    sc = plt.scatter(true, pred, c=err, cmap="magma_r", s=28, alpha=0.78, edgecolors="none")
    plt.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.1, label="y = x")
    plt.title("True ECR vs Predicted ECR")
    plt.xlabel("True ECR")
    plt.ylabel("Predicted ECR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc="upper left")
    cbar = plt.colorbar(sc)
    cbar.set_label("Absolute error")
    label = (
        f"Pearson r={metrics['pred_vs_true']['pearson']:.3f}\n"
        f"MAE={metrics['pred_vs_true']['mae']:.3f}\n"
        f"RMSE={metrics['pred_vs_true']['rmse']:.3f}"
    )
    plt.text(
        0.04,
        0.96,
        label,
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    save_fig(out_dir / "true_vs_predicted_scatter.png")


def chart_pred_after_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    pred, after = finite_pair(df["Predicted ECR"], df["After-edit ECR"])
    delta = after - pred
    colors = np.where(delta >= 0, "#16A34A", "#DC2626")
    plt.figure(figsize=(6.5, 6.2))
    plt.scatter(pred, after, c=colors, s=28, alpha=0.76, edgecolors="none")
    plt.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.1, label="no change")
    plt.title("Predicted ECR Before vs After Auto Edit")
    plt.xlabel("Predicted ECR before edit")
    plt.ylabel("Predicted ECR after edit")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    improved = int((delta > 0).sum())
    total = int(delta.shape[0])
    plt.text(
        0.04,
        0.96,
        f"Improved: {improved}/{total} ({improved / total:.1%})\n"
        f"Mean delta: {delta.mean():+.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    plt.legend(loc="lower right")
    save_fig(out_dir / "predicted_vs_after_scatter.png")


def chart_delta_hist(df: pd.DataFrame, out_dir: Path) -> None:
    pred, after = finite_pair(df["Predicted ECR"], df["After-edit ECR"])
    delta = after - pred
    lim = max(0.15, float(np.nanmax(np.abs(delta))) if delta.size else 0.15)
    bins = np.linspace(-lim, lim, 31)
    plt.figure(figsize=(9.5, 5.0))
    plt.hist(delta, bins=bins, color="#7C3AED", alpha=0.72, edgecolor="white")
    plt.axvline(0, color="#111827", linestyle="--", linewidth=1.2)
    plt.axvline(delta.mean(), color="#F59E0B", linewidth=1.8, label=f"mean {delta.mean():+.3f}")
    plt.axvline(np.median(delta), color="#0891B2", linewidth=1.8, label=f"median {np.median(delta):+.3f}")
    plt.title("After-edit Delta Distribution")
    plt.xlabel("After-edit ECR - Predicted ECR")
    plt.ylabel("Video count")
    plt.grid(axis="y")
    plt.legend()
    save_fig(out_dir / "after_edit_delta_distribution.png")


def chart_deciles(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    work = df[SCORE_COLS].dropna().copy()
    work["decile"] = pd.qcut(work["True ECR"], q=10, labels=False, duplicates="drop") + 1
    grouped = (
        work.groupby("decile", as_index=False)
        .agg(
            true_mean=("True ECR", "mean"),
            pred_mean=("Predicted ECR", "mean"),
            after_mean=("After-edit ECR", "mean"),
            count=("True ECR", "size"),
            true_min=("True ECR", "min"),
            true_max=("True ECR", "max"),
        )
        .sort_values("decile")
    )
    x = grouped["decile"].to_numpy()
    plt.figure(figsize=(10.0, 5.5))
    plt.plot(x, grouped["true_mean"], marker="o", color="#2563EB", label="True mean")
    plt.plot(x, grouped["pred_mean"], marker="o", color="#DC2626", label="Predicted mean")
    plt.plot(x, grouped["after_mean"], marker="o", color="#16A34A", label="After-edit mean")
    plt.title("Mean Scores by True ECR Decile")
    plt.xlabel("True ECR decile (low to high)")
    plt.ylabel("Mean ECR score")
    plt.ylim(0, 1)
    plt.xticks(x, [f"D{int(i)}" for i in x])
    plt.grid(True)
    plt.legend()
    save_fig(out_dir / "ecr_by_true_decile.png")
    return grouped


def chart_new_vs_previous(df: pd.DataFrame, out_dir: Path, new_count: int) -> None:
    new_count = min(new_count, len(df))
    previous = df.iloc[: len(df) - new_count]
    new = df.iloc[len(df) - new_count :]
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(9.5, 5.0))
    if not previous.empty:
        plt.hist(
            previous["True ECR"].dropna(),
            bins=bins,
            alpha=0.45,
            density=True,
            color="#0EA5E9",
            label=f"Existing {len(previous)}",
            edgecolor="white",
        )
    plt.hist(
        new["True ECR"].dropna(),
        bins=bins,
        alpha=0.45,
        density=True,
        color="#F97316",
        label=f"New {len(new)}",
        edgecolor="white",
    )
    plt.title("True ECR Distribution: Existing vs Newly Added")
    plt.xlabel("True ECR")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.grid(axis="y")
    plt.legend()
    save_fig(out_dir / "new300_vs_existing_true_distribution.png")


def build_summary(
    df: pd.DataFrame,
    deciles: pd.DataFrame,
    official_ids: set[str],
    new_count: int,
    out_dir: Path,
) -> dict[str, Any]:
    new_count = min(new_count, len(df))
    existing = df.iloc[: len(df) - new_count]
    new = df.iloc[len(df) - new_count :]
    duplicate_count = int(df["Id"].duplicated().sum())
    official_overlap = sorted(set(df["Id"]) & official_ids)
    existing_new_overlap = sorted(set(existing["Id"]) & set(new["Id"])) if not existing.empty else []

    pred_minus_true = df["Predicted ECR"] - df["True ECR"]
    after_minus_pred = df["After-edit ECR"] - df["Predicted ECR"]

    pred_vs_true = {
        "pearson": safe_corr(df["True ECR"], df["Predicted ECR"], "pearson"),
        "spearman": safe_corr(df["True ECR"], df["Predicted ECR"], "spearman"),
        "mae": mae(df["True ECR"], df["Predicted ECR"]),
        "rmse": rmse(df["True ECR"], df["Predicted ECR"]),
        "bias_mean_pred_minus_true": float(pred_minus_true.mean()),
        "bias_median_pred_minus_true": float(pred_minus_true.median()),
    }
    after_delta = {
        "mean": float(after_minus_pred.mean()),
        "median": float(after_minus_pred.median()),
        "std": float(after_minus_pred.std(ddof=1)),
        "min": float(after_minus_pred.min()),
        "q25": float(after_minus_pred.quantile(0.25)),
        "q75": float(after_minus_pred.quantile(0.75)),
        "max": float(after_minus_pred.max()),
        "improved_count": int((after_minus_pred > 0).sum()),
        "worsened_count": int((after_minus_pred < 0).sum()),
        "unchanged_count": int((after_minus_pred == 0).sum()),
        "improved_rate": float((after_minus_pred > 0).mean()),
    }

    summary = {
        "row_count": int(len(df)),
        "new_count_assumed": int(new_count),
        "existing_count_before_new": int(len(existing)),
        "duplicate_id_count": duplicate_count,
        "official_overlap_count": int(len(official_overlap)),
        "official_overlap_sample": official_overlap[:10],
        "existing_new_overlap_count": int(len(existing_new_overlap)),
        "existing_new_overlap_sample": existing_new_overlap[:10],
        "all_stats": {col: series_stats(df[col]) for col in SCORE_COLS},
        "new_stats": {col: series_stats(new[col]) for col in SCORE_COLS},
        "existing_stats": {col: series_stats(existing[col]) for col in SCORE_COLS},
        "pred_vs_true": pred_vs_true,
        "after_delta": after_delta,
        "deciles": deciles.to_dict(orient="records"),
        "chart_files": sorted(path.name for path in out_dir.glob("*.png")),
    }
    return to_jsonable(summary)


def write_summary(summary: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "analysis_summary.json"
    txt_path = out_dir / "analysis_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "Proper KD auto-edit batch analysis",
        "=" * 38,
        f"Rows: {summary['row_count']}",
        f"New rows assumed in latest append: {summary['new_count_assumed']}",
        f"Duplicate IDs: {summary['duplicate_id_count']}",
        f"Official split overlap: {summary['official_overlap_count']}",
        f"Existing/new overlap: {summary['existing_new_overlap_count']}",
        "",
        "All-score statistics",
    ]
    for col in SCORE_COLS:
        stats = summary["all_stats"][col]
        lines.append(
            f"- {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"median={stats['median']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
        )
    pred = summary["pred_vs_true"]
    delta = summary["after_delta"]
    lines.extend(
        [
            "",
            "Prediction quality against original ECR",
            f"- Pearson r: {pred['pearson']:.4f}",
            f"- Spearman r: {pred['spearman']:.4f}",
            f"- MAE: {pred['mae']:.4f}",
            f"- RMSE: {pred['rmse']:.4f}",
            f"- Mean bias (Predicted - True): {pred['bias_mean_pred_minus_true']:+.4f}",
            "",
            "After-edit movement",
            f"- Mean delta (After - Predicted): {delta['mean']:+.4f}",
            f"- Median delta: {delta['median']:+.4f}",
            f"- Improved: {delta['improved_count']} ({delta['improved_rate']:.1%})",
            f"- Worsened: {delta['worsened_count']}",
            f"- Min/max delta: {delta['min']:+.4f} / {delta['max']:+.4f}",
            "",
            "Charts",
        ]
    )
    for chart in summary["chart_files"]:
        lines.append(f"- {chart}")
    txt_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot()

    df = read_results(args.results)
    official_ids = load_official_ids(args.official_dir)

    chart_distributions(df, args.output_dir)
    chart_boxplot(df, args.output_dir)

    metrics_stub = {
        "pred_vs_true": {
            "pearson": safe_corr(df["True ECR"], df["Predicted ECR"], "pearson"),
            "mae": mae(df["True ECR"], df["Predicted ECR"]),
            "rmse": rmse(df["True ECR"], df["Predicted ECR"]),
        }
    }
    chart_true_pred_scatter(df, args.output_dir, metrics_stub)
    chart_pred_after_scatter(df, args.output_dir)
    chart_delta_hist(df, args.output_dir)
    deciles = chart_deciles(df, args.output_dir)
    if not args.skip_new_vs_existing and 0 < args.new_count < len(df):
        chart_new_vs_previous(df, args.output_dir, args.new_count)

    summary = build_summary(df, deciles, official_ids, args.new_count, args.output_dir)
    write_summary(summary, args.output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

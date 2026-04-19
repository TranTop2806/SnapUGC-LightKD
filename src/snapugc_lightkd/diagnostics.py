"""Diagnostic plots for bounded SnapUGC-LightKD runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_KEYS = ("teacher", "student_kd", "student_baseline")
METRIC_KEYS = ("plcc", "srcc", "final_score")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_feature_diagnostics(features_path: str | Path) -> pd.DataFrame:
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    rows = []
    for item in features:
        dover = item.get("dover_scores") or {}
        rows.append(
            {
                "video_id": item.get("video_id"),
                "ecr": _safe_float(item.get("ecr")),
                "dover_overall": _safe_float(dover.get("overall")),
                "dover_technical": _safe_float(dover.get("technical")),
                "dover_aesthetic": _safe_float(dover.get("aesthetic")),
                "caption_len": len(str(item.get("blip_caption") or item.get("caption") or "")),
            }
        )
    return pd.DataFrame(rows)


def load_metric_table(report: dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_KEYS:
        metrics = report.get(model_key) or {}
        row = {"model": model_key}
        for metric_key in METRIC_KEYS:
            row[metric_key] = _safe_float(metrics.get(metric_key))
        row["mse"] = _safe_float(metrics.get("mse"))
        row["mae"] = _safe_float(metrics.get("mae"))
        row["params"] = int(metrics.get("params", 0) or 0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_diagnostics(
    *,
    features_path: str | Path,
    report_path: str | Path,
    subset_csv: str | Path,
    out_dir: str | Path,
    bins: int = 10,
) -> list[Path]:
    """Write distribution plots, metric plots, CSV tables, and a summary JSON."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    feature_df = load_feature_diagnostics(features_path)
    subset_df = pd.read_csv(subset_csv)
    metric_df = load_metric_table(report)

    subset_df["ECR"] = pd.to_numeric(subset_df["ECR"], errors="coerce")
    ecr = subset_df["ECR"].dropna()
    metric_df.to_csv(out / "train_metrics.csv", index=False)
    feature_df.to_csv(out / "feature_diagnostics.csv", index=False)

    written: list[Path] = []

    plt.figure(figsize=(8, 5))
    plt.hist(ecr, bins=40, color="#315a8c", edgecolor="white", alpha=0.9)
    plt.axvline(ecr.mean(), color="#c43d2b", linestyle="--", linewidth=2, label=f"mean={ecr.mean():.3f}")
    plt.title("Subset ECR Distribution")
    plt.xlabel("ECR")
    plt.ylabel("Videos")
    plt.legend()
    plt.tight_layout()
    path = out / "diagnostic_ecr_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(path)

    q = min(int(bins), int(ecr.nunique()), int(len(ecr)))
    if q >= 2:
        counts = pd.qcut(ecr, q=q, duplicates="drop").value_counts().sort_index()
        labels = [f"Q{i + 1}" for i in range(len(counts))]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, counts.values, color="#58734d")
        plt.title("Selected Videos per ECR Quantile")
        plt.xlabel("ECR quantile")
        plt.ylabel("Videos")
        plt.tight_layout()
        path = out / "diagnostic_ecr_quantiles.png"
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(path)

    if feature_df["dover_overall"].notna().any():
        plt.figure(figsize=(7, 5))
        plt.scatter(
            feature_df["dover_overall"],
            feature_df["ecr"],
            s=12,
            alpha=0.45,
            color="#6b4f9a",
            edgecolors="none",
        )
        corr = feature_df[["dover_overall", "ecr"]].corr().iloc[0, 1]
        plt.title(f"DOVER Overall vs ECR (r={corr:.3f})")
        plt.xlabel("DOVER overall")
        plt.ylabel("ECR")
        plt.tight_layout()
        path = out / "diagnostic_dover_vs_ecr.png"
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(path)

    x = np.arange(len(metric_df))
    width = 0.24
    plt.figure(figsize=(9, 5))
    for i, metric_key in enumerate(METRIC_KEYS):
        plt.bar(x + (i - 1) * width, metric_df[metric_key], width=width, label=metric_key.upper())
    plt.xticks(x, metric_df["model"], rotation=10)
    plt.ylim(0, max(0.05, float(np.nanmax(metric_df[list(METRIC_KEYS)].to_numpy())) * 1.15))
    plt.title("Validation Metrics by Model")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    path = out / "diagnostic_train_metrics.png"
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(path)

    summary = {
        "subset_rows": int(len(subset_df)),
        "features_rows": int(len(feature_df)),
        "ecr": {
            "min": float(ecr.min()),
            "max": float(ecr.max()),
            "mean": float(ecr.mean()),
            "std": float(ecr.std(ddof=0)),
        },
        "best_final_score_model": str(
            metric_df.sort_values("final_score", ascending=False).iloc[0]["model"]
        ),
        "plots": [p.name for p in written],
    }
    with open(out / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return written

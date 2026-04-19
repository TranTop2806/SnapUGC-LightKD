#!/usr/bin/env python3
"""Run tabular baselines directly on extracted feature JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def as_vector(value, dim: int | None = None) -> np.ndarray:
    if value is None:
        arr = np.zeros(0, dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        arr = arr.reshape(-1)
    if dim is None:
        return arr
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    if arr.size > dim:
        arr = arr[:dim]
    return arr.astype(np.float32, copy=False)


def quality_values(item: dict) -> list[float]:
    quality = item.get("dover_scores") or item.get("quality_scores") or {}
    vals = []
    for key in ("technical", "aesthetic", "overall"):
        value = quality.get(key, 0.5)
        value = float(value)
        vals.append(value / 10.0 if value > 1.0 else value)
    return vals


def feature_groups(item: dict) -> dict[str, np.ndarray]:
    clip = item.get("clip_mean_embedding") or item.get("clip_frame_embeddings") or item.get("visual_emb")
    motion = item.get("motion_mean_embedding") or item.get("motion_clip_embeddings")
    audio = item.get("yamnet_embedding_mean") or item.get("audio_emb")
    text = item.get("metadata_text_embedding") or item.get("text_emb")
    caption = item.get("caption_embedding")
    rationale = item.get("rationale_embedding")

    numeric = np.asarray(
        [
            float(item.get("duration") or 0.0),
            float(item.get("fps") or 0.0),
            float(item.get("width") or 0.0),
            float(item.get("height") or 0.0),
            *quality_values(item),
        ],
        dtype=np.float32,
    )

    return {
        "quality": np.asarray(quality_values(item), dtype=np.float32),
        "numeric": numeric,
        "clip": as_vector(clip, 512),
        "motion": as_vector(motion, 512),
        "audio": as_vector(audio, 1024),
        "text": as_vector(text, 768),
        "caption": as_vector(caption, 768),
        "rationale": as_vector(rationale, 768),
    }


FEATURE_SETS = {
    "quality": ("quality",),
    "student": ("numeric", "quality", "clip", "audio", "text"),
    "teacher": ("numeric", "quality", "clip", "motion", "audio", "text", "caption", "rationale"),
    "text_only": ("text", "caption", "rationale"),
}


def build_matrix(data: list[dict], feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    groups = FEATURE_SETS[feature_set]
    rows, targets = [], []
    for item in data:
        if item.get("ecr") is None:
            continue
        item_groups = feature_groups(item)
        rows.append(np.concatenate([item_groups[name] for name in groups]).astype(np.float32))
        targets.append(float(item["ecr"]))
    return np.vstack(rows), np.asarray(targets, dtype=np.float32)


def split_indices(n: int, val_ratio: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    split = int(n * (1 - val_ratio))
    return indices[:split], indices[split:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        plcc = 0.0
    else:
        plcc = float(pearsonr(y_true, y_pred)[0])
    return {
        "plcc": plcc,
        "srcc": float(spearmanr(y_true, y_pred)[0]),
        "ktau": float(kendalltau(y_true, y_pred)[0]),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "final_score": float(0.6 * spearmanr(y_true, y_pred)[0] + 0.4 * plcc),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
    }


def build_models(n_train: int, n_features: int, quick: bool = False):
    alphas = np.logspace(-3, 4, 12)
    models = {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
    }
    if not quick:
        max_components = max(2, min(64, n_train - 1, n_features))
        component_grid = sorted({c for c in (2, 4, 8, 16, 32, max_components) if c <= max_components})
        for n_components in component_grid:
            models[f"pls_{n_components}"] = make_pipeline(
                StandardScaler(),
                PLSRegression(n_components=n_components),
            )
        models["extra_trees"] = ExtraTreesRegressor(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    with open(args.features, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    rows = []
    for feature_set in FEATURE_SETS:
        x, y = build_matrix(data, feature_set)
        train_idx, val_idx = split_indices(len(y), args.val_ratio, args.seed)
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        print(
            f"Feature set={feature_set} shape={x.shape} "
            f"target mean={y.mean():.4f} std={y.std():.4f} min={y.min():.4f} max={y.max():.4f}"
        )
        for model_name, model in build_models(len(train_idx), x.shape[1], quick=args.quick).items():
            try:
                model.fit(x_train, y_train)
                pred = model.predict(x_val)
                row = {"feature_set": feature_set, "model": model_name, **metrics(y_val, pred)}
                rows.append(row)
                print(
                    f"  {model_name:14s} PLCC={row['plcc']:.4f} SRCC={row['srcc']:.4f} "
                    f"final={row['final_score']:.4f} pred_std={row['pred_std']:.4f}"
                )
            except Exception as exc:
                print(f"  {model_name:14s} FAILED: {exc}", file=sys.stderr)

    result = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    print("\nBest tabular baselines:")
    print(result.head(20).to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

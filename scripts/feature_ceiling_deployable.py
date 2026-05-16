#!/usr/bin/env python3
"""Estimate the ceiling of the deployable SnapUGC student feature interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.official_artifacts import (  # noqa: E402
    StudentInputConfig,
    artifact_keys_for_input_config,
    load_official_artifact_rows,
    split_rows,
)


def metrics_from_arrays(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    plcc = pearsonr(pred, true)[0] if pred.std() > 0 and true.std() > 0 else 0.0
    srcc = spearmanr(pred, true).correlation
    ktau = kendalltau(pred, true).correlation
    metrics = {
        "plcc": 0.0 if np.isnan(plcc) else float(plcc),
        "srcc": 0.0 if np.isnan(srcc) else float(srcc),
        "ktau": 0.0 if np.isnan(ktau) else float(ktau),
        "mse": float(np.mean((pred - true) ** 2)),
        "mae": float(np.mean(np.abs(pred - true))),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "true_mean": float(true.mean()),
        "true_std": float(true.std()),
    }
    metrics["final_score"] = 0.6 * metrics["srcc"] + 0.4 * metrics["plcc"]
    return metrics


def ensure_2d(array: np.ndarray, dim: int) -> np.ndarray:
    if array.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1).astype(np.float32, copy=False)
    return array.reshape(array.shape[0], -1).astype(np.float32, copy=False)


def fit_2d(array: np.ndarray, length: int, dim: int) -> np.ndarray:
    array = ensure_2d(array, dim)
    fitted = np.zeros((length, dim), dtype=np.float32)
    if array.size == 0:
        return fitted
    n = min(length, array.shape[0])
    d = min(dim, array.shape[1])
    fitted[:n, :d] = array[:n, :d]
    return fitted


def summarize_sequence(array: np.ndarray, dim: int, max_len: int) -> np.ndarray:
    seq = ensure_2d(array, dim)[:max_len]
    if seq.size == 0:
        stats = np.zeros((dim * 8 + 2,), dtype=np.float32)
    else:
        diffs = np.diff(seq, axis=0) if len(seq) > 1 else np.zeros((1, dim), dtype=np.float32)
        stats = np.concatenate(
            [
                seq.mean(axis=0),
                seq.std(axis=0),
                seq.min(axis=0),
                seq.max(axis=0),
                seq[0],
                seq[-1],
                diffs.mean(axis=0),
                np.abs(diffs).mean(axis=0),
                np.asarray([len(seq) / max_len, float(seq.std())], dtype=np.float32),
            ]
        )
    return stats.astype(np.float32, copy=False)


def select_text_pooled(row: dict[str, object]) -> np.ndarray:
    text = ensure_2d(np.asarray(row.get("text_pooled", np.zeros((0,))), dtype=np.float32), 768)
    selected = np.zeros((3, 768), dtype=np.float32)
    n = min(3, text.shape[0])
    selected[:n, : text.shape[1]] = text[:n, :768]
    return selected


def row_to_features(
    row: dict[str, object],
    *,
    max_clips: int,
    include_flattened_frames: bool,
) -> np.ndarray:
    frame = ensure_2d(np.asarray(row["frame_fusion_feature"], dtype=np.float32), 1024)
    text = select_text_pooled(row)
    pieces = [
        summarize_sequence(frame, 1024, max_clips),
        text.reshape(-1),
        text.mean(axis=0),
        text.std(axis=0),
    ]
    if include_flattened_frames:
        pieces.insert(1, fit_2d(frame, max_clips, 1024).reshape(-1))
    if "quality_features" in row:
        q = ensure_2d(np.asarray(row["quality_features"], dtype=np.float32), 18)
        pieces.append(summarize_sequence(q, 18, max_clips))
    return np.concatenate(pieces).astype(np.float32, copy=False)


def build_matrix(
    rows: list[dict[str, object]],
    *,
    max_clips: int,
    include_flattened_frames: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features = [
        row_to_features(
            row,
            max_clips=max_clips,
            include_flattened_frames=include_flattened_frames,
        )
        for row in rows
    ]
    y = np.asarray([float(row["ecr_true"]) for row in rows], dtype=np.float32)
    teacher = np.asarray([float(row["teacher_ecr"]) for row in rows], dtype=np.float32)
    ids = [str(row["Id"]) for row in rows]
    return np.vstack(features), y, teacher, ids


def fit_and_eval(
    name: str,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    model.fit(x_train, y_train)
    pred = np.clip(model.predict(x_val), 0.0, 1.0)
    return {"name": name, "metrics": metrics_from_arrays(pred, y_val)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--no-flattened-frames", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quality-features")
    return parser.parse_args()



def attach_quality_features(rows, quality_path):
    if not quality_path:
        return
    with np.load(quality_path) as npz:
        ids = [str(v) for v in npz["ids"]]
        features = npz["quality_features"].astype(np.float32)
        mask = npz["quality_mask"].astype(bool)
    by_id = {}
    for idx, vid in enumerate(ids):
        feat = features[idx]
        m = mask[idx]
        if m.any():
            by_id[vid] = feat[m]
        else:
            by_id[vid] = np.zeros((0, features.shape[-1]), dtype=np.float32)
    missing = 0
    for row in rows:
        feat = by_id.get(str(row["Id"]))
        if feat is None:
            missing += 1
            feat = np.zeros((0, features.shape[-1]), dtype=np.float32)
        row["quality_features"] = feat
    if missing:
        print(f"Warning: missing quality features for {missing} rows", flush=True)


def main() -> None:
    args = parse_args()
    input_config = StudentInputConfig.from_preset("visual_text_sound")
    rows = load_official_artifact_rows(
        args.artifact_dir,
        args.labels_csv,
        ragged_keys=artifact_keys_for_input_config(input_config),
    )
    attach_quality_features(rows, args.quality_features)
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.split_seed)
    include_flattened_frames = not args.no_flattened_frames
    x_train, y_train, teacher_train, _ = build_matrix(
        train_rows,
        max_clips=args.max_clips,
        include_flattened_frames=include_flattened_frames,
    )
    x_val, y_val, teacher_val, _ = build_matrix(
        val_rows,
        max_clips=args.max_clips,
        include_flattened_frames=include_flattened_frames,
    )

    models = [
        (
            "ridge_alpha_100",
            make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        ),
        (
            "ridge_alpha_300",
            make_pipeline(StandardScaler(), Ridge(alpha=300.0)),
        ),
        (
            "ridge_alpha_1000",
            make_pipeline(StandardScaler(), Ridge(alpha=1000.0)),
        ),
        (
            "ridge_alpha_3000",
            make_pipeline(StandardScaler(), Ridge(alpha=3000.0)),
        ),
        (
            "sgd_elasticnet_alpha_0.0001",
            make_pipeline(
                StandardScaler(),
                SGDRegressor(
                    penalty="elasticnet",
                    alpha=0.0001,
                    l1_ratio=0.05,
                    max_iter=3000,
                    tol=1e-4,
                    random_state=args.random_state,
                ),
            ),
        ),
        (
            "svd128_hist_gradient",
            make_pipeline(
                StandardScaler(),
                TruncatedSVD(n_components=128, random_state=args.random_state),
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.03,
                    l2_regularization=0.05,
                    early_stopping=True,
                    random_state=args.random_state,
                ),
            ),
        ),
        (
            "svd128_mlp",
            make_pipeline(
                StandardScaler(),
                TruncatedSVD(n_components=128, random_state=args.random_state),
                MLPRegressor(
                    hidden_layer_sizes=(128, 32),
                    alpha=0.02,
                    learning_rate_init=0.001,
                    max_iter=250,
                    early_stopping=True,
                    random_state=args.random_state,
                ),
            ),
        ),
    ]

    results = []
    print(
        f"Feature matrix: train={x_train.shape} val={x_val.shape} "
        f"flattened_frames={include_flattened_frames}",
        flush=True,
    )
    for name, model in models:
        result = fit_and_eval(name, model, x_train, y_train, x_val, y_val)
        results.append(result)
        metrics = result["metrics"]
        print(
            f"{name}: Final={metrics['final_score']:.4f} "
            f"PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f}",
            flush=True,
        )

    teacher_metrics = metrics_from_arrays(teacher_val, y_val)
    report = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "input": "deployable visual_text_sound + quality: frame_fusion_feature + sound/title/description text_pooled + cheap quality/motion",
        "split_seed": args.split_seed,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "feature_dim": int(x_train.shape[1]),
        "include_flattened_frames": include_flattened_frames,
        "teacher_on_split": teacher_metrics,
        "results": sorted(results, key=lambda row: row["metrics"]["final_score"], reverse=True),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        "Best:",
        report["results"][0]["name"],
        json.dumps(report["results"][0]["metrics"], indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()

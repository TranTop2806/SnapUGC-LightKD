#!/usr/bin/env python3
"""Evaluate an ensemble of retained official-artifact student checkpoints."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.official_artifacts import (  # noqa: E402
    OfficialTeacherArtifactDataset,
    StudentInputConfig,
    artifact_keys_for_input_config,
    collate_student_batch,
    load_official_artifact_rows,
    split_rows,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402


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


def attach_quality_features(
    rows: list[dict[str, object]],
    quality_path: str | None,
) -> int:
    if not quality_path:
        return 0
    with np.load(quality_path) as npz:
        ids = [str(value) for value in npz["ids"]]
        features = npz["quality_features"].astype(np.float32)
    quality_by_id = {video_id: features[idx] for idx, video_id in enumerate(ids)}
    missing = 0
    for row in rows:
        feature = quality_by_id.get(str(row["Id"]))
        if feature is None:
            missing += 1
            feature = np.zeros((0, features.shape[-1]), dtype=np.float32)
        row["quality_features"] = feature
    if missing:
        print(f"Warning: missing quality features for {missing} rows", flush=True)
    return int(features.shape[-1])


def attach_dover_features(
    rows: list[dict[str, object]],
    dover_path: str | None,
    feature_mode: str = "full",
) -> int:
    if not dover_path:
        return 0
    with np.load(dover_path) as npz:
        ids = [str(value) for value in npz["ids"]]
        tech_score = npz["technical_score"].astype(np.float32).reshape(-1, 1)
        aest_score = npz["aesthetic_score"].astype(np.float32).reshape(-1, 1)
        if feature_mode == "scalars":
            features = np.concatenate([tech_score, aest_score], axis=-1)
        elif feature_mode == "full":
            tech_feat = npz["technical_feature"].astype(np.float32).reshape(-1, 768)
            aest_feat = npz["aesthetic_feature"].astype(np.float32).reshape(-1, 768)
            features = np.concatenate([tech_score, aest_score, tech_feat, aest_feat], axis=-1)
        else:
            raise ValueError(f"Unknown DOVER feature mode: {feature_mode}")
    dover_by_id = {video_id: features[idx] for idx, video_id in enumerate(ids)}
    missing = 0
    for row in rows:
        feature = dover_by_id.get(str(row["Id"]))
        if feature is None:
            missing += 1
            feature = np.zeros((0, features.shape[-1]), dtype=np.float32)
        row["dover_features"] = feature
    if missing:
        print(f"Warning: missing DOVER features for {missing} rows", flush=True)
    return int(features.shape[-1])


def load_run(run_path: Path) -> tuple[dict[str, object], Path]:
    report_path = run_path / "official_student_kd_report.json"
    if run_path.is_file() and run_path.suffix == ".json":
        report_path = run_path
        run_path = report_path.parent
    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)
    checkpoint = None
    for key in ("kd", "baseline"):
        if report.get(key) and report[key].get("checkpoint"):
            candidate = Path(report[key]["checkpoint"])
            checkpoint = candidate if candidate.exists() else run_path / candidate.name
            break
    if checkpoint is None:
        checkpoint = run_path / "student_kd_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for {run_path}: {checkpoint}")
    return report, checkpoint


def compatible_model_kwargs(model_kwargs: dict[str, object]) -> dict[str, object]:
    allowed = set(inspect.signature(OfficialArtifactStudent).parameters)
    return {key: value for key, value in model_kwargs.items() if key in allowed}


@torch.no_grad()
def predict_one(
    *,
    report: dict[str, object],
    checkpoint: Path,
    rows: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
    clip_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    input_info = report.get("input_config", {})
    if input_info:
        input_config = StudentInputConfig(
            preset=input_info.get("preset", "visual_text_sound"),
            use_frame_fusion=input_info.get("use_frame_fusion", True),
            use_sound_text=input_info.get("use_sound_text", False),
            use_title_text=input_info.get("use_title_text", False),
            use_description_text=input_info.get("use_description_text", False),
            use_text_tokens=input_info.get("use_text_tokens", False),
            use_quality_features=input_info.get("use_quality_features", False),
            quality_feature_dim=input_info.get("quality_feature_dim", 0),
            quality_fusion=input_info.get("quality_fusion", "input_concat"),
            use_dover_features=input_info.get("use_dover_features", False),
            dover_feature_dim=input_info.get("dover_feature_dim", 0),
            dover_fusion=input_info.get("dover_fusion", "input_concat"),
            use_teacher_compressed_tokens=input_info.get("use_teacher_compressed_tokens", False),
        )
    else:
        preset = str(report.get("input_preset") or "visual_text_sound")
        input_config = StudentInputConfig.from_preset(preset).with_text_tokens(
            bool(report.get("use_text_tokens", False))
        )

    # Make a copy of rows to safely attach run-specific features without cross-pollution
    rows_copy = [{k: v for k, v in r.items()} for r in rows]

    if input_config.use_quality_features:
        quality_path = report.get("quality_features")
        if quality_path:
            p = Path(quality_path)
            if not p.exists():
                p = ROOT / quality_path
            if p.exists():
                attach_quality_features(rows_copy, str(p))
            else:
                raise FileNotFoundError(f"Quality features file not found: {quality_path}")

    if input_config.use_dover_features:
        dover_path = report.get("dover_features")
        dover_mode = report.get("dover_feature_mode", "full")
        if dover_path:
            p = Path(dover_path)
            if not p.exists():
                p = ROOT / dover_path
            if p.exists():
                attach_dover_features(rows_copy, str(p), dover_mode)
            else:
                raise FileNotFoundError(f"DOVER features file not found: {dover_path}")

    dataset = OfficialTeacherArtifactDataset(
        rows_copy,
        input_config,
        max_clips=int(report.get("model_kwargs", {}).get("max_clips", 16)),
        clip_offset=clip_offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_student_batch,
    )
    model_kwargs = compatible_model_kwargs(dict(report.get("model_kwargs", {})))
    model_kwargs["clip_input_dim"] = dataset.clip_dim
    model = OfficialArtifactStudent(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    preds, true, teacher, ids = [], [], [], []
    for batch in loader:
        ids.extend(batch["ids"])
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        outputs = model(
            batch["clip_inputs"],
            batch["clip_mask"],
            batch["text_inputs"],
            batch["text_mask"],
            batch.get("dover_inputs"),
            batch.get("quality_inputs"),
        )
        preds.extend(outputs["predicted_ecr"].detach().cpu().numpy().tolist())
        true.extend(batch["ecr_true"].detach().cpu().numpy().tolist())
        teacher.extend(batch["teacher_ecr"].detach().cpu().numpy().tolist())
    return np.array(preds), np.array(true), np.array(teacher), ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions-csv")
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--clip-offsets", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load all possible ragged keys by default so they are available for all models
    rows = load_official_artifact_rows(
        args.artifact_dir,
        args.labels_csv,
    )
    eval_rows = rows
    if not args.all_rows:
        _, eval_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.split_seed)
    device = torch.device(args.device)

    individual = []
    predictions = []
    ids = None
    true = None
    teacher = None
    for run in args.runs:
        report, checkpoint = load_run(Path(run))
        offset_predictions = []
        run_true = run_teacher = None
        run_ids = None
        for clip_offset in args.clip_offsets:
            pred, offset_true, offset_teacher, offset_ids = predict_one(
                report=report,
                checkpoint=checkpoint,
                rows=eval_rows,
                device=device,
                batch_size=args.batch,
                clip_offset=clip_offset,
            )
            if run_ids is None:
                run_ids = offset_ids
                run_true = offset_true
                run_teacher = offset_teacher
            elif run_ids != offset_ids:
                raise RuntimeError(f"Prediction order mismatch for {run} offset {clip_offset}")
            offset_predictions.append(pred)
        pred = np.vstack(offset_predictions).mean(axis=0)
        if ids is None:
            ids = run_ids
            true = run_true
            teacher = run_teacher
        elif ids != run_ids:
            raise RuntimeError(f"Prediction order mismatch for {run}")
        predictions.append(pred)
        individual.append(
            {
                "run": str(run),
                "checkpoint": str(checkpoint),
                "metrics": metrics_from_arrays(pred, run_true),
            }
        )

    stacked = np.vstack(predictions)
    ensemble_pred = stacked.mean(axis=0)
    report = {
        "runs": args.runs,
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "split_seed": args.split_seed,
        "all_rows": args.all_rows,
        "n_eval": len(eval_rows),
        "clip_offsets": args.clip_offsets,
        "individual": individual,
        "ensemble": metrics_from_arrays(ensemble_pred, true),
        "teacher": metrics_from_arrays(teacher, true),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.predictions_csv:
        pred_df = pd.DataFrame({"Id": ids, "ecr_true": true, "teacher_ecr": teacher})
        for idx, pred in enumerate(predictions):
            pred_df[f"pred_{idx}"] = pred
        pred_df["ensemble_pred"] = ensemble_pred
        pred_df.to_csv(args.predictions_csv, index=False)

    print(json.dumps({"ensemble": report["ensemble"], "teacher": report["teacher"]}, indent=2))


if __name__ == "__main__":
    main()

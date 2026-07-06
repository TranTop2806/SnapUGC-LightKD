#!/usr/bin/env python3
"""Train the retained SnapUGC deployable student with or without KD."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.official_artifacts import (  # noqa: E402
    OfficialTeacherArtifactDataset,
    StudentInputConfig,
    artifact_keys_for_input_config,
    collate_student_batch,
    load_official_artifact_rows,
    read_id_file,
    select_rows_by_ids,
    split_rows,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent, compute_losses  # noqa: E402


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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


def scheduled_kd_weights(
    weights: dict[str, float],
    *,
    epoch: int,
    epochs: int,
    curriculum: str,
) -> dict[str, float]:
    if curriculum == "none":
        return weights
    scheduled = dict(weights)
    if curriculum == "three_phase":
        phase1 = max(1, int(round(epochs * 0.25)))
        phase2 = max(phase1 + 1, int(round(epochs * 0.625)))
        if epoch <= phase1:
            for key in (
                "hard_ecr",
                "hard_rank",
                "clip_ecr",
                "temporal_hidden",
                "fusion_hidden",
                "attention",
                "teacher_pearson",
                "teacher_spearman",
                "teacher_listwise",
                "hard_ldl",
                "spkd",
            ):
                scheduled[key] = 0.0
            scheduled["soft_ecr"] = weights.get("soft_ecr", 0.0)
            scheduled["teacher_rank"] = weights.get("teacher_rank", 0.0)
            scheduled["teacher_ldl"] = weights.get("teacher_ldl", 0.0)
        elif epoch <= phase2:
            t = (epoch - phase1) / max(1, phase2 - phase1)
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            scheduled["hard_ecr"] = ease * weights.get("hard_ecr", 1.0)
            scheduled["hard_rank"] = ease * weights.get("hard_rank", 0.0)
            scheduled["soft_ecr"] = (1.0 - 0.35 * ease) * weights.get("soft_ecr", 0.0)
            for key in (
                "clip_ecr",
                "temporal_hidden",
                "fusion_hidden",
                "attention",
                "teacher_pearson",
                "teacher_spearman",
                "teacher_listwise",
                "hard_ldl",
                "spkd",
            ):
                scheduled[key] = ease * weights.get(key, 0.0)
        else:
            t = (epoch - phase2) / max(1, epochs - phase2)
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            scheduled["hard_ecr"] = weights.get("hard_ecr", 1.0)
            scheduled["hard_rank"] = weights.get("hard_rank", 0.0)
            scheduled["soft_ecr"] = (0.65 - 0.5 * ease) * weights.get("soft_ecr", 0.0)
            for key in (
                "clip_ecr",
                "temporal_hidden",
                "fusion_hidden",
                "attention",
                "teacher_pearson",
                "teacher_spearman",
                "teacher_listwise",
                "teacher_ldl",
                "spkd",
            ):
                scheduled[key] = (1.0 - ease) * weights.get(key, 0.0)
            scheduled["hard_ldl"] = weights.get("hard_ldl", 0.0)
        return scheduled
    elif curriculum == "feature_first":
        phase1 = max(1, int(round(epochs * 0.25)))
        phase2 = max(phase1 + 1, int(round(epochs * 0.625)))
        if epoch <= phase1:
            for key in (
                "hard_ecr",
                "soft_ecr",
                "clip_ecr",
                "hard_rank",
                "teacher_rank",
                "teacher_pearson",
                "teacher_spearman",
                "teacher_listwise",
                "hard_ldl",
                "teacher_ldl",
                "pseudo_ecr",
            ):
                scheduled[key] = 0.0
            scheduled["temporal_hidden"] = weights.get("temporal_hidden", 0.0)
            scheduled["fusion_hidden"] = weights.get("fusion_hidden", 0.0)
            scheduled["attention"] = weights.get("attention", 0.0)
            scheduled["spkd"] = weights.get("spkd", 0.0)
        elif epoch <= phase2:
            t = (epoch - phase1) / max(1, phase2 - phase1)
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            scheduled["hard_ecr"] = ease * weights.get("hard_ecr", 1.0)
            scheduled["soft_ecr"] = ease * weights.get("soft_ecr", 0.0)
            scheduled["teacher_rank"] = ease * weights.get("teacher_rank", 0.0)
            scheduled["hard_rank"] = ease * weights.get("hard_rank", 0.0)
            scheduled["clip_ecr"] = ease * weights.get("clip_ecr", 0.0)
            scheduled["hard_ldl"] = ease * weights.get("hard_ldl", 0.0)
            scheduled["teacher_ldl"] = ease * weights.get("teacher_ldl", 0.0)
            
            scheduled["temporal_hidden"] = (1.0 - ease) * weights.get("temporal_hidden", 0.0)
            scheduled["fusion_hidden"] = (1.0 - ease) * weights.get("fusion_hidden", 0.0)
            scheduled["spkd"] = (1.0 - ease) * weights.get("spkd", 0.0)
        else:
            t = (epoch - phase2) / max(1, epochs - phase2)
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            scheduled["hard_ecr"] = weights.get("hard_ecr", 1.0)
            scheduled["soft_ecr"] = weights.get("soft_ecr", 0.0)
            scheduled["hard_rank"] = weights.get("hard_rank", 0.0)
            scheduled["teacher_rank"] = weights.get("teacher_rank", 0.0)
            scheduled["hard_ldl"] = weights.get("hard_ldl", 0.0)
            
            for key in ("temporal_hidden", "fusion_hidden", "attention", "spkd", "clip_ecr", "teacher_ldl"):
                scheduled[key] = 0.0
        return scheduled
    raise ValueError(f"Unknown KD curriculum: {curriculum}")



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


def attach_lite_action(rows: list[dict[str, object]], path: str | None) -> int:
    if not path:
        return 0
    with np.load(path) as npz:
        ids = [str(v) for v in npz["ids"]]
        features = npz["lite_action_features"].astype(np.float32)
    feat_by_id = {vid: features[idx] for idx, vid in enumerate(ids)}
    missing = 0
    for row in rows:
        feature = feat_by_id.get(str(row["Id"]))
        if feature is None:
            missing += 1
            feature = np.zeros((0, features.shape[-1]), dtype=np.float32)
        row["lite_action_features"] = feature
    if missing:
        print(f"Warning: missing lite action features for {missing} rows", flush=True)
    return int(features.shape[-1])


def attach_pseudo_labels(
    rows: list[dict[str, object]],
    pseudo_path: str | None,
    column: str = "ensemble_pred",
) -> int:
    if not pseudo_path:
        return 0
    path = Path(pseudo_path)
    if path.suffix == ".npz":
        with np.load(path) as npz:
            ids = [str(value) for value in npz["ids"]]
            values = npz[column].astype(np.float32)
    else:
        df = pd.read_csv(path)
        if "Id" not in df.columns or column not in df.columns:
            raise ValueError(f"{pseudo_path} must contain Id and {column} columns")
        ids = df["Id"].astype(str).tolist()
        values = df[column].astype(np.float32).to_numpy()
    pseudo_by_id = {video_id: float(values[idx]) for idx, video_id in enumerate(ids)}
    missing = 0
    for row in rows:
        pseudo = pseudo_by_id.get(str(row["Id"]))
        if pseudo is None:
            missing += 1
            pseudo = float(row["teacher_ecr"])
        row["pseudo_ecr"] = pseudo
    if missing:
        print(f"Warning: missing pseudo labels for {missing} rows", flush=True)
    return len(rows) - missing


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, true, teacher = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
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
    metrics = metrics_from_arrays(np.array(preds), np.array(true))
    teacher_metrics = metrics_from_arrays(np.array(teacher), np.array(true))
    metrics["teacher_final_score_on_split"] = teacher_metrics["final_score"]
    metrics["teacher_plcc_on_split"] = teacher_metrics["plcc"]
    metrics["teacher_srcc_on_split"] = teacher_metrics["srcc"]
    return metrics


def train_one(
    *,
    name: str,
    model: OfficialArtifactStudent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_kd: bool,
    weights: dict[str, float],
    repr_loss: str,
    rank_temperature: float,
    soft_rank_temperature: float,
    contrastive_temperature: float,
    ldl_sigma: float,
    kd_curriculum: str,
    focal_teacher_alpha: float,
    teacher_score_loss: str,
    teacher_huber_beta: float,
    hard_pair_similarity: float,
    hard_pair_target_margin: float,
    kd_transfer_beta: float,
    prototype_sigma: float,
    prototype_temperature: float,
    save_path: Path,
) -> dict[str, object]:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.02)
    best_score = -1e9
    best_epoch = 0
    history = []
    started = time.time()
    initial_metrics = evaluate(model, val_loader, device)
    best_score = initial_metrics["final_score"]
    best_epoch = 0
    torch.save(model.state_dict(), save_path)
    history.append({"epoch": 0, **initial_metrics})
    print(
        f"{name} epoch 000/{epochs} "
        f"PLCC={initial_metrics['plcc']:.4f} SRCC={initial_metrics['srcc']:.4f} "
        f"score={initial_metrics['final_score']:.4f} *",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sums: dict[str, float] = {}
        n_batches = 0
        epoch_weights = scheduled_kd_weights(
            weights,
            epoch=epoch,
            epochs=epochs,
            curriculum=kd_curriculum if use_kd else "none",
        )
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["clip_inputs"],
                batch["clip_mask"],
                batch["text_inputs"],
                batch["text_mask"],
                batch.get("dover_inputs"),
                batch.get("quality_inputs"),
            )
            loss, loss_parts = compute_losses(
                outputs,
                batch,
                use_kd=use_kd,
                weights=epoch_weights,
                repr_loss=repr_loss,
                rank_temperature=rank_temperature,
                soft_rank_temperature=soft_rank_temperature,
                contrastive_temperature=contrastive_temperature,
                ldl_sigma=ldl_sigma,
                focal_teacher_alpha=focal_teacher_alpha,
                teacher_score_loss=teacher_score_loss,
                teacher_huber_beta=teacher_huber_beta,
                hard_pair_similarity=hard_pair_similarity,
                hard_pair_target_margin=hard_pair_target_margin,
                kd_transfer_beta=kd_transfer_beta,
                prototype_sigma=prototype_sigma,
                prototype_temperature=prototype_temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in loss_parts.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + value
            n_batches += 1

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / max(1, n_batches) for k, v in loss_sums.items()},
            **val_metrics,
        }
        history.append(row)
        if val_metrics["final_score"] > best_score:
            best_score = val_metrics["final_score"]
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
        print(
            f"{name} epoch {epoch:03d}/{epochs} "
            f"loss={row.get('train_total', 0.0):.5f} "
            f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
            f"score={val_metrics['final_score']:.4f}"
            f"{' *' if best_epoch == epoch else ''}",
            flush=True,
        )

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return {
        "best_epoch": best_epoch,
        "best": evaluate(model, val_loader, device),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(save_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--input-preset",
        choices=("visual_text_sound", "clip_mobilenet_text"),
        default="visual_text_sound",
    )
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "cross_attention"),
        default="concat",
    )
    parser.add_argument("--projection-head", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--ecr-bins", type=int, default=0)
    parser.add_argument(
        "--temporal-aggregation",
        choices=("attention", "global_local"),
        default="attention",
    )
    parser.add_argument("--local-clips", type=int, default=5)
    parser.add_argument("--semantic-gated-fusion", action="store_true")
    parser.add_argument("--shared-distill-dim", type=int, default=0)
    parser.add_argument("--fusion-experts", type=int, default=1)
    parser.add_argument("--quality-features")
    parser.add_argument(
        "--quality-fusion",
        choices=("input_concat", "clip_add"),
        default="input_concat",
    )
    parser.add_argument("--dover-features")
    parser.add_argument("--lite-action-features")
    parser.add_argument("--dover-feature-mode", choices=("full", "scalars"), default="full")
    parser.add_argument(
        "--dover-fusion",
        choices=("input_concat", "late_add", "late_concat"),
        default="input_concat",
    )
    parser.add_argument("--temporal-conv", choices=("none", "depthwise", "full"), default="none")
    parser.add_argument("--shared-transformer-weights", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--train-ids", help="Explicit training IDs, one per line.")
    parser.add_argument("--val-ids", help="Explicit model-selection IDs, one per line.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-kind", choices=("baseline", "kd", "both"), default="kd")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--pseudo-labels")
    parser.add_argument("--pseudo-label-column", default="ensemble_pred")
    parser.add_argument(
        "--repr-loss",
        choices=("raw_mse", "normalized_mse", "cosine"),
        default="cosine",
    )
    parser.add_argument("--soft-weight", type=float, default=1.1)
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=1.0,
        help="Weight for MSE(student ECR, ground-truth ECR) in KD runs.",
    )
    parser.add_argument("--pseudo-weight", type=float, default=0.0)
    parser.add_argument("--hard-ldl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-ldl-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-ldl-weight", type=float, default=0.0)
    parser.add_argument("--ldl-sigma", type=float, default=0.06)
    parser.add_argument(
        "--kd-curriculum",
        choices=("none", "three_phase", "feature_first"),
        default="none",
    )
    parser.add_argument("--clip-weight", type=float, default=0.08)
    parser.add_argument("--temporal-weight", type=float, default=0.02)
    parser.add_argument("--fusion-weight", type=float, default=0.02)
    parser.add_argument("--attention-weight", type=float, default=0.005)
    parser.add_argument("--hard-rank-weight", type=float, default=0.04)
    parser.add_argument("--hard-pearson-weight", type=float, default=0.0)
    parser.add_argument("--hard-ccc-weight", type=float, default=0.0)
    parser.add_argument("--hard-spearman-weight", type=float, default=0.0)
    parser.add_argument("--hard-listwise-weight", type=float, default=0.0)
    parser.add_argument("--hard-std-weight", type=float, default=0.0)
    parser.add_argument("--teacher-rank-weight", type=float, default=0.18)
    parser.add_argument("--teacher-pearson-weight", type=float, default=0.02)
    parser.add_argument("--teacher-ccc-weight", type=float, default=0.0)
    parser.add_argument("--teacher-spearman-weight", type=float, default=0.015)
    parser.add_argument("--teacher-listwise-weight", type=float, default=0.02)
    parser.add_argument("--teacher-score-relation-weight", type=float, default=0.0)
    parser.add_argument("--hard-score-relation-weight", type=float, default=0.0)
    parser.add_argument("--student-teacher-relation-weight", type=float, default=0.0)
    parser.add_argument("--teacher-prototype-weight", type=float, default=0.0)
    parser.add_argument("--hard-prototype-weight", type=float, default=0.0)
    parser.add_argument("--rkd-distance-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-hidden-weight", type=float, default=0.0)
    parser.add_argument("--spkd-weight", type=float, default=0.0)
    parser.add_argument("--rank-temperature", type=float, default=0.15)
    parser.add_argument("--soft-rank-temperature", type=float, default=0.08)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--focal-teacher-alpha", type=float, default=0.0)
    parser.add_argument(
        "--teacher-score-loss",
        choices=("mse", "huber"),
        default="mse",
        help="Pointwise loss used to imitate the teacher ECR score.",
    )
    parser.add_argument("--teacher-huber-beta", type=float, default=0.05)
    parser.add_argument("--kd-transfer-beta", type=float, default=0.0)
    parser.add_argument("--prototype-sigma", type=float, default=0.10)
    parser.add_argument("--prototype-temperature", type=float, default=0.08)
    parser.add_argument("--hard-pair-similarity", type=float, default=0.0)
    parser.add_argument("--hard-pair-target-margin", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    input_config = StudentInputConfig.from_preset(args.input_preset)
    ragged_keys = set(artifact_keys_for_input_config(input_config))
    rows = load_official_artifact_rows(
        args.artifact_dir,
        args.labels_csv,
        ragged_keys=tuple(ragged_keys),
    )
    quality_dim = attach_quality_features(rows, args.quality_features)
    input_config = input_config.with_quality_features(
        bool(args.quality_features),
        quality_dim,
        args.quality_fusion,
    )
    dover_dim = attach_dover_features(rows, args.dover_features, args.dover_feature_mode)
    input_config = input_config.with_dover_features(
        bool(args.dover_features),
        dover_dim,
        args.dover_fusion,
    )
    lite_action_dim = attach_lite_action(rows, args.lite_action_features)
    input_config = input_config.with_lite_action(
        bool(args.lite_action_features),
        lite_action_dim,
    )
    n_pseudo_labels = attach_pseudo_labels(rows, args.pseudo_labels, args.pseudo_label_column)
    if bool(args.train_ids) != bool(args.val_ids):
        raise ValueError("--train-ids and --val-ids must be provided together")
    if args.train_ids:
        train_ids = read_id_file(args.train_ids)
        val_ids = read_id_file(args.val_ids)
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(f"Explicit train/val splits overlap on {len(overlap)} IDs")
        train_rows = select_rows_by_ids(rows, train_ids, split_name="train split")
        val_rows = select_rows_by_ids(rows, val_ids, split_name="validation split")
    else:
        train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=split_seed)
    train_dataset = OfficialTeacherArtifactDataset(
        train_rows,
        input_config,
        max_clips=args.max_clips,
    )
    val_dataset = OfficialTeacherArtifactDataset(
        val_rows,
        input_config,
        max_clips=args.max_clips,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_student_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch,
        shuffle=False,
        collate_fn=collate_student_batch,
    )

    model_kwargs = {
        "clip_input_dim": train_dataset.clip_dim,
        "hidden_dim": args.hidden_dim,
        "max_clips": args.max_clips,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "dropout": args.dropout,
        "fusion_mode": args.fusion_mode,
        "projection_head": args.projection_head,
        "quality_input_dim": quality_dim if args.quality_features and args.quality_fusion != "input_concat" else 0,
        "quality_fusion": "clip_add" if args.quality_features and args.quality_fusion == "clip_add" else "none",
        "ecr_bins": args.ecr_bins,
        "temporal_aggregation": args.temporal_aggregation,
        "local_clips": args.local_clips,
        "semantic_gated_fusion": args.semantic_gated_fusion,
        "shared_distill_dim": args.shared_distill_dim,
        "fusion_experts": args.fusion_experts,
        "dover_input_dim": dover_dim if args.dover_features and args.dover_fusion != "input_concat" else 0,
        "dover_fusion": args.dover_fusion if args.dover_features and args.dover_fusion != "input_concat" else "none",
        "temporal_conv": args.temporal_conv,
        "shared_transformer_weights": args.shared_transformer_weights,
        "drop_path": args.drop_path,
    }
    print(
        f"Loaded official artifacts: total={len(rows)} train={len(train_rows)} "
        f"val={len(val_rows)} clip_dim={train_dataset.clip_dim} "
        f"input_preset={args.input_preset}",
        flush=True,
    )

    baseline_result = None
    if args.run_kind in ("baseline", "both"):
        baseline = OfficialArtifactStudent(**model_kwargs).to(device)
        if args.init_checkpoint:
            baseline.load_state_dict(
                torch.load(args.init_checkpoint, map_location=device, weights_only=True),
                strict=False,
            )
        baseline_result = train_one(
            name="baseline",
            model=baseline,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_kd=False,
            weights={
                "hard_ecr": 1.0,
                "hard_rank": args.hard_rank_weight,
                "hard_pearson": args.hard_pearson_weight,
                "hard_spearman": args.hard_spearman_weight,
                "hard_listwise": args.hard_listwise_weight,
                "hard_std": args.hard_std_weight,
            },
            repr_loss=args.repr_loss,
            rank_temperature=args.rank_temperature,
            soft_rank_temperature=args.soft_rank_temperature,
            contrastive_temperature=args.contrastive_temperature,
            ldl_sigma=args.ldl_sigma,
            kd_curriculum="none",
            focal_teacher_alpha=0.0,
            teacher_score_loss="mse",
            teacher_huber_beta=args.teacher_huber_beta,
            hard_pair_similarity=0.0,
            hard_pair_target_margin=args.hard_pair_target_margin,
            kd_transfer_beta=0.0,
            prototype_sigma=args.prototype_sigma,
            prototype_temperature=args.prototype_temperature,
            save_path=save_dir / "student_baseline_best.pth",
        )

    kd_weights = {
        "hard_ecr": args.hard_weight,
        "hard_rank": args.hard_rank_weight,
        "hard_pearson": args.hard_pearson_weight,
        "hard_ccc": args.hard_ccc_weight,
        "hard_spearman": args.hard_spearman_weight,
        "hard_listwise": args.hard_listwise_weight,
        "hard_std": args.hard_std_weight,
        "hard_ldl": args.hard_ldl_weight,
        "soft_ecr": args.soft_weight,
        "pseudo_ecr": args.pseudo_weight,
        "teacher_ldl": args.teacher_ldl_weight,
        "pseudo_ldl": args.pseudo_ldl_weight,
        "clip_ecr": args.clip_weight,
        "temporal_hidden": args.temporal_weight,
        "fusion_hidden": args.fusion_weight,
        "attention": args.attention_weight,
        "teacher_rank": args.teacher_rank_weight,
        "teacher_pearson": args.teacher_pearson_weight,
        "teacher_ccc": args.teacher_ccc_weight,
        "teacher_spearman": args.teacher_spearman_weight,
        "teacher_listwise": args.teacher_listwise_weight,
        "teacher_score_relation": args.teacher_score_relation_weight,
        "hard_score_relation": args.hard_score_relation_weight,
        "student_teacher_relation": args.student_teacher_relation_weight,
        "teacher_prototype": args.teacher_prototype_weight,
        "hard_prototype": args.hard_prototype_weight,
        "rkd_distance": args.rkd_distance_weight,
        "contrastive_hidden": args.contrastive_hidden_weight,
        "spkd": args.spkd_weight,
    }
    kd_result = None
    if args.run_kind in ("kd", "both"):
        kd_model = OfficialArtifactStudent(**model_kwargs).to(device)
        if args.init_checkpoint:
            kd_model.load_state_dict(
                torch.load(args.init_checkpoint, map_location=device, weights_only=True),
                strict=False,
            )
        kd_result = train_one(
            name="kd",
            model=kd_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_kd=True,
            weights=kd_weights,
            repr_loss=args.repr_loss,
            rank_temperature=args.rank_temperature,
            soft_rank_temperature=args.soft_rank_temperature,
            contrastive_temperature=args.contrastive_temperature,
            ldl_sigma=args.ldl_sigma,
            kd_curriculum=args.kd_curriculum,
            focal_teacher_alpha=args.focal_teacher_alpha,
            teacher_score_loss=args.teacher_score_loss,
            teacher_huber_beta=args.teacher_huber_beta,
            hard_pair_similarity=args.hard_pair_similarity,
            hard_pair_target_margin=args.hard_pair_target_margin,
            kd_transfer_beta=args.kd_transfer_beta,
            prototype_sigma=args.prototype_sigma,
            prototype_temperature=args.prototype_temperature,
            save_path=save_dir / "student_kd_best.pth",
        )

    report = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "input_preset": args.input_preset,
        "input_config": input_config.__dict__,
        "quality_features": args.quality_features,
        "quality_fusion": args.quality_fusion,
        "dover_features": args.dover_features,
        "dover_feature_mode": args.dover_feature_mode,
        "dover_fusion": args.dover_fusion,
        "lite_action_features": args.lite_action_features,
        "pseudo_labels": args.pseudo_labels,
        "pseudo_label_column": args.pseudo_label_column,
        "n_pseudo_labels": n_pseudo_labels,
        "ragged_keys": sorted(ragged_keys),
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "seed": args.seed,
        "split_seed": split_seed,
        "train_ids": str(Path(args.train_ids).resolve()) if args.train_ids else None,
        "val_ids": str(Path(args.val_ids).resolve()) if args.val_ids else None,
        "model_kwargs": model_kwargs,
        "repr_loss": args.repr_loss,
        "rank_temperature": args.rank_temperature,
        "soft_rank_temperature": args.soft_rank_temperature,
        "ldl_sigma": args.ldl_sigma,
        "kd_curriculum": args.kd_curriculum,
        "focal_teacher_alpha": args.focal_teacher_alpha,
        "teacher_score_loss": args.teacher_score_loss,
        "teacher_huber_beta": args.teacher_huber_beta,
        "kd_transfer_beta": args.kd_transfer_beta,
        "prototype_sigma": args.prototype_sigma,
        "prototype_temperature": args.prototype_temperature,
        "hard_pair_similarity": args.hard_pair_similarity,
        "hard_pair_target_margin": args.hard_pair_target_margin,
        "run_kind": args.run_kind,
        "init_checkpoint": args.init_checkpoint,
        "kd_weights": kd_weights,
        "baseline": baseline_result,
        "kd": kd_result,
        "kd_gain_final_score": (
            kd_result["best"]["final_score"] - baseline_result["best"]["final_score"]
            if kd_result is not None and baseline_result is not None
            else None
        ),
    }
    with (save_dir / "official_student_kd_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary = {
        "save_dir": str(save_dir),
        "input_preset": args.input_preset,
        "baseline_final": baseline_result["best"]["final_score"] if baseline_result else None,
        "kd_final": kd_result["best"]["final_score"] if kd_result else None,
        "kd_gain_final_score": report["kd_gain_final_score"],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

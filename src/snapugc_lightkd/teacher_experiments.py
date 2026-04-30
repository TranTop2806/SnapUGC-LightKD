"""Teacher-only experiment runner for architecture/version search."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .models import count_params
from .training import (
    KDDataset,
    add_optional_model_inputs,
    evaluate_model,
    is_better_metric,
    load_data,
    split_data,
)


MODEL_REGISTRY = {
    "v2_original": ("snapugc_lightkd.models", "TeacherModel"),
    "v3_regularized_7tok": (
        "snapugc_lightkd.teacher_v3_regularized",
        "TeacherV3Regularized",
    ),
    "v4_shared_text_5tok": (
        "snapugc_lightkd.teacher_v4_shared_text",
        "TeacherV4SharedText",
    ),
    "v5_gated_late_fusion": (
        "snapugc_lightkd.teacher_v5_gated_late_fusion",
        "TeacherV5GatedLateFusion",
    ),
    "v6_pooled_rank": (
        "snapugc_lightkd.teacher_v6_pooled_rank",
        "TeacherV6PooledRank",
    ),
    "v7_corr_rank": (
        "snapugc_lightkd.teacher_v7_corr_rank",
        "TeacherV7CorrRank",
    ),
    "v8_rich_inputs": (
        "snapugc_lightkd.teacher_v8_rich_inputs",
        "TeacherV8RichInputs",
    ),
    "v9_rich_corr_rank": (
        "snapugc_lightkd.teacher_v9_rich_corr_rank",
        "TeacherV9RichCorrRank",
    ),
    "v10_compact_rich": (
        "snapugc_lightkd.teacher_v10_compact_rich",
        "TeacherV10CompactRich",
    ),
    "v12_rich_linear_head": (
        "snapugc_lightkd.teacher_v12_rich_linear_head",
        "TeacherV12RichLinearHead",
    ),
    "v13_rich_sampledrop": (
        "snapugc_lightkd.teacher_v13_rich_sampledrop",
        "TeacherV13RichSampleDrop",
    ),
}


@dataclass
class EpochRow:
    epoch: int
    lr: float
    train_loss: float
    mse: float
    mae: float
    plcc: float
    srcc: float
    ktau: float
    final_score: float
    pred_mean: float
    pred_std: float
    is_best: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_teacher_class(model_version: str):
    if model_version not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown --model-version {model_version!r}. Known: {known}")
    module_name, class_name = MODEL_REGISTRY[model_version]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def train_teacher_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        kwargs = {
            "clip_frames": batch["clip_frames"].to(device),
            "clip_mask": batch["clip_mask"].to(device),
            "motion_clips": batch["motion_clips"].to(device),
            "motion_mask": batch["motion_mask"].to(device),
            "audio_emb": batch["audio_emb"].to(device),
            "text_emb": batch["text_emb"].to(device),
            "caption_emb": batch["caption_emb"].to(device),
            "rationale_emb": batch["rationale_emb"].to(device),
            "quality_scores": batch["quality_scores"].to(device),
            "ecr_targets": batch["ecr"].to(device),
            "aesthetic_targets": batch["aesthetic"].to(device),
            "technical_targets": batch["technical"].to(device),
        }
        kwargs = add_optional_model_inputs(model, kwargs, batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(**kwargs)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(out["loss"].item())
        n += 1
    return total_loss / max(1, n)


def make_scheduler(optimizer, *, epochs: int, warmup_epochs: int, eta_min: float):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    warmup = LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=eta_min,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def write_epoch_csv(path: Path, rows: list[EpochRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_teacher_experiment(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data, max_samples=args.max_samples)
    train_data, val_data = split_data(data, val_ratio=args.val_ratio, seed=args.split_seed)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Split seed: {args.split_seed}")

    train_loader = DataLoader(
        KDDataset(
            train_data,
            max_frames=args.max_frames,
            max_motion_clips=args.max_motion_clips,
            opening_seconds=args.opening_seconds,
        ),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        KDDataset(
            val_data,
            max_frames=args.max_frames,
            max_motion_clips=args.max_motion_clips,
            opening_seconds=args.opening_seconds,
        ),
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    teacher_cls = load_teacher_class(args.model_version)
    teacher_kwargs = {
        "hidden_dim": args.teacher_hidden,
        "n_blocks": args.teacher_blocks,
        "n_heads": args.teacher_heads,
        "dropout": args.dropout,
        "max_frames": args.max_frames,
        "max_motion_clips": args.max_motion_clips,
        "aux_weight": args.aux_weight,
    }
    teacher = teacher_cls(**teacher_kwargs).to(device)
    total_params, trainable_params = count_params(teacher)
    print(f"Model: {args.model_version} | Params: {total_params:,}")
    print(f"Save dir: {save_dir}")

    optimizer = AdamW(
        teacher.parameters(),
        lr=args.teacher_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=args.teacher_epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
    )

    best_metric = float("inf") if args.selection_metric in {"mse", "mae"} else -float("inf")
    best_epoch = 0
    best_metrics = None
    rows: list[EpochRow] = []
    checkpoint_path = save_dir / "best_teacher.pth"
    started = time.time()

    config = {
        "model_version": args.model_version,
        "model_class": f"{teacher_cls.__module__}.{teacher_cls.__name__}",
        "data": args.data,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "params": total_params,
        "trainable_params": trainable_params,
        "teacher_kwargs": teacher_kwargs,
        "training": {
            "teacher_epochs": args.teacher_epochs,
            "teacher_lr": args.teacher_lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "eta_min": args.eta_min,
            "batch": args.batch,
            "eval_batch": args.eval_batch,
            "selection_metric": args.selection_metric,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "opening_seconds": args.opening_seconds,
        },
    }
    with (save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.teacher_epochs + 1):
        train_loss = train_teacher_epoch(teacher, train_loader, optimizer, device)
        scheduler.step()
        val_metrics = evaluate_model(teacher, val_loader, device, "teacher")
        better, metric_value = is_better_metric(
            val_metrics,
            best_metric,
            args.selection_metric,
        )
        if better:
            best_metric = metric_value
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            torch.save(teacher.state_dict(), checkpoint_path)

        lr = optimizer.param_groups[0]["lr"]
        row = EpochRow(
            epoch=epoch,
            lr=float(lr),
            train_loss=float(train_loss),
            mse=float(val_metrics["mse"]),
            mae=float(val_metrics["mae"]),
            plcc=float(val_metrics["plcc"]),
            srcc=float(val_metrics["srcc"]),
            ktau=float(val_metrics["ktau"]),
            final_score=float(val_metrics["final_score"]),
            pred_mean=float(val_metrics["pred_mean"]),
            pred_std=float(val_metrics["pred_std"]),
            is_best=bool(better),
        )
        rows.append(row)
        write_epoch_csv(save_dir / "metrics_by_epoch.csv", rows)

        print(
            f"Epoch {epoch:03d}/{args.teacher_epochs} "
            f"loss={train_loss:.5f} lr={lr:.2e} "
            f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
            f"Score={val_metrics['final_score']:.4f} MSE={val_metrics['mse']:.5f}"
            f"{' *' if better else ''}"
        )

    elapsed = time.time() - started
    if best_metrics is None:
        best_metrics = evaluate_model(teacher, val_loader, device, "teacher")

    report = {
        **config,
        "best_epoch": best_epoch,
        "best": {
            **best_metrics,
            "params": total_params,
            "checkpoint": str(checkpoint_path),
        },
        "elapsed_seconds": elapsed,
    }
    with (save_dir / "teacher_experiment_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Best epoch: {best_epoch} | Best score: {best_metrics['final_score']:.4f}")
    print(f"Report saved: {save_dir / 'teacher_experiment_report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train one teacher architecture version.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--model-version", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--teacher-hidden", type=int, default=384)
    parser.add_argument("--teacher-blocks", type=int, default=1)
    parser.add_argument("--teacher-heads", type=int, default=6)
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument(
        "--selection-metric",
        choices=["final_score", "srcc", "plcc", "mse", "mae"],
        default="final_score",
    )
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-motion-clips", type=int, default=4)
    parser.add_argument(
        "--opening-seconds",
        type=float,
        default=None,
        help="Approximate first-N-second visual/motion window from uniformly sampled tokens.",
    )
    args = parser.parse_args()
    run_teacher_experiment(args)


if __name__ == "__main__":
    main()

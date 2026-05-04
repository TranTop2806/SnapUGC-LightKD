#!/usr/bin/env python3
"""Train Student baseline/KD from a teacher prediction CSV.

Use this after an exact LMM-EVQA teacher run. The teacher CSV should contain
`Id` and `ECR`. If predictions are on the 0-100 scale used by LMM-EVQA, pass
`--teacher-scale 100`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.models import StudentModel, count_params
from snapugc_lightkd.training import (
    KDDataset,
    evaluate_model,
    is_better_metric,
    load_data,
    split_data,
    train_student_epoch,
)


def sample_id(item: dict) -> str:
    for key in ("id", "Id", "video_id"):
        if key in item:
            return str(item[key])
    video_path = item.get("video_path") or item.get("path")
    if video_path:
        return Path(video_path).stem
    raise KeyError("Feature item has no id/Id/video_id/video_path/path")


def attach_teacher_predictions(data: list[dict], pred_csv: Path, scale: float) -> list[dict]:
    df = pd.read_csv(pred_csv)
    if not {"Id", "ECR"}.issubset(df.columns):
        raise ValueError(f"{pred_csv} must contain Id and ECR columns")
    pred = {str(row.Id): float(row.ECR) / scale for row in df.itertuples(index=False)}
    enriched = []
    missing = []
    for item in data:
        copied = copy.deepcopy(item)
        sid = sample_id(copied)
        if sid not in pred:
            missing.append(sid)
            continue
        copied["teacher_ecr"] = pred[sid]
        enriched.append(copied)
    if missing:
        print(f"Skipped {len(missing)} samples without teacher predictions; preview={missing[:10]}")
    if not enriched:
        raise RuntimeError("No feature rows matched teacher predictions")
    values = np.array([row["teacher_ecr"] for row in enriched], dtype=np.float32)
    print(
        f"Attached teacher predictions: n={len(enriched)} "
        f"mean={values.mean():.4f} std={values.std():.4f}"
    )
    return enriched


def split_by_csv(data: list[dict], split_csv: Path) -> tuple[list[dict], list[dict]]:
    df = pd.read_csv(split_csv)
    if not {"Id", "split"}.issubset(df.columns):
        raise ValueError(f"{split_csv} must contain Id and split columns")
    split_map = {str(row.Id): str(row.split).lower() for row in df.itertuples(index=False)}
    train, val, skipped = [], [], []
    for item in data:
        sid = sample_id(item)
        split = split_map.get(sid)
        if split == "train":
            train.append(item)
        elif split in {"val", "valid", "validation", "test"}:
            val.append(item)
        else:
            skipped.append(sid)
    if not train or not val:
        raise RuntimeError(
            f"Invalid split from {split_csv}: train={len(train)} val={len(val)} skipped={len(skipped)}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} samples without split assignment; preview={skipped[:10]}")
    return train, val


def train_one(
    *,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    selection_metric,
    checkpoint_path,
    use_kd,
    loss_weights,
) -> dict:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_metric = float("inf") if selection_metric in {"mse", "mae"} else -float("inf")
    best = None
    best_epoch = 0
    rows = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = train_student_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_kd=use_kd,
            loss_weights=loss_weights,
        )
        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, "student")
        better, metric_value = is_better_metric(val_metrics, best_metric, selection_metric)
        if better:
            best_metric = metric_value
            best = dict(val_metrics)
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                **{k: float(v) for k, v in val_metrics.items()},
                "is_best": bool(better),
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs} loss={train_metrics['loss']:.5f} "
            f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
            f"Score={val_metrics['final_score']:.4f}{' *' if better else ''}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    final_eval = evaluate_model(model, val_loader, device, "student")
    return {
        "best_epoch": best_epoch,
        "best": best or final_eval,
        "final_eval": final_eval,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--teacher-preds", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--split-csv",
        default=None,
        help="Optional split.csv from prepare_lmm_evqa_videollama2_data.py.",
    )
    parser.add_argument("--teacher-scale", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--student-hidden", type=int, default=64)
    parser.add_argument("--student-heads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-motion-clips", type=int, default=4)
    parser.add_argument("--no-student-audio", action="store_true")
    parser.add_argument("--no-student-text", action="store_true")
    parser.add_argument("--selection-metric", default="final_score")
    parser.add_argument("--alpha", type=float, default=0.5, help="Soft ECR KD weight")
    parser.add_argument("--gamma", type=float, default=0.0, help="Aesthetic auxiliary weight")
    parser.add_argument("--delta", type=float, default=0.0, help="Technical auxiliary weight")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.features)
    data = attach_teacher_predictions(data, Path(args.teacher_preds), args.teacher_scale)
    if args.split_csv:
        train_data, val_data = split_by_csv(data, Path(args.split_csv))
        print(f"Train: {len(train_data)} | Val: {len(val_data)} | split_csv={args.split_csv}")
    else:
        train_data, val_data = split_data(data, val_ratio=0.2, seed=args.split_seed)
        print(f"Train: {len(train_data)} | Val: {len(val_data)} | split_seed={args.split_seed}")

    train_plain = DataLoader(
        KDDataset(train_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch,
        shuffle=True,
    )
    val_plain = DataLoader(
        KDDataset(val_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.eval_batch,
        shuffle=False,
    )
    train_kd = DataLoader(
        KDDataset(train_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch,
        shuffle=True,
    )

    student_kwargs = {
        "hidden_dim": args.student_hidden,
        "teacher_hidden_dim": args.student_hidden,
        "n_heads": args.student_heads,
        "max_frames": args.max_frames,
        "dropout": args.dropout,
        "use_audio": not args.no_student_audio,
        "use_text": not args.no_student_text,
    }
    baseline = StudentModel(**student_kwargs).to(device)
    total_params, trainable_params = count_params(baseline)
    print(f"Student params: {total_params:,} ({trainable_params:,} trainable)")

    baseline_result = train_one(
        model=baseline,
        train_loader=train_plain,
        val_loader=val_plain,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
        checkpoint_path=save_dir / "student_baseline_best.pth",
        use_kd=False,
        loss_weights={"ecr_hard": 1.0, "ecr_soft": 0.0, "aesthetic": 0.0, "technical": 0.0},
    )

    student_kd = StudentModel(**student_kwargs).to(device)
    kd_weights = {
        "ecr_hard": 1.0,
        "ecr_soft": args.alpha,
        "kd_repr": 0.0,
        "temporal_attn": 0.0,
        "aesthetic": args.gamma,
        "technical": args.delta,
    }
    kd_result = train_one(
        model=student_kd,
        train_loader=train_kd,
        val_loader=val_plain,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
        checkpoint_path=save_dir / "student_kd_best.pth",
        use_kd=True,
        loss_weights=kd_weights,
    )

    report = {
        "features": args.features,
        "teacher_preds": args.teacher_preds,
        "teacher_scale": args.teacher_scale,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "student_params": total_params,
        "student_kwargs": student_kwargs,
        "training": vars(args),
        "kd_weights": kd_weights,
        "student_baseline": baseline_result,
        "student_kd": kd_result,
        "kd_gain_final_score": float(
            kd_result["best"]["final_score"] - baseline_result["best"]["final_score"]
        ),
        "kd_gain_srcc": float(kd_result["best"]["srcc"] - baseline_result["best"]["srcc"]),
        "kd_gain_plcc": float(kd_result["best"]["plcc"] - baseline_result["best"]["plcc"]),
    }
    out = save_dir / "student_from_prediction_teacher_report.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"Baseline best: {baseline_result['best']}")
    print(f"KD best: {kd_result['best']}")
    print(f"KD gain final_score: {report['kd_gain_final_score']:+.4f}")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()

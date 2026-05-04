#!/usr/bin/env python3
"""Train Student baseline/KD from saved teacher experiment checkpoint(s)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.models import StudentModel, count_params
from snapugc_lightkd.teacher_experiments import load_teacher_class
from snapugc_lightkd.training import (
    KDDataset,
    add_optional_model_inputs,
    evaluate_model,
    is_better_metric,
    load_data,
    split_data,
    train_student_epoch,
)


def resolve_report(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "teacher_experiment_report.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_teacher(report_path: Path, device: torch.device):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cls = load_teacher_class(report["model_version"])
    model = cls(**report["teacher_kwargs"]).to(device)
    checkpoint = Path(report["best"]["checkpoint"])
    if not checkpoint.exists():
        checkpoint = report_path.parent / checkpoint.name
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, report


@torch.no_grad()
def generate_targets(reports: list[Path], data_list, device, batch_size, max_frames, max_motion_clips):
    teachers = [load_teacher(path, device) for path in reports]
    dataset = KDDataset(data_list, max_frames=max_frames, max_motion_clips=max_motion_clips)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ecr, all_hidden, all_attn = [], [], []
    for batch in loader:
        member_ecr, member_hidden, member_attn = [], [], []
        for model, _report in teachers:
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
            }
            out = model(**add_optional_model_inputs(model, kwargs, batch, device))
            member_ecr.append(out["predicted_ecr"].detach().cpu().numpy())
            member_hidden.append(out["hidden"].detach().cpu().numpy())
            member_attn.append(out["temporal_attention"].detach().cpu().numpy())
        all_ecr.append(np.mean(member_ecr, axis=0))
        all_hidden.append(np.mean(member_hidden, axis=0))
        all_attn.append(np.mean(member_attn, axis=0))

    ecrs = np.concatenate(all_ecr)
    hiddens = np.concatenate(all_hidden)
    attns = np.concatenate(all_attn)
    enriched = []
    for i, item in enumerate(data_list):
        copied = dict(item)
        copied["teacher_ecr"] = float(ecrs[i])
        copied["teacher_hidden"] = hiddens[i].tolist()
        copied["teacher_temporal_attention"] = attns[i].tolist()
        enriched.append(copied)

    print(f"Generated teacher targets from {len(reports)} teacher(s) for {len(enriched)} samples")
    print(f"Teacher ECR targets: mean={ecrs.mean():.4f}, std={ecrs.std():.4f}")
    return enriched


def train_student_model(
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
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_metric = float("inf") if selection_metric in {"mse", "mae"} else -float("inf")
    best_metrics = None
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
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            **{k: float(v) for k, v in val_metrics.items()},
            "is_best": bool(better),
        }
        rows.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} loss={train_metrics['loss']:.5f} "
            f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
            f"Score={val_metrics['final_score']:.4f} MSE={val_metrics['mse']:.5f}"
            f"{' *' if better else ''}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    final_metrics = evaluate_model(model, val_loader, device, "student")
    if best_metrics is None:
        best_metrics = final_metrics
    return {
        "best_epoch": best_epoch,
        "best": best_metrics,
        "final_eval": final_metrics,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-report", action="append", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--student-hidden", type=int, default=256)
    parser.add_argument("--student-heads", type=int, default=4)
    parser.add_argument("--student-epochs", type=int, default=60)
    parser.add_argument("--student-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-motion-clips", type=int, default=4)
    parser.add_argument(
        "--teacher-target-max-frames",
        type=int,
        default=None,
        help="Frames used when generating teacher targets. Defaults to --max-frames.",
    )
    parser.add_argument(
        "--teacher-target-max-motion-clips",
        type=int,
        default=None,
        help="Motion clips used when generating teacher targets. Defaults to --max-motion-clips.",
    )
    parser.add_argument("--no-student-audio", action="store_true")
    parser.add_argument("--no-student-text", action="store_true")
    parser.add_argument("--selection-metric", default="final_score")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--attn-kd", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    reports = [resolve_report(path) for path in args.teacher_report]
    data = load_data(args.data)
    train_data, val_data = split_data(data, val_ratio=0.2, seed=args.split_seed)
    train_data = [copy.deepcopy(item) for item in train_data]
    val_data = [copy.deepcopy(item) for item in val_data]
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | split_seed={args.split_seed}")
    teacher_target_max_frames = args.teacher_target_max_frames or args.max_frames
    teacher_target_max_motion_clips = (
        args.teacher_target_max_motion_clips or args.max_motion_clips
    )

    train_teacher_data = generate_targets(
        reports,
        train_data,
        device,
        args.eval_batch,
        teacher_target_max_frames,
        teacher_target_max_motion_clips,
    )
    val_teacher_data = generate_targets(
        reports,
        val_data,
        device,
        args.eval_batch,
        teacher_target_max_frames,
        teacher_target_max_motion_clips,
    )

    train_loader_plain = DataLoader(
        KDDataset(train_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader_plain = DataLoader(
        KDDataset(val_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.eval_batch,
        shuffle=False,
    )
    train_loader_kd = DataLoader(
        KDDataset(train_teacher_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader_kd = DataLoader(
        KDDataset(val_teacher_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.eval_batch,
        shuffle=False,
    )

    teacher_hidden = json.loads(reports[0].read_text(encoding="utf-8"))["teacher_kwargs"]["hidden_dim"]
    baseline = StudentModel(
        hidden_dim=args.student_hidden,
        teacher_hidden_dim=teacher_hidden,
        n_heads=args.student_heads,
        max_frames=args.max_frames,
        dropout=args.dropout,
        use_audio=not args.no_student_audio,
        use_text=not args.no_student_text,
    ).to(device)
    total_params, trainable_params = count_params(baseline)
    print(f"Student params: {total_params:,} ({trainable_params:,} trainable)")

    print("\n=== Student baseline ===")
    baseline_result = train_student_model(
        model=baseline,
        train_loader=train_loader_plain,
        val_loader=val_loader_plain,
        device=device,
        epochs=args.student_epochs,
        lr=args.student_lr,
        weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
        checkpoint_path=save_dir / "student_baseline_best.pth",
        use_kd=False,
        loss_weights={
            "ecr_hard": 1.0,
            "ecr_soft": 0.0,
            "kd_repr": 0.0,
            "temporal_attn": 0.0,
            "aesthetic": 0.0,
            "technical": 0.0,
        },
    )

    print("\n=== Student KD ===")
    student_kd = StudentModel(
        hidden_dim=args.student_hidden,
        teacher_hidden_dim=teacher_hidden,
        n_heads=args.student_heads,
        max_frames=args.max_frames,
        dropout=args.dropout,
        use_audio=not args.no_student_audio,
        use_text=not args.no_student_text,
    ).to(device)
    kd_weights = {
        "ecr_hard": 1.0,
        "ecr_soft": args.alpha,
        "kd_repr": args.beta,
        "temporal_attn": args.attn_kd,
        "aesthetic": args.gamma,
        "technical": args.delta,
    }
    kd_result = train_student_model(
        model=student_kd,
        train_loader=train_loader_kd,
        val_loader=val_loader_kd,
        device=device,
        epochs=args.student_epochs,
        lr=args.student_lr,
        weight_decay=args.weight_decay,
        selection_metric=args.selection_metric,
        checkpoint_path=save_dir / "student_kd_best.pth",
        use_kd=True,
        loss_weights=kd_weights,
    )

    report = {
        "data": args.data,
        "teacher_reports": [str(path) for path in reports],
        "n_train": len(train_data),
        "n_val": len(val_data),
        "student_params": total_params,
        "student_kwargs": {
            "hidden_dim": args.student_hidden,
            "n_heads": args.student_heads,
            "max_frames": args.max_frames,
            "dropout": args.dropout,
            "teacher_hidden_dim": teacher_hidden,
            "use_audio": not args.no_student_audio,
            "use_text": not args.no_student_text,
        },
        "teacher_target_kwargs": {
            "max_frames": teacher_target_max_frames,
            "max_motion_clips": teacher_target_max_motion_clips,
        },
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
    out = save_dir / "student_from_teacher_report.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nBaseline best: {baseline_result['best']}")
    print(f"KD best: {kd_result['best']}")
    print(f"KD gain final_score: {report['kd_gain_final_score']:+.4f}")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train student baseline and artifact-aware KD from official teacher shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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
    split_rows,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent, compute_losses  # noqa: E402


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


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
        )
        preds.extend(outputs["predicted_ecr"].detach().cpu().numpy().tolist())
        true.extend(batch["ecr_true"].detach().cpu().numpy().tolist())
        teacher.extend(batch["teacher_ecr"].detach().cpu().numpy().tolist())
    metrics = metrics_from_arrays(np.array(preds), np.array(true))
    metrics["teacher_plcc_on_split"] = metrics_from_arrays(np.array(teacher), np.array(true))["plcc"]
    metrics["teacher_srcc_on_split"] = metrics_from_arrays(np.array(teacher), np.array(true))["srcc"]
    metrics["teacher_final_score_on_split"] = metrics_from_arrays(
        np.array(teacher), np.array(true)
    )["final_score"]
    return metrics


def train_one(
    *,
    name: str,
    model,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_kd: bool,
    weights: dict[str, float],
    repr_loss: str,
    save_path: Path,
) -> dict[str, object]:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.02)
    best_score = -1e9
    best_epoch = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sums: dict[str, float] = {}
        n_batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["clip_inputs"],
                batch["clip_mask"],
                batch["text_inputs"],
                batch["text_mask"],
            )
            loss, loss_parts = compute_losses(
                outputs,
                batch,
                use_kd=use_kd,
                weights=weights,
                repr_loss=repr_loss,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--input-preset", default="visual_text")
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-kind", choices=("both", "baseline", "kd"), default="both")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument(
        "--repr-loss",
        choices=("raw_mse", "normalized_mse", "cosine"),
        default="raw_mse",
    )
    parser.add_argument("--soft-weight", type=float, default=0.5)
    parser.add_argument("--clip-weight", type=float, default=0.2)
    parser.add_argument("--temporal-weight", type=float, default=0.2)
    parser.add_argument("--fusion-weight", type=float, default=0.1)
    parser.add_argument("--attention-weight", type=float, default=0.05)
    parser.add_argument("--hard-rank-weight", type=float, default=0.0)
    parser.add_argument("--teacher-rank-weight", type=float, default=0.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    input_config = StudentInputConfig.from_preset(args.input_preset)
    rows = load_official_artifact_rows(
        args.artifact_dir,
        args.labels_csv,
        ragged_keys=artifact_keys_for_input_config(input_config),
    )
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = OfficialTeacherArtifactDataset(train_rows, input_config, max_clips=args.max_clips)
    val_dataset = OfficialTeacherArtifactDataset(val_rows, input_config, max_clips=args.max_clips)
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
    }
    print(
        f"Loaded official teacher artifacts: total={len(rows)} "
        f"train={len(train_rows)} val={len(val_rows)} clip_dim={train_dataset.clip_dim} "
        f"input_preset={args.input_preset}",
        flush=True,
    )

    baseline_result = None
    if args.run_kind in ("both", "baseline"):
        baseline = OfficialArtifactStudent(**model_kwargs).to(device)
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
            weights={"hard_ecr": 1.0, "hard_rank": args.hard_rank_weight},
            repr_loss=args.repr_loss,
            save_path=save_dir / "student_baseline_best.pth",
        )
    elif args.baseline_report:
        with Path(args.baseline_report).open("r", encoding="utf-8") as f:
            loaded_baseline = json.load(f).get("baseline")
        if loaded_baseline is not None:
            baseline_result = {
                "best_epoch": loaded_baseline.get("best_epoch"),
                "best": loaded_baseline.get("best"),
                "checkpoint": loaded_baseline.get("checkpoint"),
            }

    kd_weights = {
        "hard_ecr": 1.0,
        "hard_rank": args.hard_rank_weight,
        "soft_ecr": args.soft_weight,
        "clip_ecr": args.clip_weight,
        "temporal_hidden": args.temporal_weight,
        "fusion_hidden": args.fusion_weight,
        "attention": args.attention_weight,
        "teacher_rank": args.teacher_rank_weight,
    }
    kd_result = None
    if args.run_kind in ("both", "kd"):
        kd_model = OfficialArtifactStudent(**model_kwargs).to(device)
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
            save_path=save_dir / "student_kd_best.pth",
        )

    report = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "input_preset": args.input_preset,
        "input_config": input_config.__dict__,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "model_kwargs": model_kwargs,
        "repr_loss": args.repr_loss,
        "run_kind": args.run_kind,
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
        "repr_loss": args.repr_loss,
        "baseline_final": baseline_result["best"]["final_score"] if baseline_result else None,
        "kd_best_epoch": kd_result["best_epoch"] if kd_result else None,
        "kd_final": kd_result["best"]["final_score"] if kd_result else None,
        "kd_gain_final_score": report["kd_gain_final_score"],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

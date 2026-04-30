#!/usr/bin/env python3
"""Evaluate prediction ensembles across teacher experiment checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.teacher_experiments import load_teacher_class
from snapugc_lightkd.training import (
    KDDataset,
    add_optional_model_inputs,
    load_data,
    split_data,
)
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    plcc = 0.0 if y_pred.std() < 1e-12 else float(pearsonr(y_true, y_pred)[0])
    srcc = float(spearmanr(y_true, y_pred)[0])
    return {
        "plcc": plcc,
        "srcc": srcc,
        "ktau": float(kendalltau(y_true, y_pred)[0]),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "final_score": float(0.6 * srcc + 0.4 * plcc),
        "pred_mean": float(y_pred.mean()),
        "pred_std": float(y_pred.std()),
    }


@torch.no_grad()
def predict(report_path: Path, data, device: torch.device, batch_size: int):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cls = load_teacher_class(report["model_version"])
    model = cls(**report["teacher_kwargs"]).to(device)
    model.load_state_dict(torch.load(report["best"]["checkpoint"], map_location=device, weights_only=True))
    model.eval()

    training = report.get("training") or {}
    dataset = KDDataset(
        data,
        max_frames=report["teacher_kwargs"].get("max_frames", 16),
        max_motion_clips=report["teacher_kwargs"].get("max_motion_clips", 4),
        opening_seconds=training.get("opening_seconds"),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
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
        }
        out = model(**add_optional_model_inputs(model, kwargs, batch, device))
        preds.append(out["predicted_ecr"].cpu().numpy())
        trues.append(batch["ecr"].numpy())
    return np.concatenate(preds), np.concatenate(trues), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-models", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    reports = sorted(root.glob("*/teacher_experiment_report.json"))
    if not reports:
        raise FileNotFoundError(f"No reports under {root}")

    ranked = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        ranked.append((float(report["best"]["final_score"]), path))
    ranked.sort(reverse=True)
    reports = [path for _, path in ranked[: args.max_models]]

    all_data = load_data(args.data)
    _, val_data = split_data(all_data, val_ratio=0.2, seed=42)
    device = torch.device(args.device)

    preds = {}
    y_true = None
    rows = []
    for path in reports:
        pred, true, report = predict(path, val_data, device, args.batch)
        y_true = true if y_true is None else y_true
        name = path.parent.name
        preds[name] = pred
        rows.append({"name": name, "members": name, **score(y_true, pred)})
        print(f"Loaded {name}: {rows[-1]['final_score']:.4f}")

    names = list(preds)
    for r in range(2, min(len(names), args.max_models) + 1):
        for combo in itertools.combinations(names, r):
            ens = np.mean([preds[name] for name in combo], axis=0)
            rows.append({"name": f"ens_{r}", "members": "+".join(combo), **score(y_true, ens)})

    rows.sort(key=lambda row: row["final_score"], reverse=True)
    print("\nBest ensembles:")
    for row in rows[:20]:
        print(
            f"{row['final_score']:.4f} srcc={row['srcc']:.4f} plcc={row['plcc']:.4f} "
            f"members={row['members']}"
        )

    out = Path(args.out) if args.out else root / "teacher_ensemble_results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect teacher experiment reports into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing teacher experiment subdirs")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    reports = sorted(root.glob("*/teacher_experiment_report.json"))
    if not reports:
        raise FileNotFoundError(f"No teacher_experiment_report.json files under {root}")

    rows = []
    for path in reports:
        report = load_report(path)
        best = report.get("best") or {}
        training = report.get("training") or {}
        teacher_kwargs = report.get("teacher_kwargs") or {}
        rows.append(
            {
                "experiment": path.parent.name,
                "model_version": report.get("model_version"),
                "params": report.get("params"),
                "best_epoch": report.get("best_epoch"),
                "final_score": best.get("final_score"),
                "srcc": best.get("srcc"),
                "plcc": best.get("plcc"),
                "ktau": best.get("ktau"),
                "mse": best.get("mse"),
                "mae": best.get("mae"),
                "pred_mean": best.get("pred_mean"),
                "pred_std": best.get("pred_std"),
                "hidden": teacher_kwargs.get("hidden_dim"),
                "blocks": teacher_kwargs.get("n_blocks"),
                "heads": teacher_kwargs.get("n_heads"),
                "dropout": teacher_kwargs.get("dropout"),
                "aux_weight": teacher_kwargs.get("aux_weight"),
                "lr": training.get("teacher_lr"),
                "weight_decay": training.get("weight_decay"),
                "warmup_epochs": training.get("warmup_epochs"),
                "batch": training.get("batch"),
                "opening_seconds": training.get("opening_seconds"),
                "seed": training.get("seed"),
            }
        )

    rows.sort(key=lambda row: float(row["final_score"] or -999), reverse=True)
    out_path = Path(args.out) if args.out else root / "teacher_experiment_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    for row in rows:
        print(
            f"{row['experiment']:<32} score={float(row['final_score']):.4f} "
            f"srcc={float(row['srcc']):.4f} plcc={float(row['plcc']):.4f} "
            f"params={int(row['params']):,}"
        )


if __name__ == "__main__":
    main()

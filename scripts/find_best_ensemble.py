#!/usr/bin/env python3
import sys
import json
import torch
import numpy as np
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from snapugc_lightkd.official_artifacts import (
    load_official_artifact_rows,
    split_rows,
)
from evaluate_student_ensemble import (
    load_run,
    predict_one,
    metrics_from_arrays,
)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    artifact_dir = "results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts"
    labels_csv = "data/train_subset_balanced_5000.csv"

    print("Loading official artifacts...")
    rows = load_official_artifact_rows(artifact_dir, labels_csv)
    _, val_rows = split_rows(rows, val_ratio=0.2, seed=42)

    # Candidate runs
    tier3_runs = sorted(Path("results/loss_ablation_controlled_2026").glob("tier3_current_*"))
    candidate_dirs = [
        str(tier3_runs[0]) if tier3_runs else "results/loss_ablation_controlled_2026/tier3_current",
        "results/loss_research_2026/A_rich_s43",
        "results/loss_research_2026/A_lean_s43",
        "results/kd_tuning_official_5k/v35_teacher_action_caption_clipadd_kd",
        "results/kd_tuning_official_5k/student_kd_basic_soft_mse_clip_vitb32_clipadd",
        "results/kd_tuning_official_5k/improve_clip_vitb32_clip_add_e100",
    ]

    valid_runs = []
    for d in candidate_dirs:
        p = Path(d)
        if (p / "official_student_kd_report.json").exists():
            valid_runs.append(p)

    print(f"Found {len(valid_runs)} valid runs to search over.")

    # Predict outputs for each run
    run_predictions = {}
    true_labels = None

    for run in valid_runs:
        print(f"Predicting for run: {run.name} ...")
        try:
            report, checkpoint = load_run(run)
            preds, true, _, _ = predict_one(
                report=report,
                checkpoint=checkpoint,
                rows=val_rows,
                device=device,
                batch_size=128,
                clip_offset=0,
            )
            run_predictions[run.name] = preds
            if true_labels is None:
                true_labels = true
        except Exception as e:
            print(f"Error predicting for {run.name}: {e}")

    if not run_predictions:
        print("No successful predictions!")
        return

    # Find the best combination and weights
    best_score = -1.0
    best_combo = None
    best_weights = None

    run_names = list(run_predictions.keys())
    print("\nIndividual Performance:")
    for name in run_names:
        m = metrics_from_arrays(run_predictions[name], true_labels)
        print(f"  {name:50s}: SRCC={m['srcc']:.4f} PLCC={m['plcc']:.4f} Score={m['final_score']:.4f}")

    print("\nSearching combinations of up to 4 models...")
    for k in range(2, min(5, len(run_names) + 1)):
        for combo in combinations(run_names, k):
            preds_stack = np.stack([run_predictions[name] for name in combo])
            
            # 1. Simple average ensemble
            avg_pred = preds_stack.mean(axis=0)
            m = metrics_from_arrays(avg_pred, true_labels)
            if m["final_score"] > best_score:
                best_score = m["final_score"]
                best_combo = combo
                best_weights = [1.0 / k] * k
            
            # 2. Weighted search (coarse grid search for best weights)
            if k == 2:
                for w in np.linspace(0.1, 0.9, 9):
                    w_pred = w * run_predictions[combo[0]] + (1.0 - w) * run_predictions[combo[1]]
                    m = metrics_from_arrays(w_pred, true_labels)
                    if m["final_score"] > best_score:
                        best_score = m["final_score"]
                        best_combo = combo
                        best_weights = [w, 1.0 - w]
            elif k == 3:
                for w1 in [0.2, 0.3, 0.4, 0.5]:
                    for w2 in [0.2, 0.3, 0.4, 0.5]:
                        w3 = 1.0 - w1 - w2
                        if w3 < 0.0:
                            continue
                        w_pred = (
                            w1 * run_predictions[combo[0]] +
                            w2 * run_predictions[combo[1]] +
                            w3 * run_predictions[combo[2]]
                        )
                        m = metrics_from_arrays(w_pred, true_labels)
                        if m["final_score"] > best_score:
                            best_score = m["final_score"]
                            best_combo = combo
                            best_weights = [w1, w2, w3]

    print("\n" + "=" * 50)
    print("BEST ENSEMBLE FOUND:")
    print(f"Score: {best_score:.5f}")
    print("Models and Weights:")
    for name, w in zip(best_combo, best_weights):
        m = metrics_from_arrays(run_predictions[name], true_labels)
        print(f"  - {name} (Weight: {w:.2f}) -> Individual Score: {m['final_score']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()

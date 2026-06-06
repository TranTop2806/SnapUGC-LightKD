#!/usr/bin/env python3
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from snapugc_lightkd.official_artifacts import (
    OfficialTeacherArtifactDataset,
    StudentInputConfig,
    artifact_keys_for_input_config,
    collate_student_batch,
    load_official_artifact_rows,
    split_rows,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent
from train_official_student_kd import (
    attach_quality_features,
    attach_dover_features,
    attach_lite_action,
    attach_pseudo_labels,
    evaluate,
    metrics_from_arrays,
)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Same parameters as training
    artifact_dir = "results/original_snapugc_official_balanced_5000_artifacts_g2_32/teacher_artifacts"
    labels_csv = "data/train_subset_balanced_5000.csv"
    save_dir = Path("results/kd_tuning_official_5k/improve_large_h256_l3_lite_action")

    input_config = StudentInputConfig.from_preset("visual_text_sound")
    ragged_keys = set(artifact_keys_for_input_config(input_config))
    ragged_keys.update({"action_feature", "caption_feature"})

    print("Loading official artifacts...")
    rows = load_official_artifact_rows(artifact_dir, labels_csv, ragged_keys=tuple(ragged_keys))

    print("Attaching features...")
    quality_dim = attach_quality_features(rows, "results/clip_vitb32_keyframe_features_5000.npz")
    input_config = input_config.with_quality_features(True, quality_dim, "clip_add")

    lite_action_dim = attach_lite_action(rows, "results/lite_action_features_5000.npz")
    input_config = input_config.with_lite_action(True, lite_action_dim)

    train_rows, val_rows = split_rows(rows, val_ratio=0.2, seed=42)
    val_dataset = OfficialTeacherArtifactDataset(val_rows, input_config, max_clips=16)
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_student_batch,
    )

    model_kwargs = {
        "clip_input_dim": val_dataset.clip_dim,
        "hidden_dim": 256,
        "max_clips": 16,
        "n_layers": 3,
        "n_heads": 8,
        "dropout": 0.25,
        "fusion_mode": "concat",
        "projection_head": "mlp",
        "quality_input_dim": quality_dim,
        "quality_fusion": "clip_add",
        "ecr_bins": 0,
        "temporal_aggregation": "attention",
        "local_clips": 5,
        "semantic_gated_fusion": False,
        "shared_distill_dim": 0,
        "fusion_experts": 1,
        "dover_input_dim": 0,
        "dover_fusion": "none",
        "use_hallucination": True,
        "hallucination_feedback": True,
        "hallucination_feedback_dim": 0,
        "temporal_conv": "none",
        "shared_transformer_weights": False,
        "drop_path": 0.0,
    }

    print("Initializing model...")
    model = OfficialArtifactStudent(**model_kwargs).to(device)
    checkpoint_path = save_dir / "student_kd_best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    print("Evaluating best saved checkpoint on validation split...")
    metrics = evaluate(model, val_loader, device)
    print("\nValidation Metrics:")
    print(json.dumps(metrics, indent=2))

    report = {
        "artifact_dir": str(Path(artifact_dir).resolve()),
        "labels_csv": str(Path(labels_csv).resolve()),
        "input_preset": "visual_text_sound",
        "input_config": input_config.__dict__,
        "quality_features": "results/clip_vitb32_keyframe_features_5000.npz",
        "quality_fusion": "clip_add",
        "dover_features": None,
        "lite_action_features": "results/lite_action_features_5000.npz",
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "seed": 42,
        "model_kwargs": model_kwargs,
        "kd": {
            "best_epoch": 35,
            "best": metrics,
        },
    }

    with (save_dir / "official_student_kd_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved successfully to save_dir!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark batch-1 student forward latency on one real cached video sample."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_student_ensemble import compatible_model_kwargs, load_run  # noqa: E402
from train_official_student_kd import attach_lite_action, attach_quality_features  # noqa: E402

from snapugc_lightkd.official_artifacts import (  # noqa: E402
    OfficialTeacherArtifactDataset,
    StudentInputConfig,
    collate_student_batch,
    load_official_artifact_rows,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402

DEFAULT_RUNS = (
    "Baseline Student (No KD)=results/kd_tuning_official_5k/"
    "baseline_clip_vitb32_clipadd",
    "Student KD basic=results/kd_tuning_official_5k/"
    "student_kd_basic_soft_mse_clip_vitb32_clipadd",
    "Student KD full=results/loss_ablation_controlled_2026/"
    "tier3_current_nonhalluc",
    "Proper / Full Pipeline KD=results/proper_kd/medium_kd_h192_l2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="results/original_snapugc_official_balanced_5000_artifacts_g2_32/"
        "teacher_artifacts",
    )
    parser.add_argument("--labels-csv", default="data/train_subset_balanced_5000.csv")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        metavar="LABEL=REPORT_OR_RUN_DIR",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--hardware-label", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def input_config_from_report(report: dict[str, object]) -> StudentInputConfig:
    info = dict(report.get("input_config", {}))
    if not info:
        return StudentInputConfig.from_preset(str(report.get("input_preset", "visual_text_sound")))
    allowed = set(StudentInputConfig.__dataclass_fields__)
    return StudentInputConfig(**{key: value for key, value in info.items() if key in allowed})


def resolve_feature_path(value: object) -> Path:
    path = Path(str(value))
    if not path.exists():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_batch_and_model(
    report: dict[str, object],
    checkpoint: Path,
    row: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, object], OfficialArtifactStudent]:
    config = input_config_from_report(report)
    rows = [{key: value for key, value in row.items()}]

    if config.use_quality_features:
        attach_quality_features(rows, str(resolve_feature_path(report["quality_features"])))
    if config.use_lite_action:
        attach_lite_action(rows, str(resolve_feature_path(report["lite_action_features"])))

    max_clips = int(dict(report.get("model_kwargs", {})).get("max_clips", 16))
    dataset = OfficialTeacherArtifactDataset(rows, config, max_clips=max_clips)
    batch = collate_student_batch([dataset[0]])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    model_kwargs = compatible_model_kwargs(dict(report.get("model_kwargs", {})))
    model_kwargs["clip_input_dim"] = dataset.clip_dim
    model = OfficialArtifactStudent(**model_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return batch, model


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def forward(model: OfficialArtifactStudent, batch: dict[str, object]) -> torch.Tensor:
    outputs = model(
        batch["clip_inputs"],
        batch["clip_mask"],
        batch["text_inputs"],
        batch["text_mask"],
        batch.get("dover_inputs"),
        batch.get("quality_inputs"),
    )
    return outputs["predicted_ecr"]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark(
    model: OfficialArtifactStudent,
    batch: dict[str, object],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, float], float]:
    with torch.inference_mode():
        for _ in range(warmup):
            prediction = forward(model, batch)
        synchronize(device)

        latencies_ms = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            prediction = forward(model, batch)
            synchronize(device)
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    return (
        {
            "mean_ms": statistics.fmean(latencies_ms),
            "median_ms": statistics.median(latencies_ms),
            "p90_ms": percentile(latencies_ms, 90),
            "std_ms": statistics.pstdev(latencies_ms),
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
        },
        float(prediction.item()),
    )


def main() -> None:
    args = parse_args()
    if args.sample_index < 0 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("sample-index and warmup must be non-negative; repeats must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    rows = load_official_artifact_rows(
        args.artifact_dir,
        args.labels_csv,
        max_rows=args.sample_index + 1,
    )
    row = rows[args.sample_index]
    device = torch.device(args.device)
    run_results = []

    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Run must use LABEL=PATH syntax: {spec}")
        label, raw_path = spec.split("=", 1)
        report, checkpoint = load_run(resolve_feature_path(raw_path))
        batch, model = prepare_batch_and_model(report, checkpoint, row, device)
        timing, prediction = benchmark(model, batch, device, args.warmup, args.repeats)
        run_results.append(
            {
                "label": label,
                "run": str(Path(raw_path)),
                "checkpoint": str(checkpoint),
                "input_preset": report.get("input_preset"),
                "clip_shape": list(batch["clip_inputs"].shape),
                "text_shape": list(batch["text_inputs"].shape),
                "prediction": prediction,
                **timing,
            }
        )
        print(
            f"{label}: median={timing['median_ms']:.3f} ms "
            f"p90={timing['p90_ms']:.3f} ms",
            flush=True,
        )

    result = {
        "scope": "student forward only; cached input tensors to ECR prediction",
        "excludes": [
            "model and checkpoint loading",
            "video/audio decoding",
            "raw-input feature extraction",
            "batch collation and host-to-device transfer",
        ],
        "sample_id": row["Id"],
        "sample_index": args.sample_index,
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "device": str(device),
        "hardware": args.hardware_label or platform.platform(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "runs": run_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

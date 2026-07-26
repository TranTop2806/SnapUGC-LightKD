#!/usr/bin/env python3
"""Benchmark Proper KD raw-video inference and a Full KD cached student head."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_student_latency import prepare_batch_and_model  # noqa: E402
from snapugc_lightkd.official_artifacts import load_official_artifact_rows  # noqa: E402
from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402
from snapugc_lightkd.student_native import build_native_student_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--proper-report", required=True)
    parser.add_argument("--proper-checkpoint", required=True)
    parser.add_argument("--full-report")
    parser.add_argument("--full-checkpoint")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--labels-csv")
    parser.add_argument("--proper-repeats", type=int, default=5)
    parser.add_argument("--head-repeats", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize_ms(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "std_ms": float(array.std()),
    }


def load_student(report_path: Path, checkpoint_path: Path, device: torch.device):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = OfficialArtifactStudent(**dict(report["model_kwargs"])).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return report, model


def forward(model: OfficialArtifactStudent, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["clip_inputs"],
        batch["clip_mask"],
        batch["text_inputs"],
        batch["text_mask"],
        batch.get("dover_inputs"),
        batch.get("quality_inputs"),
    )["predicted_ecr"]


def proper_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    report, model = load_student(
        Path(args.proper_report), Path(args.proper_checkpoint), device
    )
    model_kwargs = dict(report["model_kwargs"])

    def extract_and_predict() -> tuple[float, dict[str, torch.Tensor]]:
        native = build_native_student_inputs(
            args.video,
            title=args.title,
            description=args.description,
            max_clips=int(model_kwargs.get("max_clips", 16)),
            clip_dim=int(model_kwargs["clip_input_dim"]),
            device=device,
            input_preset="clip_mobilenet_text",
        )
        batch = {
            key: value.to(device)
            for key, value in native.as_batch().items()
            if isinstance(value, torch.Tensor)
        }
        with torch.inference_mode():
            prediction = forward(model, batch)
        sync(device)
        return float(prediction.item()), batch

    sync(device)
    start = time.perf_counter_ns()
    prediction, batch = extract_and_predict()
    cold_ms = (time.perf_counter_ns() - start) / 1_000_000

    raw_latencies = []
    for _ in range(args.proper_repeats):
        sync(device)
        start = time.perf_counter_ns()
        prediction, batch = extract_and_predict()
        raw_latencies.append((time.perf_counter_ns() - start) / 1_000_000)

    with torch.inference_mode():
        for _ in range(20):
            prediction_tensor = forward(model, batch)
        sync(device)
        head_latencies = []
        for _ in range(args.head_repeats):
            sync(device)
            start = time.perf_counter_ns()
            prediction_tensor = forward(model, batch)
            sync(device)
            head_latencies.append((time.perf_counter_ns() - start) / 1_000_000)

    return {
        "scope": "raw video decode + CLIP/MobileNet/text extraction + student forward",
        "cold_ms": cold_ms,
        "warm_raw_video": summarize_ms(raw_latencies),
        "student_forward_only": summarize_ms(head_latencies),
        "prediction": float(prediction_tensor.item()),
        "raw_repeats": args.proper_repeats,
        "head_repeats": args.head_repeats,
    }


def full_cached_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, object] | None:
    required = (
        args.full_report,
        args.full_checkpoint,
        args.artifact_dir,
        args.labels_csv,
    )
    if not all(required):
        return None
    report = json.loads(Path(args.full_report).read_text(encoding="utf-8"))
    rows = load_official_artifact_rows(args.artifact_dir, args.labels_csv, max_rows=1)
    batch, model = prepare_batch_and_model(
        report, Path(args.full_checkpoint), rows[0], device
    )
    with torch.inference_mode():
        for _ in range(20):
            prediction = forward(model, batch)
        sync(device)
        latencies = []
        for _ in range(args.head_repeats):
            sync(device)
            start = time.perf_counter_ns()
            prediction = forward(model, batch)
            sync(device)
            latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "scope": "cached teacher-frontend tensors + CLIP tensors to student ECR",
        "latency": summarize_ms(latencies),
        "prediction": float(prediction.item()),
        "head_repeats": args.head_repeats,
        "sample_id": rows[0]["Id"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    result = {
        "video": str(Path(args.video).resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "proper_kd": proper_benchmark(args, device),
        "full_kd_cached": full_cached_benchmark(args, device),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

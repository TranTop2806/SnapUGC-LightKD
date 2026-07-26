#!/usr/bin/env python3
"""Benchmark Full KD from raw video through the selected teacher frontend."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
ECR_DIR = ROOT / "third_party" / "SnapUGC_Engagement" / "ECR_inference"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ECR_DIR))

from modules.EVQA import EVQA  # noqa: E402
from modules.distort import Distortion  # noqa: E402
from modules.efficientnet_v2 import EfficientNetV2  # noqa: E402
from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--frame-batch", type=int, default=24)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "std_ms": float(array.std()),
    }


def load_frames(video_path: Path) -> tuple[torch.Tensor, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    bgr_frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        bgr_frames.append(frame)
    capture.release()
    if not bgr_frames:
        raise RuntimeError(f"Could not decode frames from {video_path}")
    frames = torch.from_numpy(np.stack(bgr_frames)).float().div_(255.0)
    return frames.permute(0, 3, 1, 2), bgr_frames


def prepare_official_frames(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    _, _, height, width = frames.shape
    if height < width:
        frames = frames.permute(0, 1, 3, 2)
    frames = F.interpolate(frames, (452, 256), mode="bicubic")
    frames = torch.flip(frames.to(device), (1,))
    residual = frames.shape[0] % 16
    multiple = frames.shape[0] // 16
    if residual > 3:
        frames = torch.cat((frames[: multiple * 16], frames[-16:]), dim=0)
    elif residual:
        frames = frames[: multiple * 16]
    if frames.shape[0] < 16:
        raise RuntimeError("Full KD frontend needs at least 16 decoded frames")
    return frames


@torch.inference_mode()
def distortion_features(
    model: nn.Module, frames: torch.Tensor, frame_batch: int, avg: nn.Module
) -> torch.Tensor:
    outputs = []
    for start in range(0, frames.shape[0], frame_batch):
        feat_x, _, _, _ = model(frames[start : start + frame_batch])
        outputs.append(avg(feat_x))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def semantic_features(
    model: nn.Module, frames: torch.Tensor, frame_batch: int, avg: nn.Module
) -> torch.Tensor:
    outputs = []
    normalize = torchvision.transforms.Normalize(0.5, 0.5)
    for start in range(0, frames.shape[0], frame_batch):
        batch = frames[start : start + frame_batch]
        features = model.get_features(normalize(batch))
        outputs.append(torch.cat([avg(feature) for feature in features], dim=1))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def frame_fusion(evqa: EVQA, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    feat1 = feat1.squeeze(3).squeeze(2)
    feat2 = feat2.squeeze(3).squeeze(2)
    merged = torch.cat((evqa.fc1(feat1), evqa.fc3(feat2)), dim=1)
    merged = merged.view(merged.shape[0] // 16, 16 * evqa.num_class * 2)
    return evqa.fc_merge12(merged)


def uniform_indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    step = total / count
    return [int(step * idx + step / 2) for idx in range(count)]


@torch.inference_mode()
def clip_features(
    model: nn.Module,
    preprocess,
    bgr_frames: list[np.ndarray],
    count: int,
    device: torch.device,
) -> torch.Tensor:
    from PIL import Image

    indices = uniform_indices(len(bgr_frames), 16)
    images = [Image.fromarray(cv2.cvtColor(bgr_frames[idx], cv2.COLOR_BGR2RGB)) for idx in indices]
    batch = torch.stack([preprocess(image) for image in images]).to(device)
    encoded = F.normalize(model.encode_image(batch), dim=-1).float()
    if encoded.shape[0] < count:
        encoded = torch.cat((encoded, encoded[-1:].repeat(count - encoded.shape[0], 1)), dim=0)
    return encoded[:count]


def ensure_sample_rate(rate: int, waveform: np.ndarray, desired: int = 16000):
    if rate != desired:
        length = int(round(float(len(waveform)) / rate * desired))
        waveform = scipy.signal.resample(waveform, length)
    return desired, waveform


def load_yamnet_classes(model) -> list[str]:
    import tensorflow as tf

    path = model.class_map_path().numpy()
    with tf.io.gfile.GFile(path) as handle:
        return [row["display_name"] for row in csv.DictReader(handle)]


def sound_labels(video_path: Path, yamnet, classes: list[str]) -> str:
    import tensorflow as tf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-i", str(video_path), "-ac", "1", "-f", "wav", str(wav_path)],
            check=True,
        )
        rate, waveform = wavfile.read(wav_path, "rb")
        _, waveform = ensure_sample_rate(rate, waveform)
        scores, _, _ = yamnet(waveform / tf.int16.max)
        top = scores.numpy().mean(axis=0).argsort()[-5:][::-1]
        return ", ".join(classes[idx] for idx in top)
    finally:
        wav_path.unlink(missing_ok=True)


@torch.inference_mode()
def text_features(evqa: EVQA, texts: list[str], device: torch.device) -> torch.Tensor:
    tokens = evqa.tokenizer(
        texts,
        padding="max_length",
        max_length=evqa.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return evqa.text_encoder(tokens.input_ids.to(device))[0].mean(dim=1).float()


def build_models(checkpoint_dir: Path, device: torch.device):
    import open_clip
    import tensorflow as tf
    import tensorflow_hub as hub
    from transformers import CLIPTextModel, CLIPTokenizer

    # Match the official L4 run, where YAMNet executes on CPU and PyTorch owns
    # the GPU. Otherwise TensorFlow preallocates nearly all L4 memory.
    tf.config.set_visible_devices([], "GPU")

    model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device).eval()
    evqa = EVQA(3, 16, tokenizer, text_encoder).to(device).eval()
    state = torch.load(checkpoint_dir / "EVQA.pth", map_location=device, weights_only=False)["params"]
    evqa.load_state_dict(state, strict=False)

    distortion = Distortion().to(device).eval()
    state = torch.load(
        checkpoint_dir / "net_distort6_g_latest.pth", map_location=device, weights_only=False
    )["params"]
    distortion.load_state_dict(state)

    # Parameter values do not change latency; avoid the retired upstream
    # checkpoint format while preserving the exact EfficientNetV2-S graph.
    semantic = EfficientNetV2("s", in_channels=3, n_classes=50, pretrained=False).to(device).eval()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    clip_model.eval()
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    yamnet_classes = load_yamnet_classes(yamnet)

    # The retained Full KD run uses frame_fusion_feature plus CLIP clip-add.
    student = OfficialArtifactStudent(
        clip_input_dim=1024,
        text_input_dim=768,
        hidden_dim=96,
        max_clips=16,
        n_layers=1,
        n_heads=4,
        dropout=0.25,
        quality_input_dim=512,
        quality_fusion="clip_add",
    ).to(device).eval()
    return evqa, distortion, semantic, clip_model, clip_preprocess, yamnet, yamnet_classes, student


@torch.inference_mode()
def student_forward(
    student: OfficialArtifactStudent,
    frame_inputs: torch.Tensor,
    quality_inputs: torch.Tensor,
    text_inputs: torch.Tensor,
) -> torch.Tensor:
    count = frame_inputs.shape[0]
    return student(
        frame_inputs.unsqueeze(0),
        torch.ones((1, count), dtype=torch.bool, device=frame_inputs.device),
        text_inputs.unsqueeze(0),
        torch.ones((1, text_inputs.shape[0]), dtype=torch.bool, device=frame_inputs.device),
        quality_inputs=quality_inputs.unsqueeze(0),
    )["predicted_ecr"]


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    video_path = Path(args.video)
    models = build_models(checkpoint_dir, device)
    evqa, distortion, semantic, clip_model, clip_preprocess, yamnet, classes, student = models
    avg = nn.AdaptiveAvgPool2d((1, 1))

    stage_names = ("decode", "frame_fusion", "clip", "sound", "text", "student")
    recorded = {name: [] for name in stage_names}
    total_values: list[float] = []
    prediction = 0.0

    for iteration in range(args.warmup + args.repeats):
        stage: dict[str, float] = {}
        sync(device)
        total_start = time.perf_counter_ns()

        start = time.perf_counter_ns()
        raw_frames, bgr_frames = load_frames(video_path)
        official_frames = prepare_official_frames(raw_frames, device)
        sync(device)
        stage["decode"] = (time.perf_counter_ns() - start) / 1_000_000

        start = time.perf_counter_ns()
        feat2 = distortion_features(distortion, official_frames, args.frame_batch, avg)
        feat1 = semantic_features(semantic, official_frames, args.frame_batch, avg)
        fused = frame_fusion(evqa, feat1, feat2)
        sync(device)
        stage["frame_fusion"] = (time.perf_counter_ns() - start) / 1_000_000

        start = time.perf_counter_ns()
        quality = clip_features(clip_model, clip_preprocess, bgr_frames, fused.shape[0], device)
        sync(device)
        stage["clip"] = (time.perf_counter_ns() - start) / 1_000_000

        start = time.perf_counter_ns()
        sound = sound_labels(video_path, yamnet, classes)
        stage["sound"] = (time.perf_counter_ns() - start) / 1_000_000

        start = time.perf_counter_ns()
        texts = text_features(evqa, [sound, args.title, args.description], device)
        sync(device)
        stage["text"] = (time.perf_counter_ns() - start) / 1_000_000

        start = time.perf_counter_ns()
        output = student_forward(student, fused, quality, texts)
        sync(device)
        stage["student"] = (time.perf_counter_ns() - start) / 1_000_000
        prediction = float(output.item())
        total_ms = (time.perf_counter_ns() - total_start) / 1_000_000

        if iteration >= args.warmup:
            for name in stage_names:
                recorded[name].append(stage[name])
            total_values.append(total_ms)
        print(
            f"iteration={iteration + 1}/{args.warmup + args.repeats} "
            f"total={total_ms:.2f}ms clips={fused.shape[0]}",
            flush=True,
        )

    result = {
        "scope": "raw video + metadata -> selected teacher frontend -> Full KD student -> ECR",
        "includes": [
            "video decode and official frame preprocessing",
            "EfficientNetV2 + distortion encoder + teacher frame-fusion projection",
            "CLIP ViT-B/32 keyframe features",
            "YAMNet sound labels",
            "Stable Diffusion CLIP text encoder and mean-token pooling",
            "Full KD student forward",
        ],
        "excludes": [
            "model/checkpoint loading",
            "mPLUG captioning",
            "R3D motion encoder",
            "teacher multimodal fusion, temporal transformer, and ECR head",
        ],
        "video": str(video_path.resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "total": summarize(total_values),
        "stages": {name: summarize(values) for name, values in recorded.items()},
        "prediction_random_latency_model": prediction,
        "note": "Student weights are randomly initialized; architecture matches the retained Full KD run. Weight values do not affect latency.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

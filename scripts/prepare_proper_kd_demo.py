#!/usr/bin/env python3
"""Download/cache the models needed by the Proper KD raw-video demo."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json"
DEFAULT_CHECKPOINT = DEFAULT_REPORT.parent / "student_kd_best.pth"
DEFAULT_DOWNLOAD_CHECKPOINT = Path.home() / "Downloads/student_kd_best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        default=os.environ.get("SNAPUGC_REPORT_JSON", str(DEFAULT_REPORT)),
        help="Proper KD report JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SNAPUGC_STUDENT_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        help="Proper KD student checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-source",
        default=str(DEFAULT_DOWNLOAD_CHECKPOINT),
        help="Local source copied into --checkpoint if --checkpoint is missing.",
    )
    parser.add_argument(
        "--text-encoder-model",
        default=os.environ.get("SNAPUGC_TEXT_ENCODER_MODEL", "CompVis/stable-diffusion-v1-4"),
        help="Stable-Diffusion-compatible text encoder repo.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=os.environ.get("HF_HOME"),
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="Only cache visual encoders. Useful when HF access is temporarily unavailable.",
    )
    args = parser.parse_args()

    report_path = Path(args.report_json).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint_source = Path(args.checkpoint_source).expanduser()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = str(Path(args.hf_cache_dir).expanduser())

    ensure_report(report_path)
    ensure_checkpoint(checkpoint_path, checkpoint_source)
    cache_visual_encoders()
    if not args.skip_text_encoder:
        cache_text_encoder(args.text_encoder_model, args.hf_cache_dir)

    print("PROPER_KD_DEMO_READY=1")
    print("SNAPUGC_STUDENT_INPUT_PRESET=clip_mobilenet_text")
    print(f"SNAPUGC_REPORT_JSON={report_path}")
    print(f"SNAPUGC_STUDENT_CHECKPOINT={checkpoint_path}")
    print(f"SNAPUGC_TEXT_ENCODER_MODEL={args.text_encoder_model}")


def ensure_report(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing Proper KD report JSON: {path}")
    print(f"Report OK: {path}")


def ensure_checkpoint(path: Path, source: Path) -> None:
    if path.exists():
        print(f"Student checkpoint OK: {path}")
        return
    if not source.exists():
        raise FileNotFoundError(
            f"Missing student checkpoint: {path}. Also could not find local source: {source}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    print(f"Copied student checkpoint: {source} -> {path}")


def cache_visual_encoders() -> None:
    try:
        import open_clip
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise SystemExit(
            "Missing Proper KD dependency. Run: python -m pip install -r requirements.txt"
        ) from exc

    print("Caching OpenCLIP ViT-B/32 openai weights...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device="cpu",
    )
    model.eval()
    print("Caching MobileNetV3-Small ImageNet weights...")
    mobile = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mobile.eval()


def cache_text_encoder(model_id: str, cache_dir: str | None) -> None:
    from transformers import CLIPTextModel, CLIPTokenizer

    cache = str(Path(cache_dir).expanduser()) if cache_dir else None
    print(f"Caching text tokenizer: {model_id}")
    CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache)
    print(f"Caching text encoder: {model_id}")
    encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", cache_dir=cache)
    encoder.eval()


if __name__ == "__main__":
    main()

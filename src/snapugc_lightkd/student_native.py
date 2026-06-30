"""Student-only video inference helpers for new SnapUGC demo videos.

This module intentionally does not call the official teacher. It builds the
compact student's inputs from lightweight native signals: sampled video frames,
ImageNet visual features when available, deterministic text embeddings, and
simple temporal quality/action descriptors that can be verbalized.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


TEXT_DIM = 768
CLIP_DIM = 1024


@dataclass
class NativeClip:
    index: int
    frame_index: int
    start_pct: float
    end_pct: float
    image: Image.Image
    metrics: dict[str, float]


@dataclass
class NativeInputs:
    clip_inputs: torch.Tensor
    clip_mask: torch.Tensor
    text_inputs: torch.Tensor
    text_mask: torch.Tensor
    teacher_attention: torch.Tensor | None
    clips: list[NativeClip]
    text_streams: list[str]
    metadata: dict[str, Any]
    heuristic_score: float

    def as_batch(self) -> dict[str, object]:
        batch: dict[str, object] = {
            "clip_inputs": self.clip_inputs.unsqueeze(0),
            "clip_mask": self.clip_mask.unsqueeze(0),
            "text_inputs": self.text_inputs.unsqueeze(0),
            "text_mask": self.text_mask.unsqueeze(0),
        }
        if self.teacher_attention is not None:
            batch["teacher_attention"] = self.teacher_attention.unsqueeze(0)
        return batch


def build_native_student_inputs(
    video_path: str | Path,
    *,
    title: str | None = None,
    description: str | None = None,
    max_clips: int = 16,
    clip_dim: int = CLIP_DIM,
    text_dim: int = TEXT_DIM,
    device: torch.device | str = "cpu",
    efficientnet_weights: str | Path | None = None,
    no_visual_encoder: bool = False,
    input_preset: str = "clip_mobilenet_text",
    text_encoder_model: str = "CompVis/stable-diffusion-v1-4",
) -> NativeInputs:
    """Extract teacher-free student inputs from a raw video and metadata."""

    frames, frame_indices, total_frames = sample_video_frames(video_path, max_clips=max_clips)
    if not frames:
        raise RuntimeError(f"Could not decode any frame from {video_path}")

    metrics = compute_clip_metrics(frames)
    if input_preset == "clip_mobilenet_text":
        if no_visual_encoder:
            raise ValueError("clip_mobilenet_text requires the CLIP and MobileNet visual encoders")
        visual_features = clip_mobilenet_feature_matrix(frames, device=device)
        if visual_features.shape[-1] != clip_dim:
            raise ValueError(
                f"Proper KD extractor produced {visual_features.shape[-1]} features, "
                f"but checkpoint expects clip_input_dim={clip_dim}"
            )
        extractor = "open_clip_vit_b32_plus_mobilenet_v3_small"
    else:
        visual_features = low_level_feature_matrix(metrics, clip_dim=clip_dim)
        extractor = "low_level_metrics"
    if input_preset != "clip_mobilenet_text" and not no_visual_encoder:
        try:
            encoded = efficientnet_feature_matrix(
                frames,
                clip_dim=clip_dim,
                device=device,
                weights_path=efficientnet_weights,
            )
            visual_features = 0.82 * encoded + 0.18 * visual_features
            extractor = "efficientnet_v2_s_plus_low_level_metrics"
        except Exception:
            # The demo must remain usable offline; low-level features are enough
            # to keep prediction/explanation operational if weights are missing.
            pass

    n = len(frames)
    clips = []
    for i, (frame, frame_idx, row) in enumerate(zip(frames, frame_indices, metrics)):
        clips.append(
            NativeClip(
                index=i,
                frame_index=int(frame_idx),
                start_pct=float(i / max(n, 1)),
                end_pct=float((i + 1) / max(n, 1)),
                image=frame,
                metrics={key: float(value) for key, value in row.items()},
            )
        )

    clean_title = clean_text(title)
    clean_description = clean_text(description)
    if input_preset == "clip_mobilenet_text":
        # Training uses the first three pooled teacher text streams in this
        # exact order: YAMNet sound labels, title, description. Raw-video demo
        # currently has no lightweight audio labeler, so the sound string is
        # intentionally empty while title/description use the same SD v1.4
        # CLIP text encoder and mean-token pooling as artifact export.
        text_streams = ["sound", "title", "description"]
        text_inputs = stable_diffusion_text_embeddings(
            ["", clean_title or "", clean_description or ""],
            device=device,
            model_id=text_encoder_model,
        )
        if text_inputs.shape[-1] != text_dim:
            raise ValueError(
                f"Text encoder produced {text_inputs.shape[-1]} features, "
                f"but checkpoint expects text_input_dim={text_dim}"
            )
        text_extractor = f"{text_encoder_model}:mean_token_pool"
    else:
        text_rows: list[np.ndarray] = []
        text_streams: list[str] = []
        if clean_title:
            text_rows.append(hash_text_embedding(clean_title, dim=text_dim))
            text_streams.append("title")
        if clean_description:
            text_rows.append(hash_text_embedding(clean_description, dim=text_dim))
            text_streams.append("description")
        if not text_rows:
            text_rows.append(np.zeros((text_dim,), dtype=np.float32))
            text_streams.append("empty_metadata")
        text_inputs = np.stack(text_rows).astype(np.float32, copy=False)
        text_extractor = "deterministic_hash"
    heuristic = heuristic_engagement_score(metrics, clean_title, clean_description)
    return NativeInputs(
        clip_inputs=torch.from_numpy(visual_features.astype(np.float32, copy=False)),
        clip_mask=torch.ones((n,), dtype=torch.bool),
        text_inputs=torch.from_numpy(text_inputs),
        text_mask=torch.ones((len(text_inputs),), dtype=torch.bool),
        teacher_attention=None,
        clips=clips,
        text_streams=text_streams,
        metadata={
            "title": clean_title,
            "description": clean_description,
            "video_path": str(Path(video_path)),
            "total_frames": int(total_frames),
            "visual_extractor": extractor,
            "text_extractor": text_extractor,
            "sound_stream": "empty_no_audio_labeler",
        },
        heuristic_score=float(heuristic),
    )


def sample_video_frames(video_path: str | Path, *, max_clips: int = 16) -> tuple[list[Image.Image], list[int], int]:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(path), ctx=cpu(0))
        total = len(reader)
        if total <= 0:
            return [], [], 0
        indices = _even_indices(total, max_clips)
        batch = reader.get_batch(indices).asnumpy()
        frames = [Image.fromarray(arr).convert("RGB") for arr in batch]
        return frames, indices, total
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            total_raw = reader.count_frames()
        except Exception:
            total_raw = reader.get_length()
        total = int(total_raw) if math.isfinite(float(total_raw)) and total_raw > 0 else 0
        if total:
            indices = _even_indices(total, max_clips)
            frames = [Image.fromarray(reader.get_data(idx)).convert("RGB") for idx in indices]
            reader.close()
            return frames, indices, total

        collected = []
        for idx, arr in enumerate(reader):
            collected.append((idx, Image.fromarray(arr).convert("RGB")))
        reader.close()
        if not collected:
            return [], [], 0
        take = _even_indices(len(collected), max_clips)
        frames = [collected[idx][1] for idx in take]
        selected_indices = [collected[idx][0] for idx in take]
        return frames, selected_indices, len(collected)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode video {path}: {exc}") from exc


def compute_clip_metrics(frames: list[Image.Image]) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    prev_gray: np.ndarray | None = None
    for image in frames:
        arr = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        gy, gx = np.gradient(gray)
        edge_energy = float(np.sqrt(gx * gx + gy * gy).mean())
        lap = np.gradient(gx)[1] + np.gradient(gy)[0]
        sharpness = float(np.var(lap))
        saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
        colorfulness = float(arr.std(axis=(0, 1)).mean())
        motion = 0.0 if prev_gray is None else float(np.abs(gray - prev_gray).mean())
        prev_gray = gray
        metrics.append(
            {
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "edge_energy": edge_energy,
                "saturation": saturation,
                "colorfulness": colorfulness,
                "motion": motion,
            }
        )
    if len(metrics) > 1:
        metrics[0]["motion"] = float(np.median([row["motion"] for row in metrics[1:]]))
    return metrics


def low_level_feature_matrix(metrics: list[dict[str, float]], *, clip_dim: int = CLIP_DIM) -> np.ndarray:
    rows = []
    keys = [
        "brightness",
        "contrast",
        "sharpness",
        "edge_energy",
        "saturation",
        "colorfulness",
        "motion",
    ]
    for idx, row in enumerate(metrics):
        base = np.array([row[key] for key in keys], dtype=np.float32)
        time = np.array(
            [
                idx / max(len(metrics) - 1, 1),
                math.sin(idx + 1.0),
                math.cos(idx + 1.0),
            ],
            dtype=np.float32,
        )
        seed = np.concatenate([base, time])
        tiled = np.resize(seed, clip_dim).astype(np.float32, copy=False)
        tiled = (tiled - tiled.mean()) / (tiled.std() + 1e-6)
        rows.append(tiled)
    return np.stack(rows)


@torch.no_grad()
def efficientnet_feature_matrix(
    frames: list[Image.Image],
    *,
    clip_dim: int = CLIP_DIM,
    device: torch.device | str = "cpu",
    weights_path: str | Path | None = None,
) -> np.ndarray:
    from torchvision import models, transforms

    weights = None
    if weights_path is None:
        try:
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    model = models.efficientnet_v2_s(weights=weights)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch = torch.stack([preprocess(frame) for frame in frames]).to(device)
    feats = model(batch)
    feats = F.layer_norm(feats, feats.shape[-1:])
    out = torch.zeros((feats.shape[0], clip_dim), device=feats.device)
    n = min(clip_dim, feats.shape[-1])
    out[:, :n] = feats[:, :n]
    if n < clip_dim:
        out[:, n:] = feats.mean(dim=-1, keepdim=True)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


@lru_cache(maxsize=4)
def _load_proper_visual_encoders(device_name: str):
    import open_clip
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    device = torch.device(device_name)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()
    mobile_weights = MobileNet_V3_Small_Weights.DEFAULT
    mobile_model = mobilenet_v3_small(weights=mobile_weights).features.to(device).eval()
    return clip_model, clip_preprocess, mobile_model, mobile_weights.transforms()


@torch.no_grad()
def clip_mobilenet_feature_matrix(
    frames: list[Image.Image],
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Reproduce the Proper KD CLIP(512)+MobileNet action(1152) input."""

    device = torch.device(device)
    clip_model, clip_preprocess, mobile_model, mobile_preprocess = _load_proper_visual_encoders(
        str(device)
    )
    clip_batch = torch.stack([clip_preprocess(frame) for frame in frames]).to(device)
    clip_features = F.normalize(clip_model.encode_image(clip_batch), dim=-1).float()

    mobile_batch = torch.stack([mobile_preprocess(frame) for frame in frames]).to(device)
    spatial = mobile_model(mobile_batch).mean(dim=(2, 3)).float()
    motion = torch.zeros_like(spatial)
    motion[1:] = spatial[1:] - spatial[:-1]
    lite_action = torch.cat([spatial, motion], dim=-1)
    return (
        torch.cat([clip_features, lite_action], dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


@lru_cache(maxsize=4)
def _load_sd_text_encoder(model_id: str, device_name: str):
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    encoder.eval().to(torch.device(device_name))
    return tokenizer, encoder


@torch.no_grad()
def stable_diffusion_text_embeddings(
    texts: list[str],
    *,
    device: torch.device | str = "cpu",
    model_id: str = "CompVis/stable-diffusion-v1-4",
) -> np.ndarray:
    """Match teacher artifact text pooling: CLIP tokens then mean over 77 tokens."""

    device = torch.device(device)
    tokenizer, encoder = _load_sd_text_encoder(model_id, str(device))
    tokens = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    hidden = encoder(tokens.input_ids.to(device))[0]
    return hidden.mean(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)


def hash_text_embedding(text: str, *, dim: int = TEXT_DIM) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    tokens = re.findall(r"[\w#@]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else vec


def heuristic_engagement_score(
    metrics: list[dict[str, float]],
    title: str | None,
    description: str | None,
) -> float:
    if not metrics:
        return 0.5
    rows = {key: np.array([m[key] for m in metrics], dtype=np.float32) for key in metrics[0]}
    brightness = _bell(rows["brightness"].mean(), center=0.52, width=0.24)
    contrast = _clip01((rows["contrast"].mean() - 0.08) / 0.22)
    motion = _clip01(rows["motion"].mean() / 0.18)
    sharpness = _clip01(rows["sharpness"].mean() / 0.004)
    early_motion = _clip01(rows["motion"][: max(1, len(metrics) // 4)].mean() / 0.18)
    text_score = 0.0
    joined = " ".join(x for x in [title, description] if x)
    if joined:
        words = re.findall(r"\w+", joined)
        hashtags = re.findall(r"#?\b\w+\b", joined)
        text_score = min(1.0, len(set(words)) / 18.0)
        if len(hashtags) > 18:
            text_score *= 0.82
    score = (
        0.22 * brightness
        + 0.20 * contrast
        + 0.22 * motion
        + 0.14 * early_motion
        + 0.12 * sharpness
        + 0.10 * text_score
    )
    return float(_clip01(0.18 + 0.72 * score))


def semantic_clip_label(metrics: dict[str, float]) -> str:
    profile = semantic_clip_profile(metrics)
    return profile["label_vi"]


def semantic_clip_profile(metrics: dict[str, float]) -> dict[str, Any]:
    """Human-readable semantic labels derived from native clip metrics."""

    bits = []
    bits_en = []
    attributes = []

    motion = float(metrics.get("motion", 0.0))
    if motion >= 0.08:
        bits.append("chuyển động rõ")
        bits_en.append("clear motion")
        attributes.append(_semantic_attr("motion", "strong", "mạnh", motion, "Frame-to-frame change is high."))
    elif motion <= 0.025:
        bits.append("nhịp hình khá tĩnh")
        bits_en.append("mostly static pacing")
        attributes.append(_semantic_attr("motion", "weak", "yếu", motion, "Frame-to-frame change is low."))
    else:
        attributes.append(_semantic_attr("motion", "moderate", "vừa", motion, "Frame-to-frame change is moderate."))

    brightness = float(metrics.get("brightness", 0.5))
    if brightness < 0.32:
        bits.append("khung hình hơi tối")
        bits_en.append("slightly dark frame")
        attributes.append(_semantic_attr("lighting", "weak", "yếu", brightness, "Average brightness is below the comfortable range."))
    elif brightness > 0.72:
        bits.append("khung hình sáng")
        bits_en.append("bright frame")
        attributes.append(_semantic_attr("lighting", "strong", "mạnh", brightness, "Average brightness is high."))
    else:
        attributes.append(_semantic_attr("lighting", "balanced", "cân bằng", brightness, "Brightness sits in a readable range."))

    contrast = float(metrics.get("contrast", 0.0))
    edge_energy = float(metrics.get("edge_energy", 0.0))
    if contrast >= 0.18 or edge_energy >= 0.055:
        bits.append("chi tiết thị giác nổi bật")
        bits_en.append("salient visual detail")
        attributes.append(_semantic_attr("visual_detail", "strong", "mạnh", max(contrast, edge_energy), "Contrast or edge energy is high."))
    else:
        attributes.append(_semantic_attr("visual_detail", "moderate", "vừa", max(contrast, edge_energy), "Contrast and edge energy are not dominant."))

    sharpness = float(metrics.get("sharpness", 0.0))
    if sharpness <= 0.0008:
        bits.append("độ nét thấp")
        bits_en.append("low sharpness")
        attributes.append(_semantic_attr("sharpness", "weak", "yếu", sharpness, "Laplacian sharpness is low."))
    else:
        attributes.append(_semantic_attr("sharpness", "usable", "ổn", sharpness, "Sharpness is usable for sampled-frame analysis."))

    colorfulness = float(metrics.get("colorfulness", 0.0))
    saturation = float(metrics.get("saturation", 0.0))
    if colorfulness >= 0.18 or saturation >= 0.28:
        bits.append("màu sắc nổi bật")
        bits_en.append("vivid color")
        attributes.append(_semantic_attr("color", "strong", "mạnh", max(colorfulness, saturation), "Colorfulness or saturation is high."))
    else:
        attributes.append(_semantic_attr("color", "neutral", "trung tính", max(colorfulness, saturation), "Color signal is neutral."))

    return {
        "label_vi": ", ".join(bits) if bits else "tín hiệu thị giác trung tính",
        "label_en": ", ".join(bits_en) if bits_en else "neutral visual signal",
        "attributes": attributes,
    }


def build_recommendations(
    *,
    score: float,
    title: str | None,
    description: str | None,
    clip_rows: list[dict[str, Any]],
    clip_metrics: list[dict[str, float]],
) -> list[str]:
    recs: list[str] = []
    if clip_rows:
        early = [
            row
            for row in clip_rows
            if float(row["relative_time"]["end_pct"]) <= 0.25
        ]
        later = [
            row
            for row in clip_rows
            if float(row["relative_time"]["start_pct"]) >= 0.5
        ]
        early_gain = max([float(r["contribution_to_score"]) for r in early] or [0.0])
        later_gain = max([float(r["contribution_to_score"]) for r in later] or [0.0])
        if later_gain > early_gain + 0.015:
            recs.append(
                "Hook đầu video chưa mạnh bằng các đoạn sau; nên đưa hành động/chủ thể hấp dẫn lên 0-3 giây đầu."
            )
    if title:
        words = re.findall(r"\w+", title)
        hashtag_like = sum(1 for w in words if w.lower() in {"fitness", "fyp", "viral"} or len(w) <= 2)
        if len(words) > 14 or hashtag_like >= max(5, len(words) // 2):
            recs.append(
                "Title đang thiên về chuỗi hashtag/từ khóa; nên thêm một câu mô tả cụ thể điều người xem sẽ thấy."
            )
    else:
        recs.append("Nên thêm title ngắn, cụ thể, khớp trực tiếp với hành động chính trong video.")
    if not description:
        recs.append("Description đang trống; có thể thêm 1 câu ngữ cảnh để tăng tín hiệu text-video alignment.")
    if clip_metrics:
        avg_brightness = float(np.mean([m["brightness"] for m in clip_metrics]))
        avg_sharpness = float(np.mean([m["sharpness"] for m in clip_metrics]))
        avg_motion = float(np.mean([m["motion"] for m in clip_metrics]))
        if avg_brightness < 0.32:
            recs.append("Video hơi tối; tăng sáng hoặc chọn thumbnail/key moment sáng hơn sẽ dễ giữ chú ý hơn.")
        if avg_sharpness < 0.0008:
            recs.append("Một số đoạn thiếu nét; nên ưu tiên khung hình rõ chủ thể khi cắt dựng.")
        if avg_motion < 0.025:
            recs.append("Nhịp hình khá tĩnh; thêm chuyển động, cut hoặc hành động rõ hơn có thể cải thiện sức hút.")
    if not recs:
        if score >= 0.67:
            recs.append("Tín hiệu hiện khá tốt; nên giữ hook/chủ thể chính và thử A/B title cụ thể hơn.")
        else:
            recs.append("Nên tăng độ rõ của hook, làm title cụ thể hơn, và rút ngắn các đoạn đóng góp yếu.")
    return recs[:4]


def build_semantic_attributes(
    *,
    title: str | None,
    description: str | None,
    clip_rows: list[dict[str, Any]],
    clip_metrics: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """Interpretable semantic attributes derived from student-native evidence."""

    if not clip_metrics:
        return []
    motion = np.array([m["motion"] for m in clip_metrics], dtype=np.float32)
    brightness = np.array([m["brightness"] for m in clip_metrics], dtype=np.float32)
    contrast = np.array([m["contrast"] for m in clip_metrics], dtype=np.float32)
    sharpness = np.array([m["sharpness"] for m in clip_metrics], dtype=np.float32)
    early_rows = [
        row
        for row in clip_rows
        if float(row["relative_time"]["end_pct"]) <= 0.25
    ]
    early_delta = max([float(row["contribution_to_score"]) for row in early_rows] or [0.0])
    all_positive = [max(0.0, float(row["contribution_to_score"])) for row in clip_rows]
    max_delta = max(all_positive or [1e-6])
    hook = _clip01(early_delta / max(max_delta, 1e-6))
    motion_score = _clip01(float(motion.mean() / 0.18))
    clarity = _clip01(float((contrast.mean() / 0.22) * 0.55 + (sharpness.mean() / 0.006) * 0.45))
    lighting = _bell(float(brightness.mean()), center=0.52, width=0.26)
    pacing = _clip01(float(motion.std() / 0.08 + contrast.std() / 0.08) / 2.0)
    text = " ".join(x for x in [title, description] if x)
    words = re.findall(r"\w+", text.lower())
    unique_ratio = len(set(words)) / max(len(words), 1)
    specificity = _clip01((len(set(words)) / 16.0) * 0.65 + unique_ratio * 0.35) if words else 0.0

    return [
        _concept(
            "hook_strength",
            hook,
            "Độ mạnh của phần đầu video dựa trên contribution trong 25% thời lượng đầu.",
            "high_supports_engagement",
        ),
        _concept(
            "motion_action",
            motion_score,
            "Mức chuyển động/hành động ước lượng từ thay đổi giữa các frame được sample.",
            "high_supports_engagement",
        ),
        _concept(
            "visual_clarity",
            clarity,
            "Độ rõ thị giác dựa trên contrast và sharpness native.",
            "high_supports_engagement",
        ),
        _concept(
            "lighting_quality",
            lighting,
            "Mức sáng có nằm trong vùng dễ xem hay không.",
            "high_supports_engagement",
        ),
        _concept(
            "text_specificity",
            specificity,
            "Title/description có đủ cụ thể hay chỉ là chuỗi từ khóa chung chung.",
            "high_supports_engagement",
        ),
        _concept(
            "pacing_variety",
            pacing,
            "Độ thay đổi nhịp hình giữa các đoạn được sample.",
            "medium_is_ok",
        ),
    ]


def build_native_concept_bottleneck(
    *,
    title: str | None,
    description: str | None,
    clip_rows: list[dict[str, Any]],
    clip_metrics: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """Backward-compatible alias for older demo UI fields.

    These are no longer treated as a trained bottleneck. They are semantic
    attributes used by the language explanation layer.
    """

    return build_semantic_attributes(
        title=title,
        description=description,
        clip_rows=clip_rows,
        clip_metrics=clip_metrics,
    )


def save_top_clip_thumbnails(
    clips: list[NativeClip],
    top_clip_rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    prefix: str = "top_clip",
) -> list[dict[str, str]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_idx = {clip.index: clip for clip in clips}
    assets = []
    for rank, row in enumerate(top_clip_rows, start=1):
        clip = by_idx.get(int(row["clip_index"]))
        if clip is None:
            continue
        path = out_dir / f"{prefix}_{rank:02d}_clip_{clip.index:02d}.jpg"
        clip.image.save(path, quality=90)
        assets.append(
            {
                "rank": str(rank),
                "clip_index": str(clip.index),
                "path": str(path),
            }
        )
    return assets


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _even_indices(total: int, max_clips: int) -> list[int]:
    count = min(max_clips, max(1, total))
    if total <= count:
        return list(range(total))
    # Match the CLIP/Lite Action training extractors: sample the midpoint of
    # every uniform temporal bin, rather than including the first/last frames.
    step = total / count
    return [int(step * idx + step / 2) for idx in range(count)]


def _clip01(value: float | np.ndarray) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _bell(value: float, *, center: float, width: float) -> float:
    return _clip01(1.0 - abs(float(value) - center) / max(width, 1e-6))


def _concept(name: str, score: float, rationale: str, direction: str) -> dict[str, Any]:
    score = _clip01(score)
    if score >= 0.67:
        label = "strong"
        label_vi = "mạnh"
    elif score >= 0.34:
        label = "moderate"
        label_vi = "vừa"
    else:
        label = "weak"
        label_vi = "yếu"
    return {
        "name": name,
        "score": score,
        "label": label,
        "label_vi": label_vi,
        "rationale": rationale,
        "direction": direction,
    }


def _semantic_attr(name: str, label: str, label_vi: str, value: float, rationale: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "label_vi": label_vi,
        "value": float(value),
        "rationale": rationale,
    }

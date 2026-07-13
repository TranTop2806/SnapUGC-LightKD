"""NLA-inspired post-hoc explanations for SnapUGC student predictions."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def selected_text_streams(input_config: object) -> list[str]:
    order = [
        ("sound", bool(getattr(input_config, "use_sound_text", False))),
        ("title", bool(getattr(input_config, "use_title_text", False))),
        ("description", bool(getattr(input_config, "use_description_text", False))),
        ("caption", bool(getattr(input_config, "use_caption_text", False))),
    ]
    return [name for name, enabled in order if enabled]


def clean_text(value: object, *, max_chars: int = 240) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "..."
    return text


def load_metadata(csv_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if csv_path is None:
        return {}
    path = Path(csv_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Id" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        video_id = str(row.get("Id"))
        out[video_id] = {
            "title": clean_text(row.get("Title")),
            "description": clean_text(row.get("Description")),
            "ecr_true": _safe_float(row.get("ECR")),
            "split": clean_text(row.get("split"), max_chars=64),
        }
    return out


def load_captions(caption_dir: str | Path | None) -> dict[str, str]:
    if caption_dir is None:
        return {}
    root = Path(caption_dir)
    if not root.exists():
        return {}
    captions: dict[str, str] = {}
    for path in sorted(root.glob("*_captions.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                video_id = str(row.get("Id", ""))
                caption = clean_text(row.get("caption"), max_chars=320)
                if video_id and caption:
                    captions[video_id] = caption
    return captions


def engagement_band(
    score: float,
    reference_values: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    if reference_values is not None and len(reference_values) >= 10:
        values = np.asarray(reference_values, dtype=np.float64)
        lo, hi = np.quantile(values, [0.33, 0.67])
        if score < lo:
            label = "low"
            vi_label = "thấp"
        elif score > hi:
            label = "high"
            vi_label = "cao"
        else:
            label = "medium"
            vi_label = "trung bình"
        percentile = float((values <= score).mean())
        return {
            "label": label,
            "label_vi": vi_label,
            "percentile": percentile,
            "thresholds": {"low_medium_q33": float(lo), "medium_high_q67": float(hi)},
            "basis": "empirical_label_distribution",
        }

    if score < 0.33:
        label, vi_label = "low", "thấp"
    elif score > 0.67:
        label, vi_label = "high", "cao"
    else:
        label, vi_label = "medium", "trung bình"
    return {
        "label": label,
        "label_vi": vi_label,
        "percentile": None,
        "thresholds": {"low_medium": 0.33, "medium_high": 0.67},
        "basis": "fixed_score_thresholds",
    }


@torch.no_grad()
def run_student(model: torch.nn.Module, batch: dict[str, object]) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    for batch_key, model_key in (("dover_inputs", "dover_inputs"), ("quality_inputs", "quality_inputs")):
        value = batch.get(batch_key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            kwargs[model_key] = value
    return model(
        batch["clip_inputs"],
        batch["clip_mask"],
        batch["text_inputs"],
        batch["text_mask"],
        **kwargs,
    )


@torch.no_grad()
def predict_with_zeroed_evidence(
    model: torch.nn.Module,
    batch: dict[str, object],
    *,
    zero_clip_indices: set[int] | None = None,
    zero_text_indices: set[int] | None = None,
) -> float:
    b = _clone_tensor_batch(batch)
    if zero_clip_indices:
        for idx in zero_clip_indices:
            if 0 <= idx < b["clip_inputs"].shape[1]:
                b["clip_inputs"][:, idx, :] = 0.0
    if zero_text_indices and b["text_inputs"].shape[1] > 0:
        for idx in zero_text_indices:
            if 0 <= idx < b["text_inputs"].shape[1]:
                b["text_inputs"][:, idx, :] = 0.0
    out = run_student(model, b)
    return float(out["predicted_ecr"][0].detach().cpu().item())


def explain_student_prediction(
    *,
    model: torch.nn.Module,
    batch: dict[str, object],
    outputs: dict[str, torch.Tensor],
    input_config: object,
    video_id: str,
    metadata: dict[str, Any] | None = None,
    caption: str | None = None,
    reference_ecr_values: list[float] | np.ndarray | None = None,
    teacher_ecr: float | None = None,
    topk: int = 3,
) -> dict[str, Any]:
    """Build an explanation using language claims plus ablation validation.

    The design mirrors the NLA workflow at inference time: a compact verbalizer
    maps model activations to natural-language claims, then a lightweight
    reconstructor/validator measures whether the selected claims preserve or
    change the prediction under input ablations.
    """

    student_ecr = float(outputs["predicted_ecr"][0].detach().cpu().item())
    clip_mask = batch["clip_mask"][0].detach().cpu().numpy().astype(bool)
    n_clips = int(clip_mask.sum())
    clip_indices = list(range(n_clips))

    temporal_attention = outputs["temporal_attention"][0].detach().cpu().numpy()[clip_mask]
    temporal_attention = _renorm(temporal_attention)
    clip_ecr = outputs["clip_ecr"][0].detach().cpu().numpy()[clip_mask]
    if "teacher_attention" in batch and isinstance(batch["teacher_attention"], torch.Tensor):
        teacher_attention = batch["teacher_attention"][0].detach().cpu().numpy()[clip_mask]
        teacher_attention = _renorm(teacher_attention)
    else:
        teacher_attention = np.zeros((0,), dtype=np.float64)

    clip_rows: list[dict[str, Any]] = []
    for idx in clip_indices:
        ablated = predict_with_zeroed_evidence(model, batch, zero_clip_indices={idx})
        contribution = student_ecr - ablated
        start_pct = idx / max(n_clips, 1)
        end_pct = (idx + 1) / max(n_clips, 1)
        clip_rows.append(
            {
                "clip_index": idx,
                "relative_time": {
                    "start_pct": float(start_pct),
                    "end_pct": float(end_pct),
                    "label": f"{start_pct * 100:.0f}-{end_pct * 100:.0f}% video",
                },
                "student_attention": float(temporal_attention[idx]),
                "teacher_attention": float(teacher_attention[idx]) if len(teacher_attention) else None,
                "student_clip_ecr": float(clip_ecr[idx]),
                "ablated_ecr_zero_clip": float(ablated),
                "contribution_to_score": float(contribution),
                "direction": _direction_label(contribution),
            }
        )

    clip_rows.sort(
        key=lambda row: (
            abs(float(row["contribution_to_score"])) * 0.65
            + float(row["student_attention"]) * 0.35
        ),
        reverse=True,
    )
    top_clips = _select_diverse_clip_rows(clip_rows, topk=topk, n_clips=n_clips)

    stream_names = selected_text_streams(input_config)
    text_attention = outputs["text_attention"][0].detach().cpu().numpy()
    text_attention = _renorm(text_attention)
    text_rows: list[dict[str, Any]] = []
    metadata = metadata or {}
    text_sources = {
        "title": metadata.get("title"),
        "description": metadata.get("description"),
        "caption": caption,
        "sound": metadata.get("sound"),
    }
    for idx, weight in enumerate(text_attention):
        stream = stream_names[idx] if idx < len(stream_names) else f"text_{idx}"
        source_text = clean_text(text_sources.get(stream))
        if source_text is None:
            continue
        ablated = predict_with_zeroed_evidence(model, batch, zero_text_indices={idx})
        contribution = student_ecr - ablated
        text_rows.append(
            {
                "stream": stream,
                "attention": float(weight),
                "source_text": source_text,
                "ablated_ecr_zero_text": float(ablated),
                "contribution_to_score": float(contribution),
                "direction": _direction_label(contribution),
            }
        )
    text_rows.sort(
        key=lambda row: (
            abs(float(row["contribution_to_score"])) * 0.65 + float(row["attention"]) * 0.35
        ),
        reverse=True,
    )

    selected_clip_indices = {int(row["clip_index"]) for row in top_clips}
    if text_rows:
        top_stream = str(text_rows[0]["stream"])
        selected_text_indices = (
            {stream_names.index(top_stream)} if top_stream in stream_names else set()
        )
    else:
        selected_text_indices = set()

    all_clip_indices = set(range(n_clips))
    all_text_indices = set(range(len(text_attention)))
    keep_only_ecr = predict_with_zeroed_evidence(
        model,
        batch,
        zero_clip_indices=all_clip_indices - selected_clip_indices,
        zero_text_indices=all_text_indices - selected_text_indices,
    )
    remove_selected_ecr = predict_with_zeroed_evidence(
        model,
        batch,
        zero_clip_indices=selected_clip_indices,
        zero_text_indices=selected_text_indices,
    )

    reconstruction_error = abs(student_ecr - keep_only_ecr)
    necessity_delta = student_ecr - remove_selected_ecr
    selected_vs_remaining_delta = keep_only_ecr - remove_selected_ecr
    attention_overlap = _distribution_overlap(temporal_attention, teacher_attention)

    band = engagement_band(student_ecr, reference_ecr_values)
    score_delta = None if teacher_ecr is None else student_ecr - teacher_ecr
    confidence = explanation_confidence(
        reconstruction_error=reconstruction_error,
        necessity_delta=necessity_delta,
        selected_vs_remaining_delta=selected_vs_remaining_delta,
        attention_overlap=attention_overlap,
        teacher_abs_delta=None if score_delta is None else abs(score_delta),
    )
    claims = build_language_claims(
        student_ecr=student_ecr,
        band=band,
        top_clips=top_clips,
        text_rows=text_rows,
        keep_only_ecr=keep_only_ecr,
        remove_selected_ecr=remove_selected_ecr,
        caption=caption,
    )

    return _json_ready(
        {
            "video_id": str(video_id),
            "scores": {
                "student_ecr": student_ecr,
                "teacher_ecr": teacher_ecr,
                "delta_student_minus_teacher": score_delta,
                "band": band,
            },
            "input_context": {
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "caption": caption,
            },
            "evidence": {
                "top_clips": top_clips,
                "all_clips": clip_rows,
                "text_streams": text_rows,
                "attention_overlap_student_teacher": attention_overlap,
            },
            "nla_style_explanation": {
                "summary": claims[0] if claims else "",
                "claims": claims,
                "natural_language_bottleneck": {
                    "selected_clip_indices": sorted(selected_clip_indices),
                    "selected_text_streams": [
                        stream_names[idx]
                        for idx in sorted(selected_text_indices)
                        if idx < len(stream_names)
                    ],
                    "verbalizer": (
                        "template verbalizer over student activations, attention, "
                        "clip scores, and metadata"
                    ),
                    "reconstructor": (
                        "zero-ablation keep-only reconstruction of the student prediction"
                    ),
                },
                "faithfulness": {
                    "keep_only_selected_ecr": keep_only_ecr,
                    "reconstruction_error_abs": reconstruction_error,
                    "remove_selected_ecr": remove_selected_ecr,
                    "necessity_delta_student_minus_removed": necessity_delta,
                    "selected_minus_remaining_ecr": selected_vs_remaining_delta,
                    "ablation_mode": "zero_input",
                },
                "confidence": confidence,
                "limitations": (
                    "Các claim là giải thích hậu xử lý theo tinh thần NLA, "
                    "không phải bằng chứng nhân quả tuyệt đối. "
                    "Độ tin cậy nên đọc cùng reconstruction_error và ablation deltas."
                ),
            },
        }
    )


def build_language_claims(
    *,
    student_ecr: float,
    band: dict[str, Any],
    top_clips: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    keep_only_ecr: float,
    remove_selected_ecr: float,
    caption: str | None,
) -> list[str]:
    band_text = band.get("label_vi", band.get("label", "unknown"))
    claims = [
        (
            f"Dự đoán ECR={student_ecr:.4f}, thuộc nhóm {band_text}; "
            f"khi chỉ giữ evidence được verbalize, score còn {keep_only_ecr:.4f}, "
            f"và khi bỏ evidence này score thành {remove_selected_ecr:.4f}."
        )
    ]

    if top_clips:
        clip_bits = []
        for row in top_clips:
            delta = float(row["contribution_to_score"])
            verb = "kéo score lên" if delta >= 0 else "kìm score xuống"
            clip_bits.append(
                f"clip {row['clip_index']} ({row['relative_time']['label']}, "
                f"attn={row['student_attention']:.3f}, "
                f"clip_ecr={row['student_clip_ecr']:.3f}) {verb} khoảng {abs(delta):.4f}"
            )
        claims.append("Temporal evidence chính: " + "; ".join(clip_bits) + ".")

    if text_rows:
        row = text_rows[0]
        delta = float(row["contribution_to_score"])
        verb = "hỗ trợ tăng score" if delta >= 0 else "làm giảm/kìm score"
        source = f" Nội dung: {row['source_text']}" if row.get("source_text") else ""
        claims.append(
            f"Text stream nổi bật là {row['stream']} (attn={row['attention']:.3f}), "
            f"{verb} khoảng {abs(delta):.4f}.{source}"
        )

    if caption:
        claims.append(f"Caption teacher tạo ra cung cấp ngữ cảnh thị giác: {caption}")
    return claims


def explanation_confidence(
    *,
    reconstruction_error: float,
    necessity_delta: float,
    attention_overlap: float | None,
    teacher_abs_delta: float | None,
    selected_vs_remaining_delta: float = 0.0,
) -> str:
    overlap = 0.0 if attention_overlap is None else attention_overlap
    teacher_ok = teacher_abs_delta is None or teacher_abs_delta <= 0.08
    evidence_separated = selected_vs_remaining_delta >= 0.015
    if (
        reconstruction_error <= 0.035
        and abs(necessity_delta) >= 0.02
        and evidence_separated
        and overlap >= 0.45
        and teacher_ok
    ):
        return "high"
    if (
        reconstruction_error <= 0.075
        and evidence_separated
        and (abs(necessity_delta) >= 0.01 or overlap >= 0.35)
    ):
        return "medium"
    return "low"


def _clone_tensor_batch(batch: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def _distribution_overlap(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    n = min(len(a), len(b))
    a = _renorm(a[:n])
    b = _renorm(b[:n])
    return float(np.minimum(a, b).sum())


def _renorm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    total = values.sum()
    if total <= 1e-12:
        return np.ones_like(values) / max(1, len(values))
    return values / total


def _direction_label(delta: float) -> str:
    if delta >= 0.01:
        return "supports_higher_ecr"
    if delta <= -0.01:
        return "suppresses_ecr"
    return "weak_or_mixed"


def _select_diverse_clip_rows(
    clip_rows: list[dict[str, Any]],
    *,
    topk: int,
    n_clips: int,
) -> list[dict[str, Any]]:
    """Pick high-scoring clips while avoiding near-duplicate time segments."""

    limit = min(topk, len(clip_rows))
    if limit <= 1:
        return clip_rows[:limit]
    min_gap = max(2, n_clips // max(limit * 2, 1))
    selected: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for row in clip_rows:
        idx = int(row["clip_index"])
        if all(abs(idx - prev) >= min_gap for prev in selected_indices):
            selected.append(row)
            selected_indices.append(idx)
        if len(selected) >= limit:
            return selected
    for row in clip_rows:
        idx = int(row["clip_index"])
        if idx not in selected_indices:
            selected.append(row)
            selected_indices.append(idx)
        if len(selected) >= limit:
            break
    return selected


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

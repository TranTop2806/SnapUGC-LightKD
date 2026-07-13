#!/usr/bin/env python3
"""Student-only prediction and explanation for a completely new video."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.explanations import (  # noqa: E402
    engagement_band,
    explain_student_prediction,
    move_batch,
)
from snapugc_lightkd.llm_explainer import (  # noqa: E402
    build_semantic_llm_input,
    generate_semantic_explanation,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402
from snapugc_lightkd.student_native import (  # noqa: E402
    build_native_student_inputs,
    build_recommendations,
    build_semantic_attributes,
    save_top_clip_thumbnails,
    semantic_clip_label,
    semantic_clip_profile,
)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> bool:
    if not checkpoint.exists():
        return False
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return len(unexpected) == 0 and len(missing) == 0


def load_report(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a new .mp4/.mov video")
    parser.add_argument("--title", default="")
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--report-json",
        default=None,
        help="official_student_kd_report.json. Defaults to common VM result path when present.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="student_kd_best.pth. Defaults to file next to --report-json.",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Optional CSV with ECR values used only for empirical low/medium/high thresholds.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--efficientnet-weights", default=None)
    parser.add_argument("--no-visual-encoder", action="store_true")
    parser.add_argument(
        "--input-preset",
        default=os.environ.get("SNAPUGC_STUDENT_INPUT_PRESET"),
        help="Native input builder preset. Use clip_mobilenet_text for Proper KD.",
    )
    parser.add_argument(
        "--text-encoder-model",
        default=os.environ.get("SNAPUGC_TEXT_ENCODER_MODEL", "CompVis/stable-diffusion-v1-4"),
        help="Stable-Diffusion-compatible CLIP text encoder used by clip_mobilenet_text.",
    )
    parser.add_argument("--explanation-language", default="vi", choices=["vi", "en"])
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Use deterministic template explanation even if an LLM API key is configured.",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--assets-dir", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    report_path = resolve_report_path(args.report_json)
    report = load_report(report_path)
    model_kwargs = dict(
        report.get(
            "model_kwargs",
            {
                "clip_input_dim": 1024,
                "hidden_dim": 96,
                "max_clips": 16,
                "n_layers": 1,
                "n_heads": 4,
                "dropout": 0.22,
            },
        )
    )
    max_clips = int(args.max_clips or model_kwargs.get("max_clips", 16))
    model_kwargs["max_clips"] = max_clips
    native_input_preset = resolve_native_input_preset(
        args.input_preset or report.get("input_preset"),
        int(model_kwargs.get("clip_input_dim", 1024)),
    )

    native = build_native_student_inputs(
        args.video,
        title=args.title,
        description=args.description,
        max_clips=max_clips,
        clip_dim=int(model_kwargs.get("clip_input_dim", 1024)),
        device=device,
        efficientnet_weights=args.efficientnet_weights,
        no_visual_encoder=args.no_visual_encoder,
        input_preset=native_input_preset,
        text_encoder_model=args.text_encoder_model,
    )
    batch = move_batch(native.as_batch(), device)

    model = OfficialArtifactStudent(**model_kwargs).to(device)
    ckpt_path = resolve_checkpoint_path(args.checkpoint, report_path)
    checkpoint_loaded = load_checkpoint(model, ckpt_path, device) if ckpt_path else False
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["clip_inputs"],
            batch["clip_mask"],
            batch["text_inputs"],
            batch["text_mask"],
        )

    raw_student_score = float(outputs["predicted_ecr"][0].detach().cpu().item())
    if native_input_preset == "clip_mobilenet_text":
        student_ecr = raw_student_score
        score_policy = "raw_checkpoint_score"
    else:
        # Native fallback features are intentionally teacher-free and may not
        # match the exact artifact distribution used in KD. Blend with an
        # interpretable heuristic so the fallback demo remains stable while
        # still reporting both values.
        student_ecr = 0.72 * raw_student_score + 0.28 * native.heuristic_score
        score_policy = "raw_checkpoint_score_blended_with_native_heuristic"
    outputs["predicted_ecr"] = torch.tensor([student_ecr], device=device, dtype=torch.float32)

    input_config = make_input_config(native.text_streams)
    reference_values = load_reference_ecr_values(args.labels_csv)
    result = explain_student_prediction(
        model=model,
        batch=batch,
        outputs=outputs,
        input_config=input_config,
        video_id=Path(args.video).stem,
        metadata=native.metadata,
        caption=None,
        reference_ecr_values=reference_values,
        teacher_ecr=None,
        topk=args.topk,
    )

    clip_metrics = [clip.metrics for clip in native.clips]
    clip_semantics = {clip.index: semantic_clip_label(clip.metrics) for clip in native.clips}
    clip_profiles = {clip.index: semantic_clip_profile(clip.metrics) for clip in native.clips}
    for row in result["evidence"]["all_clips"]:
        idx = int(row["clip_index"])
        row["semantic_label"] = clip_semantics.get(idx)
        row["semantic_profile"] = clip_profiles.get(idx)
        if 0 <= idx < len(clip_metrics):
            row["native_visual_metrics"] = clip_metrics[idx]
    for row in result["evidence"]["top_clips"]:
        idx = int(row["clip_index"])
        row["semantic_label"] = clip_semantics.get(idx)
        row["semantic_profile"] = clip_profiles.get(idx)
        if 0 <= idx < len(clip_metrics):
            row["native_visual_metrics"] = clip_metrics[idx]

    result["scores"]["student_ecr_raw_checkpoint"] = raw_student_score
    result["scores"]["native_heuristic_ecr"] = native.heuristic_score
    result["scores"]["student_ecr"] = student_ecr
    result["scores"]["band"] = engagement_band(student_ecr, reference_values)
    recommendations = build_recommendations(
        score=student_ecr,
        title=native.metadata.get("title"),
        description=native.metadata.get("description"),
        clip_rows=result["evidence"]["all_clips"],
        clip_metrics=clip_metrics,
    )
    semantic_attributes = build_semantic_attributes(
        title=native.metadata.get("title"),
        description=native.metadata.get("description"),
        clip_rows=result["evidence"]["all_clips"],
        clip_metrics=clip_metrics,
    )
    llm_payload = build_semantic_llm_input(
        result,
        semantic_attributes=semantic_attributes,
        recommendations=recommendations,
    )
    semantic_explanation = generate_semantic_explanation(
        llm_payload,
        language=args.explanation_language,
        enabled=not args.disable_llm,
    )
    if not semantic_explanation["summary"]:
        semantic_explanation["summary"] = build_student_summary(result)
    if not semantic_explanation["claims"]:
        semantic_explanation["claims"] = build_student_claims(result)
    if not semantic_explanation["recommendations"]:
        semantic_explanation["recommendations"] = recommendations

    recommendation_groups = build_recommendation_groups(
        title=native.metadata.get("title"),
        description=native.metadata.get("description"),
        clip_rows=result["evidence"]["all_clips"],
        clip_metrics=clip_metrics,
        semantic_attributes=semantic_attributes,
        fallback_recommendations=semantic_explanation["recommendations"],
    )
    metadata_suggestion = build_metadata_suggestion(
        title=native.metadata.get("title"),
        description=native.metadata.get("description"),
        top_clips=result["evidence"]["top_clips"],
        semantic_attributes=semantic_attributes,
    )
    result["recommendations"] = [
        *recommendation_groups["post_production"],
        *recommendation_groups["content_reshoot"],
    ]
    result["recommendations_grouped"] = recommendation_groups
    result["metadata_suggestion"] = metadata_suggestion
    result["semantic_attributes"] = {
        "type": "posthoc_semantic_attributes",
        "attributes": semantic_attributes,
        "note": (
            "These attributes are deterministic semantic labels over video/text evidence. "
            "They are not a separately trained concept bottleneck model."
        ),
    }
    result["semantic_explanation"] = {
        **semantic_explanation,
        "input_package": llm_payload,
    }
    result["nla_style_explanation"]["summary"] = semantic_explanation["summary"]
    result["nla_style_explanation"]["claims"] = semantic_explanation["claims"]
    result["nla_style_explanation"]["natural_language_bottleneck"]["verbalizer"] = (
        "semantic-labeling evidence package followed by optional LLM explanation"
    )
    result["nla_style_explanation"]["limitations"] = (
        "Student-only explanation: teacher model khong duoc goi o inference. "
        "LLM/template explanation chi duoc phep dien dat lai structured evidence "
        "tu attention, ablation va semantic labels; day la NLA-inspired pipeline, "
        "khong phai full trained Natural Language Autoencoder."
    )
    result["concept_bottleneck"] = {
        "type": "deprecated_alias_for_semantic_attributes",
        "concepts": semantic_attributes,
        "note": (
            "Kept for backward-compatible UI/report readers. The current thesis framing "
            "uses semantic labeling -> LLM explanation, not an extra trained concept model."
        ),
    }
    result["meta"] = {
        "inference_mode": "student_only_native_video",
        "teacher_called_at_inference": False,
        "device": str(device),
        "explanation_pipeline": "student_attribution_ablation -> semantic_labeling -> optional_llm_or_template",
        "llm_used": semantic_explanation["llm"]["used_llm"],
        "llm_provider": semantic_explanation["llm"]["provider"],
        "video_path": str(Path(args.video).resolve()),
        "report_json": str(report_path) if report_path else None,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_loaded": checkpoint_loaded,
        "native_input_preset": native_input_preset,
        "student_score_policy": score_policy,
        "text_encoder_model": args.text_encoder_model if native_input_preset == "clip_mobilenet_text" else None,
        "input_distribution_note": (
            "Proper KD demo inputs are reconstructed from raw video with CLIP ViT-B/32, "
            "MobileNetV3-Small spatial-motion features, and Stable-Diffusion CLIP text embeddings. "
            "Teacher artifacts remain training/offline supervision only."
            if native_input_preset == "clip_mobilenet_text"
            else "Native EfficientNet/low-level features are fed to the compact student. "
            "Teacher artifacts remain training/offline supervision only."
        ),
    }

    assets_dir = Path(args.assets_dir) if args.assets_dir else None
    if assets_dir:
        assets = save_top_clip_thumbnails(
            native.clips,
            result["evidence"]["top_clips"],
            assets_dir,
            prefix=Path(args.video).stem,
        )
        result["assets"] = {"top_clip_thumbnails": assets}

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


def build_student_summary(result: dict) -> str:
    score = float(result["scores"]["student_ecr"])
    band = result["scores"]["band"].get("label_vi", result["scores"]["band"].get("label"))
    top = result["evidence"]["top_clips"][0] if result["evidence"]["top_clips"] else None
    if top:
        return (
            f"Mô hình dự đoán ECR={score:.4f}, thuộc nhóm {band}. "
            f"Đoạn nổi bật nhất là {top['relative_time']['label']} "
            f"({top.get('semantic_label', 'tín hiệu thị giác nổi bật')})."
        )
    return f"Mô hình dự đoán ECR={score:.4f}, thuộc nhóm {band}."


def build_student_claims(result: dict) -> list[str]:
    claims = [build_student_summary(result)]
    top_clips = result["evidence"]["top_clips"]
    if top_clips:
        parts = []
        for row in top_clips:
            delta = float(row["contribution_to_score"])
            verb = "hỗ trợ tăng điểm" if delta >= 0 else "kìm điểm"
            parts.append(
                f"{row['relative_time']['label']}: {row.get('semantic_label')} "
                f"({verb} khoảng {abs(delta):.4f})"
            )
        claims.append("Bằng chứng thời gian mô hình chọn: " + "; ".join(parts) + ".")
    text_rows = result["evidence"].get("text_streams", [])
    if text_rows:
        row = text_rows[0]
        if row.get("source_text"):
            claims.append(
                f"Văn bản nổi bật là {_stream_label(row['stream'])}: \"{row['source_text']}\" "
                f"(mức ảnh hưởng={row['contribution_to_score']:.4f})."
            )
    return claims


def build_recommendation_groups(
    *,
    title: str | None,
    description: str | None,
    clip_rows: list[dict],
    clip_metrics: list[dict[str, float]],
    semantic_attributes: list[dict],
    fallback_recommendations: list[str],
) -> dict[str, list[str] | str]:
    """Split suggestions into feasible post-production edits and content changes."""

    post: list[str] = []
    content: list[str] = []

    if clip_metrics:
        avg_brightness = sum(float(m.get("brightness", 0.5)) for m in clip_metrics) / len(clip_metrics)
        avg_contrast = sum(float(m.get("contrast", 0.18)) for m in clip_metrics) / len(clip_metrics)
        avg_sharpness = sum(float(m.get("sharpness", 0.002)) for m in clip_metrics) / len(clip_metrics)
        avg_saturation = sum(float(m.get("saturation", 0.24)) for m in clip_metrics) / len(clip_metrics)
        if avg_brightness < 0.42:
            post.append("Hậu kì: tăng sáng nhẹ ở các đoạn tối, tránh làm cháy vùng sáng.")
        elif avg_brightness > 0.78:
            post.append("Hậu kì: giảm sáng nhẹ ở các đoạn quá sáng để giữ chi tiết.")
        if avg_contrast < 0.16:
            post.append("Hậu kì: tăng contrast nhẹ để chủ thể và bối cảnh tách rõ hơn.")
        if avg_sharpness < 0.0012:
            post.append("Hậu kì: tăng sharpness nhẹ ở các đoạn hơi mờ, không sharpen quá tay.")
        if avg_saturation < 0.20:
            post.append("Hậu kì: tăng saturation nhẹ nếu màu đang bị nhạt.")

    text = " ".join(x for x in [title, description] if x)
    words = [w for w in text.split() if w.strip()]
    if not title:
        post.append("Metadata: thêm title ngắn, cụ thể, nêu trực tiếp hành động/chủ thể chính trong video.")
    elif len(words) > 14:
        post.append("Metadata: rút title gọn hơn, bớt chuỗi hashtag/từ khóa chung chung.")
    if not description:
        post.append("Metadata: thêm description 1 câu để giải thích ngữ cảnh chính của video.")

    if clip_rows:
        early = [row for row in clip_rows if float(row["relative_time"]["end_pct"]) <= 0.25]
        later = [row for row in clip_rows if float(row["relative_time"]["start_pct"]) >= 0.5]
        early_gain = max([float(row["contribution_to_score"]) for row in early] or [0.0])
        later_gain = max([float(row["contribution_to_score"]) for row in later] or [0.0])
        if later_gain > early_gain + 0.015:
            content.append("Cảnh quay/dựng: đưa khoảnh khắc có hành động/chủ thể mạnh lên 0-3 giây đầu để hook tốt hơn.")
        weak_rows = sorted(clip_rows, key=lambda row: float(row.get("contribution_to_score", 0.0)))[:3]
        weak_labels = [
            row.get("relative_time", {}).get("label")
            for row in weak_rows
            if float(row.get("contribution_to_score", 0.0)) < 0.01
        ]
        if weak_labels:
            content.append(
                "Cảnh quay/dựng: cân nhắc rút ngắn hoặc thay thế các đoạn đóng góp yếu "
                f"({', '.join(weak_labels)}) bằng khoảnh khắc rõ hành động hơn."
            )

    motion_attr = next((attr for attr in semantic_attributes if attr.get("name") == "motion_action"), None)
    pacing_attr = next((attr for attr in semantic_attributes if attr.get("name") == "pacing_variety"), None)
    if motion_attr and float(motion_attr.get("score", 0.0)) < 0.35:
        content.append("Cảnh quay: thêm chuyển động/hành động rõ hơn; phần này cần quay hoặc dựng lại, auto-edit không tự tạo được.")
    if pacing_attr and float(pacing_attr.get("score", 0.0)) < 0.35:
        content.append("Cảnh quay/dựng: tăng biến đổi nhịp hình giữa các đoạn để video bớt đều đều.")

    if not post:
        post.append("Hậu kì: tín hiệu ánh sáng/contrast/độ nét hiện khá ổn; auto-edit sẽ chỉ chỉnh rất nhẹ nếu cần.")
    if not content:
        for item in fallback_recommendations:
            if not _looks_like_post_production(item):
                content.append(item)
        if not content:
            content.append("Cảnh quay/dựng: giữ các top clips đang mạnh và thử A/B hook hoặc nhịp cắt ở bản dựng tiếp theo.")

    return {
        "type": "split_by_editability",
        "post_production": _dedupe(post)[:5],
        "content_reshoot": _dedupe(content)[:5],
    }


def build_metadata_suggestion(
    *,
    title: str | None,
    description: str | None,
    top_clips: list[dict],
    semantic_attributes: list[dict],
) -> dict[str, object]:
    """Generate clean user-facing metadata for the demo rerun.

    The title/description fields should read like publishable metadata, not
    like model analysis. Diagnostic reasons stay in ``changes`` only.
    """

    clean_title = _clean_metadata_text(title)
    clean_description = _clean_metadata_text(description)
    title_terms = _meaningful_terms(clean_title)
    description_terms = _meaningful_terms(clean_description)

    if title_terms:
        subject = " ".join(title_terms[:5])
    elif description_terms:
        subject = " ".join(description_terms[:5])
    else:
        subject = "khoảnh khắc đáng chú ý"

    suggested_title = _natural_metadata_title(subject)
    suggested_title = _trim_words(suggested_title, 12)

    if clean_description and not _looks_like_analysis_metadata(clean_description):
        suggested_description = _trim_sentence(clean_description, 24)
    else:
        suggested_description = _natural_metadata_description(subject)

    changes: list[str] = []
    if not clean_title:
        changes.append("Thêm title cụ thể thay vì để trống.")
    elif len(str(title).split()) > 14 or _has_hashtag_cluster(str(title)):
        changes.append("Rút gọn title và giảm chuỗi hashtag/từ khóa chung chung.")
    else:
        changes.append("Giữ title ngắn, tập trung hơn vào chủ thể/hành động chính.")
    if not clean_description:
        changes.append("Thêm description 1 câu để bổ sung ngữ cảnh.")
    else:
        changes.append("Viết lại description thành câu tự nhiên, nêu rõ điểm nổi bật của video.")

    return {
        "title": suggested_title,
        "description": suggested_description,
        "changes": changes,
        "note": "Có thể chỉnh 2 field này trên UI trước khi auto-edit và chấm lại.",
    }


def _natural_metadata_title(subject: str) -> str:
    subject = " ".join(subject.split())
    if not subject:
        return "Khoảnh khắc đáng chú ý"
    words = subject.split()
    if _mostly_ascii(subject):
        base = subject
        if len(words) < 3 and not any(word.lower() in {"moment", "highlights", "clip"} for word in words):
            base = f"{subject} moment"
        return _title_case_vi(base)
    if len(words) < 3:
        subject = f"{subject} đáng chú ý"
    return _title_case_vi(subject)


def _natural_metadata_description(subject: str) -> str:
    subject = " ".join(subject.split())
    if not subject:
        return "Một video ngắn ghi lại khoảnh khắc đáng chú ý."
    if _mostly_ascii(subject):
        return f"A short video about {subject}."
    return f"Một video ngắn về {subject}."


def _mostly_ascii(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    ascii_letters = [ch for ch in letters if ord(ch) < 128]
    return len(ascii_letters) / max(len(letters), 1) >= 0.85


def _looks_like_analysis_metadata(text: str) -> bool:
    lowered = text.lower()
    analysis_markers = [
        "chi tiết thị giác",
        "màu sắc nổi bật",
        "chuyển động/hành động",
        "hook đầu video",
        "điểm nổi bật",
        "có thể tối ưu",
        "mô hình",
        "bằng chứng",
        "% video",
    ]
    return any(marker in lowered for marker in analysis_markers)


def pretty_attribute_vi(name: object) -> str:
    labels = {
        "hook_strength": "hook đầu video",
        "motion_action": "chuyển động/hành động",
        "visual_clarity": "độ rõ hình ảnh",
        "lighting_quality": "ánh sáng",
        "text_specificity": "độ cụ thể metadata",
        "pacing_variety": "nhịp dựng",
    }
    return labels.get(str(name), str(name).replace("_", " "))


def _clean_metadata_text(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _meaningful_terms(text: str) -> list[str]:
    tokens = [
        token.strip("#@.,:;!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip("#@.,:;!?()[]{}\"'")
    ]
    stop = {
        "the",
        "and",
        "or",
        "vs",
        "a",
        "an",
        "video",
        "shorts",
        "fyp",
        "viral",
        "trending",
        "motivation",
    }
    out: list[str] = []
    for token in tokens:
        if token in stop or len(token) <= 1:
            continue
        if token not in out:
            out.append(token)
    return out


def _has_hashtag_cluster(text: str) -> bool:
    tokens = [part for part in text.split() if part.strip()]
    hashtag_like = [part for part in tokens if part.startswith("#") or part[:1].isupper()]
    return len(tokens) >= 10 and len(hashtag_like) >= max(5, len(tokens) // 2)


def _title_case_vi(text: str) -> str:
    cleaned = " ".join(text.replace("_", " ").split())
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Khoảnh khắc nổi bật trong video"


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def _trim_sentence(text: str, max_words: int) -> str:
    trimmed = _trim_words(text, max_words).rstrip(".,;:")
    return trimmed + "." if trimmed and trimmed[-1] not in ".!?" else trimmed


def _compact_semantic_label(label: str) -> str:
    parts = [part.strip() for part in label.split(",") if part.strip()]
    return ", ".join(parts[:2]) if parts else "khoảnh khắc nổi bật"


def resolve_device(raw: str) -> torch.device:
    requested = (raw or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _looks_like_post_production(text: str) -> bool:
    lowered = text.lower()
    needles = ["sáng", "contrast", "tương phản", "sharp", "nét", "màu", "saturation", "title", "tiêu đề", "mô tả", "description"]
    return any(token in lowered for token in needles)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = " ".join(item.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _stream_label(stream: object) -> str:
    return {
        "title": "tiêu đề",
        "description": "mô tả",
        "caption": "caption thị giác",
        "sound": "âm thanh",
    }.get(str(stream), str(stream))


def make_input_config(streams: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        use_sound_text="sound" in streams,
        use_title_text="title" in streams or "empty_metadata" in streams,
        use_description_text="description" in streams,
        use_caption_text=False,
    )


def resolve_native_input_preset(report_input_preset: object, clip_input_dim: int) -> str:
    """Choose a teacher-free extractor compatible with the student input width."""

    preset = str(report_input_preset or "").strip()
    if preset == "clip_mobilenet_text" and clip_input_dim == 1664:
        return preset
    if preset and preset != "clip_mobilenet_text":
        return preset
    if clip_input_dim == 1664:
        return "clip_mobilenet_text"
    return "visual_text"


def load_reference_ecr_values(csv_path: str | None) -> list[float] | None:
    if not csv_path:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "ECR" not in df.columns:
            return None
        values = [float(x) for x in df["ECR"].tolist() if math.isfinite(float(x))]
        return values or None
    except Exception:
        return None


def resolve_report_path(raw: str | None) -> Path | None:
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            Path.home()
            / "workspace/results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json",
            ROOT / "results/kd_tuning_official_5k/v05_small_cosine_rank/official_student_kd_report.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return Path(raw).expanduser() if raw else None


def resolve_checkpoint_path(raw: str | None, report_path: Path | None) -> Path | None:
    if raw:
        return Path(raw).expanduser()
    if report_path:
        return report_path.parent / "student_kd_best.pth"
    return None


if __name__ == "__main__":
    main()

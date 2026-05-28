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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--efficientnet-weights", default=None)
    parser.add_argument("--no-visual-encoder", action="store_true")
    parser.add_argument("--explanation-language", default="vi", choices=["vi", "en"])
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Use deterministic template explanation even if an LLM API key is configured.",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--assets-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
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

    native = build_native_student_inputs(
        args.video,
        title=args.title,
        description=args.description,
        max_clips=max_clips,
        clip_dim=int(model_kwargs.get("clip_input_dim", 1024)),
        device=device,
        efficientnet_weights=args.efficientnet_weights,
        no_visual_encoder=args.no_visual_encoder,
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
    # Native features are intentionally teacher-free and may not match the exact
    # teacher-artifact distribution used in KD. Blend with an interpretable
    # native heuristic so demos remain stable while still reporting both values.
    student_ecr = 0.72 * raw_student_score + 0.28 * native.heuristic_score
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

    result["recommendations"] = semantic_explanation["recommendations"]
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
        "explanation_pipeline": "student_attribution_ablation -> semantic_labeling -> optional_llm_or_template",
        "llm_used": semantic_explanation["llm"]["used_llm"],
        "llm_provider": semantic_explanation["llm"]["provider"],
        "video_path": str(Path(args.video).resolve()),
        "report_json": str(report_path) if report_path else None,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_loaded": checkpoint_loaded,
        "input_distribution_note": (
            "Native EfficientNet/low-level features are fed to the compact student. "
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
            f"Dự đoán ECR={score:.4f}, thuộc nhóm {band}. "
            f"Đoạn nổi bật nhất là {top['relative_time']['label']} "
            f"({top.get('semantic_label', 'tín hiệu thị giác nổi bật')}); "
            f"khi bỏ evidence đã chọn, score còn "
            f"{result['nla_style_explanation']['faithfulness']['remove_selected_ecr']:.4f}."
        )
    return f"Dự đoán ECR={score:.4f}, thuộc nhóm {band}."


def build_student_claims(result: dict) -> list[str]:
    claims = [build_student_summary(result)]
    top_clips = result["evidence"]["top_clips"]
    if top_clips:
        parts = []
        for row in top_clips:
            delta = float(row["contribution_to_score"])
            verb = "hỗ trợ tăng score" if delta >= 0 else "kìm score"
            parts.append(
                f"{row['relative_time']['label']}: {row.get('semantic_label')} "
                f"({verb} khoảng {abs(delta):.4f})"
            )
        claims.append("Temporal evidence student chọn: " + "; ".join(parts) + ".")
    text_rows = result["evidence"].get("text_streams", [])
    if text_rows:
        row = text_rows[0]
        if row.get("source_text"):
            claims.append(
                f"Text evidence nổi bật là {row['stream']}: \"{row['source_text']}\" "
                f"(contribution={row['contribution_to_score']:.4f})."
            )
    faith = result["nla_style_explanation"]["faithfulness"]
    claims.append(
        f"Faithfulness check: keep-only={faith['keep_only_selected_ecr']:.4f}, "
        f"remove-selected={faith['remove_selected_ecr']:.4f}, "
        f"reconstruction_error={faith['reconstruction_error_abs']:.4f}."
    )
    return claims


def make_input_config(streams: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        use_sound_text=False,
        use_title_text="title" in streams or "empty_metadata" in streams,
        use_description_text="description" in streams,
        use_caption_text=False,
    )


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

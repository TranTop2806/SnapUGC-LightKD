"""Deterministic, feasible video edits driven by explanation evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ClipEdit:
    clip_index: int
    start_pct: float
    end_pct: float
    label: str
    contribution: float
    operations: dict[str, float]
    reasons: list[str]


def build_feasible_edit_plan(
    result: dict[str, Any],
    *,
    max_clips: int = 3,
) -> dict[str, Any]:
    """Choose non-top clips that can be improved with deterministic edits."""

    top_indices = {
        int(row.get("clip_index"))
        for row in result.get("evidence", {}).get("top_clips", [])
        if row.get("clip_index") is not None
    }
    all_rows = result.get("evidence", {}).get("all_clips", [])
    candidates: list[ClipEdit] = []
    for row in all_rows:
        idx = int(row.get("clip_index", -1))
        if idx < 0 or idx in top_indices:
            continue
        metrics = row.get("native_visual_metrics") or {}
        operations, reasons = _operations_for_metrics(metrics)
        contribution = float(row.get("contribution_to_score", 0.0))
        if operations:
            rel = row.get("relative_time", {})
            candidates.append(
                ClipEdit(
                    clip_index=idx,
                    start_pct=float(rel.get("start_pct", 0.0)),
                    end_pct=float(rel.get("end_pct", 0.0)),
                    label=str(rel.get("label", f"clip {idx}")),
                    contribution=contribution,
                    operations=operations,
                    reasons=reasons,
                )
            )

    if not candidates:
        for row in sorted(all_rows, key=lambda item: float(item.get("contribution_to_score", 0.0))):
            idx = int(row.get("clip_index", -1))
            if idx < 0 or idx in top_indices:
                continue
            rel = row.get("relative_time", {})
            candidates.append(
                ClipEdit(
                    clip_index=idx,
                    start_pct=float(rel.get("start_pct", 0.0)),
                    end_pct=float(rel.get("end_pct", 0.0)),
                    label=str(rel.get("label", f"clip {idx}")),
                    contribution=float(row.get("contribution_to_score", 0.0)),
                    operations={"contrast": 1.06, "sharpness": 0.25},
                    reasons=["Clip đóng góp yếu hơn top clips; áp dụng tăng contrast/sharpness rất nhẹ."],
                )
            )
            break

    selected = sorted(candidates, key=lambda item: (item.contribution, item.clip_index))[:max_clips]
    return {
        "strategy": (
            "Chỉ chỉnh các clip không nằm trong top evidence. Các chỉnh sửa được giới hạn ở "
            "brightness/contrast/sharpness/saturation để giữ nội dung, timeline và hành động gốc."
        ),
        "top_clip_indices_preserved": sorted(top_indices),
        "selected_clip_indices": [item.clip_index for item in selected],
        "edits": [_clip_edit_to_dict(item) for item in selected],
        "skipped_capabilities": [
            "Không tự thay cảnh quay/chủ thể/hành động.",
            "Không tự rút timeline vì có thể làm lệch audio và ý nghĩa video.",
            "Không sinh thêm cảnh mới.",
        ],
    }


def apply_feasible_video_edits(
    *,
    input_video: Path,
    output_video: Path,
    result: dict[str, Any],
    max_clips: int = 3,
) -> dict[str, Any]:
    """Apply feasible edits and write a new mp4 video."""

    plan = build_feasible_edit_plan(result, max_clips=max_clips)
    edits = [
        ClipEdit(
            clip_index=int(item["clip_index"]),
            start_pct=float(item["start_pct"]),
            end_pct=float(item["end_pct"]),
            label=str(item["label"]),
            contribution=float(item["contribution_to_score"]),
            operations={str(k): float(v) for k, v in item["operations"].items()},
            reasons=[str(x) for x in item["reasons"]],
        )
        for item in plan["edits"]
    ]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if not edits:
        shutil.copy2(input_video, output_video)
        return {
            **plan,
            "status": "copied_without_edits",
            "output_video": str(output_video),
        }

    temp_video = output_video.with_name(output_video.stem + ".silent.mp4")
    try:
        _write_edited_silent_video(input_video=input_video, output_video=temp_video, edits=edits)
        _mux_original_audio(
            video_without_audio=temp_video,
            original_video=input_video,
            output_video=output_video,
        )
    finally:
        temp_video.unlink(missing_ok=True)
    sidecar = output_video.with_suffix(".edit_plan.json")
    sidecar.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        **plan,
        "status": "edited",
        "output_video": str(output_video),
        "plan_json": str(sidecar),
    }


def _operations_for_metrics(metrics: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    operations: dict[str, float] = {}
    reasons: list[str] = []
    brightness = float(metrics.get("brightness", 0.5))
    contrast = float(metrics.get("contrast", 0.18))
    sharpness = float(metrics.get("sharpness", 0.002))
    saturation = float(metrics.get("saturation", 0.24))
    colorfulness = float(metrics.get("colorfulness", 0.18))

    if brightness < 0.42:
        target = 0.50 if brightness < 0.34 else 0.47
        operations["brightness"] = float(np.clip((target - brightness) * 70.0, 4.0, 14.0))
        reasons.append("Khung hình hơi tối, tăng sáng nhẹ.")
    elif brightness > 0.78:
        operations["brightness"] = float(np.clip((0.72 - brightness) * 55.0, -10.0, -4.0))
        reasons.append("Khung hình quá sáng, giảm sáng nhẹ để đỡ cháy.")

    if contrast < 0.16:
        operations["contrast"] = float(np.clip(1.0 + (0.18 - contrast) * 0.65, 1.04, 1.10))
        reasons.append("Contrast thấp, tăng tương phản nhẹ.")

    if sharpness < 0.0012:
        operations["sharpness"] = float(np.clip(0.25 + (0.0012 - sharpness) * 120.0, 0.25, 0.45))
        reasons.append("Độ nét thấp, áp dụng unsharp mask nhẹ.")

    if saturation < 0.20 and colorfulness < 0.17:
        operations["saturation"] = 1.05
        reasons.append("Màu hơi nhạt, tăng saturation nhẹ.")

    return operations, reasons


def _write_edited_silent_video(*, input_video: Path, output_video: Path, edits: list[ClipEdit]) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {input_video}")

    ranges = [
        (
            max(0, int(round(item.start_pct * frame_count))),
            min(frame_count, int(round(item.end_pct * frame_count))),
            item.operations,
        )
        for item in edits
    ]
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write video: {output_video}")

    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            operations = _operations_for_frame(index, ranges)
            if operations:
                frame = _apply_operations(frame, operations)
            writer.write(frame)
            index += 1
    finally:
        cap.release()
        writer.release()


def _operations_for_frame(index: int, ranges: list[tuple[int, int, dict[str, float]]]) -> dict[str, float] | None:
    for start, end, operations in ranges:
        if start <= index < end:
            return operations
    return None


def _apply_operations(frame: np.ndarray, operations: dict[str, float]) -> np.ndarray:
    alpha = float(operations.get("contrast", 1.0))
    beta = float(operations.get("brightness", 0.0))
    out = np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    saturation = operations.get("saturation")
    if saturation and abs(float(saturation) - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    sharpness = float(operations.get("sharpness", 0.0))
    if sharpness > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
        out = cv2.addWeighted(out, 1.0 + sharpness, blurred, -sharpness, 0)
    return out


def _mux_original_audio(*, video_without_audio: Path, original_video: Path, output_video: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to preserve the original audio during auto-edit")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_without_audio),
        "-i",
        str(original_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        output_video.unlink(missing_ok=True)
        detail = (exc.stderr or exc.stdout or "unknown ffmpeg error").strip()[-1000:]
        raise RuntimeError(f"ffmpeg could not preserve the original audio: {detail}") from exc


def _clip_edit_to_dict(item: ClipEdit) -> dict[str, Any]:
    return {
        "clip_index": item.clip_index,
        "label": item.label,
        "start_pct": item.start_pct,
        "end_pct": item.end_pct,
        "contribution_to_score": item.contribution,
        "operations": item.operations,
        "reasons": item.reasons,
    }

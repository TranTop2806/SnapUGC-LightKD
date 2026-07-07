"""Best-effort paper assets generated from the first teacher sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_paper_sample(
    video_path: str | Path,
    video_id: str,
    caption: str,
    sound: str,
    title: str,
    description: str,
    output_dir: str | Path,
) -> None:
    """Preserve the original runner's sample images and metadata outputs."""
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {video_path} for visualization")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video {video_path} has no readable frames")
        middle = _read_frame(capture, total_frames // 2)
        frames = [_read_frame(capture, int(index * (total_frames - 1) / 5)) for index in range(6)]
    finally:
        capture.release()

    cv2.imwrite(str(output_dir / "sample_ci.png"), middle)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(output_dir / f"sample_cjk_{index}.png"), frame)
    cv2.imwrite(str(output_dir / "sample_cjk_stack.png"), _stack_frames(frames))
    _save_metadata(output_dir, video_id, caption, sound, title, description)
    _save_summary_plot(output_dir, video_id, middle, frames, caption, sound, title, description)


def _read_frame(capture: object, frame_index: int) -> np.ndarray:
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success:
        raise ValueError(f"Could not read video frame {frame_index}")
    return frame


def _stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    import cv2

    width, height = 225, 400
    dx, dy = 15, 12
    canvas = np.full(
        (height + (len(frames) - 1) * dy, width + (len(frames) - 1) * dx, 3),
        255,
        dtype=np.uint8,
    )
    for index, frame in enumerate(frames):
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        x = index * dx
        y = (len(frames) - 1 - index) * dy
        canvas[y : y + height, x : x + width] = resized
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (60, 60, 60), 2)
    return canvas


def _save_metadata(
    output_dir: Path,
    video_id: str,
    caption: str,
    sound: str,
    title: str,
    description: str,
) -> None:
    payload = {
        "video_id": video_id,
        "title": title,
        "description": description,
        "generated_caption": caption,
        "sound_classification": sound,
    }
    (output_dir / "sample_text_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = (
        f"Sample Video ID: {video_id}\n"
        "=========================================\n"
        f"Title: {title}\nDescription: {description}\n\n"
        f'Generated Caption (mPLUG-2):\n"{caption}"\n\n'
        f'Sound Classification (YAMNet):\n"{sound}"\n'
    )
    (output_dir / "sample_text_metadata.txt").write_text(text, encoding="utf-8")


def _save_summary_plot(
    output_dir: Path,
    video_id: str,
    middle: np.ndarray,
    frames: list[np.ndarray],
    caption: str,
    sound: str,
    title: str,
    description: str,
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 6), facecolor="white")
    grid = gridspec.GridSpec(2, 2, height_ratios=[1.7, 1.0], width_ratios=[1.0, 1.2])
    middle_axis = figure.add_subplot(grid[0, 0])
    middle_axis.imshow(cv2.cvtColor(middle, cv2.COLOR_BGR2RGB))
    middle_axis.set_title("Single Representative Frame ($C_i$)", fontweight="bold")
    middle_axis.axis("off")
    frame_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid[0, 1])
    for index, frame in enumerate(frames):
        axis = figure.add_subplot(frame_grid[index])
        axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axis.axis("off")
    text_axis = figure.add_subplot(grid[1, :])
    text_axis.axis("off")
    text_axis.text(
        0.01,
        0.95,
        f"Sample Video ID: {video_id}\nTitle: {title}\nDescription: {description}\n\n"
        f'Generated Caption: "{caption}"\nSound Classification: "{sound}"',
        transform=text_axis.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F8F9FA", "edgecolor": "#E2E8F0"},
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"paper_style_sample_{video_id}.png",
        dpi=150,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(figure)

"""Export intermediate tensors produced by the official SnapUGC teacher.

The official inference script imports the compatibility functions at the end
of this module. Keeping the implementation here makes the artifact format
testable without importing or modifying the heavyweight teacher stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_KEYS = (
    "clip_ecr",
    "fusion_hidden",
    "temporal_hidden",
    "caption_feature",
    "action_feature",
    "frame_fusion_feature",
    "text_pooled",
    "attention_importance",
)


class TeacherArtifactExporter:
    """Buffer teacher outputs and write backward-compatible NPZ shards."""

    def __init__(self, output_dir: str | Path | None, shard_size: int = 500) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.shard_size = shard_size
        self.rows: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def add(
        self,
        *,
        idx: int,
        video_id: str,
        teacher_ecr: float,
        artifacts: dict[str, Any] | None,
    ) -> Path | None:
        if not self.enabled:
            return None
        if artifacts is None:
            print(f"missing_teacher_artifacts {idx} {video_id}", flush=True)
            return None

        row: dict[str, Any] = {
            "idx": int(idx),
            "Id": str(video_id),
            "teacher_ecr": float(teacher_ecr),
        }
        row.update({key: _to_numpy(value) for key, value in artifacts.items()})
        self.rows.append(row)
        return self.flush()

    def flush(self, *, force: bool = False) -> Path | None:
        if not self.enabled or not self.rows:
            return None
        if not force and len(self.rows) < self.shard_size:
            return None

        rows = list(self.rows)
        self.rows.clear()
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"official_teacher_artifacts_{int(rows[0]['idx']):04d}_{int(rows[-1]['idx']):04d}"
        payload: dict[str, np.ndarray] = {
            "ids": np.asarray([row["Id"] for row in rows], dtype="<U32"),
            "order_idx": np.asarray([row["idx"] for row in rows], dtype=np.int32),
            "teacher_ecr": np.asarray([row["teacher_ecr"] for row in rows], dtype=np.float32),
        }
        for key in ARTIFACT_KEYS:
            flat, offsets, shapes = pack_ragged(rows, key)
            payload[f"{key}_flat"] = flat
            payload[f"{key}_offsets"] = offsets
            payload[f"{key}_shapes"] = shapes

        output_path = self.output_dir / f"{prefix}.npz"
        np.savez_compressed(output_path, **payload)
        print(
            f"saved_teacher_artifact_shard {prefix} n={len(rows)} dir={self.output_dir}", flush=True
        )
        return output_path


def _to_numpy(value: Any, dtype: np.dtype = np.float16) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def pack_ragged(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    offsets = [0]
    shapes: list[tuple[int, ...]] = []
    for row in rows:
        value = row.get(key)
        array = np.zeros((0,), dtype=np.float16) if value is None else np.asarray(value)
        flattened = array.reshape(-1)
        arrays.append(flattened)
        shapes.append(array.shape)
        offsets.append(offsets[-1] + flattened.size)
    flat = np.concatenate(arrays) if arrays else np.zeros((0,), dtype=np.float16)
    return flat, np.asarray(offsets, dtype=np.int64), np.asarray(shapes, dtype=np.int32)


def _exporter_from_environment() -> TeacherArtifactExporter:
    return TeacherArtifactExporter(
        os.environ.get("SNAPUGC_ARTIFACT_DIR"),
        int(os.environ.get("SNAPUGC_ARTIFACT_SHARD_SIZE", "500")),
    )


_EXPORTER = _exporter_from_environment()


def save_teacher_artifact(
    idx: int,
    video_id: str,
    teacher_ecr: float,
    model: Any,
    caption: str,
    sound: str,
    title: str,
    description: str,
    video_path: str | Path,
) -> None:
    """Compatibility hook called from the patched official inference loop."""
    _EXPORTER.add(
        idx=idx,
        video_id=video_id,
        teacher_ecr=teacher_ecr,
        artifacts=getattr(model, "last_artifacts", None),
    )
    if int(idx) == 0 and _EXPORTER.enabled:
        _save_sample_metadata(video_id, caption, sound, title, description, video_path)


def flush_teacher_artifacts(*, force: bool = False) -> Path | None:
    """Compatibility hook that flushes the final partial shard."""
    return _EXPORTER.flush(force=force)


def _save_sample_metadata(
    video_id: str,
    caption: str,
    sound: str,
    title: str,
    description: str,
    video_path: str | Path,
) -> None:
    """Generate the original first-sample paper assets without risking inference."""
    from snapugc_lightkd.teacher_visualization import save_paper_sample

    root = Path(os.environ.get("SNAPUGC_REPO_ROOT", Path.cwd()))
    try:
        save_paper_sample(video_path, video_id, caption, sound, title, description, root / "assets")
    except Exception as error:
        print(f"Failed to generate paper-style visualization: {error}", flush=True)

"""Dataset utilities for official SnapUGC teacher artifact shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


RAGGED_KEYS = (
    "clip_ecr",
    "fusion_hidden",
    "temporal_hidden",
    "caption_feature",
    "action_feature",
    "frame_fusion_feature",
    "text_tokens",
    "text_pooled",
    "attention_mean",
    "attention_importance",
)

DEFAULT_STUDENT_RAGGED_KEYS = (
    "clip_ecr",
    "fusion_hidden",
    "temporal_hidden",
    "frame_fusion_feature",
    "text_pooled",
    "attention_importance",
)


@dataclass(frozen=True)
class StudentInputConfig:
    """Which teacher-extracted features the student may use as input."""

    use_frame_fusion: bool = True
    use_action: bool = False
    use_caption_feature: bool = False
    use_sound_text: bool = False
    use_title_text: bool = True
    use_description_text: bool = True
    use_caption_text: bool = False

    @classmethod
    def from_preset(cls, preset: str) -> "StudentInputConfig":
        presets = {
            "visual_text": cls(),
            "visual_only": cls(use_title_text=False, use_description_text=False),
            "visual_text_action": cls(use_action=True),
            "visual_text_sound": cls(use_sound_text=True),
            "privileged": cls(
                use_action=True,
                use_caption_feature=True,
                use_sound_text=True,
                use_caption_text=True,
            ),
        }
        if preset not in presets:
            raise ValueError(f"Unknown student input preset {preset!r}. Choices: {sorted(presets)}")
        return presets[preset]


def artifact_keys_for_input_config(config: StudentInputConfig) -> tuple[str, ...]:
    keys = set(DEFAULT_STUDENT_RAGGED_KEYS)
    if config.use_action:
        keys.add("action_feature")
    if config.use_caption_feature:
        keys.add("caption_feature")
    return tuple(key for key in RAGGED_KEYS if key in keys)


def _read_labels(csv_path: str | Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    if not {"Id", "ECR"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain Id and ECR columns")
    return {str(row.Id): float(row.ECR) for row in df.itertuples(index=False)}


def _unpack_ragged(npz: np.lib.npyio.NpzFile, key: str, row_idx: int) -> np.ndarray:
    flat_key = f"{key}_flat"
    offsets_key = f"{key}_offsets"
    shapes_key = f"{key}_shapes"
    if flat_key not in npz or offsets_key not in npz or shapes_key not in npz:
        return np.zeros((0,), dtype=np.float32)
    offsets = npz[offsets_key]
    shapes = npz[shapes_key]
    start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
    shape = tuple(int(x) for x in shapes[row_idx] if int(x) > 0)
    values = npz[flat_key][start:end]
    if not shape or values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return values.reshape(shape).astype(np.float32, copy=False)


def load_official_artifact_rows(
    artifact_dir: str | Path,
    labels_csv: str | Path,
    *,
    require_complete_labels: bool = True,
    ragged_keys: Iterable[str] = DEFAULT_STUDENT_RAGGED_KEYS,
) -> list[dict[str, object]]:
    """Load all official teacher artifact shards and attach true ECR labels."""

    artifact_dir = Path(artifact_dir)
    shard_paths = sorted(artifact_dir.glob("official_teacher_artifacts_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No official_teacher_artifacts_*.npz files under {artifact_dir}")

    labels = _read_labels(labels_csv)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for shard_path in shard_paths:
        with np.load(shard_path) as npz:
            ids = [str(x) for x in npz["ids"]]
            order_idx = npz["order_idx"].astype(np.int64)
            teacher_ecr = npz["teacher_ecr"].astype(np.float32)
            for i, video_id in enumerate(ids):
                if video_id in seen:
                    continue
                if video_id not in labels:
                    if require_complete_labels:
                        raise KeyError(f"Missing ECR label for {video_id}")
                    continue
                row: dict[str, object] = {
                    "Id": video_id,
                    "order_idx": int(order_idx[i]),
                    "ecr_true": float(labels[video_id]),
                    "teacher_ecr": float(teacher_ecr[i]),
                }
                for key in ragged_keys:
                    row[key] = _unpack_ragged(npz, key, i)
                rows.append(row)
                seen.add(video_id)
    rows.sort(key=lambda row: int(row["order_idx"]))
    return rows


def _ensure_2d(array: np.ndarray, dim: int | None = None) -> np.ndarray:
    if array.size == 0:
        return np.zeros((0, dim or 0), dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _fit_2d(array: np.ndarray, length: int, dim: int) -> np.ndarray:
    array = _ensure_2d(array, dim)
    fitted = np.zeros((length, dim), dtype=np.float32)
    if array.size == 0 or length == 0:
        return fitted
    n = min(length, array.shape[0])
    d = min(dim, array.shape[1])
    fitted[:n, :d] = array[:n, :d]
    return fitted


def _fit_1d(array: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    fitted = np.zeros((length,), dtype=np.float32)
    if array.size == 0 or length == 0:
        return fitted
    n = min(length, array.shape[0])
    fitted[:n] = array[:n]
    return fitted


def _select_text_pooled(text_pooled: np.ndarray, config: StudentInputConfig) -> np.ndarray:
    text_pooled = _ensure_2d(text_pooled, 768)
    indices = []
    if config.use_sound_text:
        indices.append(0)
    if config.use_title_text:
        indices.append(1)
    if config.use_description_text:
        indices.append(2)
    if config.use_caption_text:
        indices.append(3)
    valid = [idx for idx in indices if idx < len(text_pooled)]
    if not valid:
        return np.zeros((0, 768), dtype=np.float32)
    return text_pooled[valid].astype(np.float32, copy=False)


class OfficialTeacherArtifactDataset(Dataset):
    """Student training dataset built from official teacher artifacts."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        input_config: StudentInputConfig,
        *,
        max_clips: int = 16,
    ):
        self.rows = rows
        self.input_config = input_config
        self.max_clips = max_clips
        self.clip_dim = self._infer_clip_dim()
        self.text_dim = 768

    def _infer_clip_dim(self) -> int:
        for row in self.rows:
            pieces = self._clip_pieces(row)
            if pieces:
                return int(sum(piece.shape[-1] for piece in pieces))
        raise RuntimeError("Could not infer student clip input dimension")

    def _clip_pieces(self, row: dict[str, object]) -> list[np.ndarray]:
        pieces = []
        if self.input_config.use_frame_fusion:
            pieces.append(_ensure_2d(row["frame_fusion_feature"], 1024))
        if self.input_config.use_action:
            pieces.append(_ensure_2d(row["action_feature"], 512))
        if self.input_config.use_caption_feature:
            pieces.append(_ensure_2d(row["caption_feature"], 1024))
        return [piece for piece in pieces if piece.size > 0]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        pieces = self._clip_pieces(row)
        if not pieces:
            raise RuntimeError(f"No student clip inputs available for {row['Id']}")
        min_len = min(piece.shape[0] for piece in pieces)
        min_len = min(min_len, self.max_clips)
        if min_len <= 0:
            raise RuntimeError(f"Zero-length student clip input for {row['Id']}")
        clip_inputs = np.concatenate([piece[:min_len] for piece in pieces], axis=-1)

        text_inputs = _select_text_pooled(row["text_pooled"], self.input_config)
        teacher_temporal = _fit_2d(row["temporal_hidden"], min_len, 512)
        teacher_fusion = _fit_2d(row["fusion_hidden"], min_len, 512)
        teacher_clip_ecr = _fit_1d(row["clip_ecr"], min_len)
        teacher_attention = _ensure_2d(row["attention_importance"], min_len)
        if teacher_attention.size > 0:
            teacher_attention = teacher_attention[:, :min_len].mean(axis=0)
        else:
            teacher_attention = np.ones((min_len,), dtype=np.float32) / max(min_len, 1)
        teacher_attention = _fit_1d(teacher_attention, min_len)

        return {
            "Id": row["Id"],
            "clip_inputs": torch.from_numpy(clip_inputs.astype(np.float32, copy=False)),
            "text_inputs": torch.from_numpy(text_inputs.astype(np.float32, copy=False)),
            "ecr_true": torch.tensor(float(row["ecr_true"]), dtype=torch.float32),
            "teacher_ecr": torch.tensor(float(row["teacher_ecr"]), dtype=torch.float32),
            "teacher_temporal": torch.from_numpy(teacher_temporal.astype(np.float32, copy=False)),
            "teacher_fusion": torch.from_numpy(teacher_fusion.astype(np.float32, copy=False)),
            "teacher_clip_ecr": torch.from_numpy(teacher_clip_ecr.astype(np.float32, copy=False)),
            "teacher_attention": torch.from_numpy(teacher_attention.astype(np.float32, copy=False)),
        }


def split_rows(
    rows: list[dict[str, object]],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(rows) * val_ratio)))
    val_ids = set(indices[:val_size].tolist())
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_ids]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_ids]
    return train_rows, val_rows


def collate_student_batch(batch: Iterable[dict[str, object]]) -> dict[str, object]:
    items = list(batch)
    batch_size = len(items)
    max_clips = max(int(item["clip_inputs"].shape[0]) for item in items)
    clip_dim = int(items[0]["clip_inputs"].shape[-1])
    max_text = max(int(item["text_inputs"].shape[0]) for item in items)
    text_dim = int(items[0]["text_inputs"].shape[-1]) if max_text else 768

    clip_inputs = torch.zeros(batch_size, max_clips, clip_dim)
    clip_mask = torch.zeros(batch_size, max_clips, dtype=torch.bool)
    text_inputs = torch.zeros(batch_size, max_text, text_dim)
    text_mask = torch.zeros(batch_size, max_text, dtype=torch.bool)
    teacher_temporal = torch.zeros(batch_size, max_clips, 512)
    teacher_fusion = torch.zeros(batch_size, max_clips, 512)
    teacher_clip_ecr = torch.zeros(batch_size, max_clips)
    teacher_attention = torch.zeros(batch_size, max_clips)

    ids = []
    for i, item in enumerate(items):
        ids.append(str(item["Id"]))
        n_clips = int(item["clip_inputs"].shape[0])
        n_text = int(item["text_inputs"].shape[0])
        clip_inputs[i, :n_clips] = item["clip_inputs"]
        clip_mask[i, :n_clips] = True
        if n_text:
            text_inputs[i, :n_text] = item["text_inputs"]
            text_mask[i, :n_text] = True
        teacher_temporal[i, :n_clips] = item["teacher_temporal"][:n_clips]
        teacher_fusion[i, :n_clips] = item["teacher_fusion"][:n_clips]
        teacher_clip_ecr[i, :n_clips] = item["teacher_clip_ecr"][:n_clips]
        attn = item["teacher_attention"][:n_clips]
        teacher_attention[i, :n_clips] = attn / attn.sum().clamp_min(1e-6)

    return {
        "ids": ids,
        "clip_inputs": clip_inputs,
        "clip_mask": clip_mask,
        "text_inputs": text_inputs,
        "text_mask": text_mask,
        "ecr_true": torch.stack([item["ecr_true"] for item in items]),
        "teacher_ecr": torch.stack([item["teacher_ecr"] for item in items]),
        "teacher_temporal": teacher_temporal,
        "teacher_fusion": teacher_fusion,
        "teacher_clip_ecr": teacher_clip_ecr,
        "teacher_attention": teacher_attention,
    }

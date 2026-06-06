"""Dataset utilities for the two retained official-artifact students."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RAGGED_KEYS = (
    "clip_ecr",
    "fusion_hidden",
    "temporal_hidden",
    "frame_fusion_feature",
    "text_tokens",
    "text_pooled",
    "attention_importance",
    "action_feature",
    "caption_feature",
)

DEFAULT_RAGGED_KEYS = RAGGED_KEYS


@dataclass(frozen=True)
class StudentInputConfig:
    """Select the retained student input interface."""

    preset: str
    use_frame_fusion: bool
    use_sound_text: bool = False
    use_title_text: bool = False
    use_description_text: bool = False
    use_text_tokens: bool = False
    use_quality_features: bool = False
    quality_feature_dim: int = 0
    quality_fusion: str = "input_concat"
    use_dover_features: bool = False
    dover_feature_dim: int = 0
    dover_fusion: str = "input_concat"
    use_teacher_compressed_tokens: bool = False
    use_lite_action: bool = False
    lite_action_dim: int = 0

    @classmethod
    def from_preset(cls, preset: str) -> StudentInputConfig:
        presets = {
            "visual_text_sound": cls(
                preset="visual_text_sound",
                use_frame_fusion=True,
                use_sound_text=True,
                use_title_text=True,
                use_description_text=True,
            ),
            "teacher_compressed_tokens": cls(
                preset="teacher_compressed_tokens",
                use_frame_fusion=False,
                use_teacher_compressed_tokens=True,
            ),
        }
        if preset not in presets:
            raise ValueError(f"Unknown student input preset {preset!r}. Choices: {sorted(presets)}")
        return presets[preset]

    def with_text_tokens(self, enabled: bool) -> StudentInputConfig:
        return replace(self, use_text_tokens=enabled)

    def with_quality_features(
        self,
        enabled: bool,
        dim: int = 0,
        fusion: str = "input_concat",
    ) -> StudentInputConfig:
        return replace(
            self,
            use_quality_features=enabled,
            quality_feature_dim=dim if enabled else 0,
            quality_fusion=fusion if enabled else "input_concat",
        )

    def with_dover_features(
        self,
        enabled: bool,
        dim: int = 0,
        fusion: str = "input_concat",
    ) -> StudentInputConfig:
        return replace(
            self,
            use_dover_features=enabled,
            dover_feature_dim=dim if enabled else 0,
            dover_fusion=fusion if enabled else "input_concat",
        )

    def with_lite_action(self, enabled: bool, dim: int = 1152) -> StudentInputConfig:
        return replace(self, use_lite_action=enabled, lite_action_dim=dim if enabled else 0)


def artifact_keys_for_input_config(config: StudentInputConfig) -> tuple[str, ...]:
    keys = {"clip_ecr", "fusion_hidden", "temporal_hidden", "attention_importance"}
    if config.use_frame_fusion:
        keys.add("frame_fusion_feature")
    if config.use_sound_text or config.use_title_text or config.use_description_text:
        keys.add("text_tokens" if config.use_text_tokens else "text_pooled")
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
    ragged_keys: Iterable[str] = DEFAULT_RAGGED_KEYS,
) -> list[dict[str, object]]:
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


def _repeat_or_fit_2d(array: np.ndarray, length: int, dim: int) -> np.ndarray:
    array = _ensure_2d(array, dim)
    if array.shape[0] == 1 and length > 1:
        array = np.repeat(array, length, axis=0)
    return _fit_2d(array, length, dim)


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
    valid = [idx for idx in indices if idx < len(text_pooled)]
    if not valid:
        return np.zeros((0, 768), dtype=np.float32)
    return text_pooled[valid].astype(np.float32, copy=False)


def _select_text_tokens(text_tokens: np.ndarray, config: StudentInputConfig) -> np.ndarray:
    if text_tokens.size == 0:
        return np.zeros((0, 768), dtype=np.float32)
    if text_tokens.ndim == 2:
        text_tokens = text_tokens.reshape(1, *text_tokens.shape)
    text_tokens = text_tokens.reshape(text_tokens.shape[0], text_tokens.shape[1], -1)
    indices = []
    if config.use_sound_text:
        indices.append(0)
    if config.use_title_text:
        indices.append(1)
    if config.use_description_text:
        indices.append(2)
    valid = [idx for idx in indices if idx < len(text_tokens)]
    if not valid:
        return np.zeros((0, 768), dtype=np.float32)
    selected = text_tokens[valid, :, :768]
    return selected.reshape(-1, 768).astype(np.float32, copy=False)


def _stats_token(array: np.ndarray, dim: int) -> np.ndarray:
    array = _ensure_2d(array, dim)
    if array.size == 0:
        return np.zeros((1, dim * 4), dtype=np.float32)
    return np.concatenate(
        [array.mean(axis=0), array.std(axis=0), array.min(axis=0), array.max(axis=0)]
    ).reshape(1, -1).astype(np.float32, copy=False)


def _scalar_stats_token(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((1, 4), dtype=np.float32)
    return np.asarray(
        [values.mean(), values.std(), values.min(), values.max()],
        dtype=np.float32,
    ).reshape(1, -1)


def _build_compressed_teacher_tokens(row: dict[str, object]) -> np.ndarray:
    temporal = _stats_token(row["temporal_hidden"], 512)
    fusion = _stats_token(row["fusion_hidden"], 512)
    clip_ecr = _scalar_stats_token(row["clip_ecr"])
    attention = _scalar_stats_token(np.asarray(row["attention_importance"], dtype=np.float32))
    scalar_stats = np.concatenate([clip_ecr, attention], axis=-1)
    scalar_stats = np.pad(scalar_stats, ((0, 0), (0, temporal.shape[-1] - scalar_stats.shape[-1])))
    return np.concatenate([temporal, fusion, scalar_stats], axis=0).astype(
        np.float32,
        copy=False,
    )


def _attention_vector(row: dict[str, object], length: int) -> np.ndarray:
    teacher_attention = _ensure_2d(row["attention_importance"], length)
    if teacher_attention.size > 0:
        teacher_attention = teacher_attention[:, :length].mean(axis=0)
    else:
        teacher_attention = np.ones((length,), dtype=np.float32) / max(length, 1)
    return _fit_1d(teacher_attention, length)


class OfficialTeacherArtifactDataset(Dataset):
    """Dataset for the retained deployable and upper-bound students."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        input_config: StudentInputConfig,
        *,
        max_clips: int = 16,
        clip_offset: int = 0,
    ):
        self.rows = rows
        self.input_config = input_config
        self.max_clips = max_clips
        self.clip_offset = max(0, clip_offset)
        self.clip_dim = self._infer_clip_dim()

    def _infer_clip_dim(self) -> int:
        for row in self.rows:
            pieces = self._clip_pieces(row)
            if pieces:
                return int(sum(piece.shape[-1] for piece in pieces))
        raise RuntimeError("Could not infer student clip input dimension")

    def _clip_pieces(self, row: dict[str, object]) -> list[np.ndarray]:
        if self.input_config.use_teacher_compressed_tokens:
            return [_build_compressed_teacher_tokens(row)]
        pieces = []
        frame_length = 0
        if self.input_config.use_frame_fusion:
            frame_piece = _ensure_2d(row["frame_fusion_feature"], 1024)
            frame_length = frame_piece.shape[0]
            pieces.append(frame_piece)
        if (
            self.input_config.use_quality_features
            and self.input_config.quality_fusion == "input_concat"
        ):
            pieces.append(
                _repeat_or_fit_2d(
                    row.get(
                        "quality_features",
                        np.zeros((0, self.input_config.quality_feature_dim), dtype=np.float32),
                    ),
                    frame_length,
                    self.input_config.quality_feature_dim,
                )
            )
        if self.input_config.use_dover_features and self.input_config.dover_fusion == "input_concat":
            pieces.append(
                _repeat_or_fit_2d(
                    row.get(
                        "dover_features",
                        np.zeros((0, self.input_config.dover_feature_dim), dtype=np.float32),
                    ),
                    frame_length,
                    self.input_config.dover_feature_dim,
                )
            )
        if self.input_config.use_lite_action:
            pieces.append(
                _repeat_or_fit_2d(
                    row.get(
                        "lite_action_features",
                        np.zeros((0, self.input_config.lite_action_dim), dtype=np.float32),
                    ),
                    frame_length,
                    self.input_config.lite_action_dim,
                )
            )
        return pieces

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        pieces = self._clip_pieces(row)
        if not pieces:
            raise RuntimeError(f"No student clip inputs available for {row['Id']}")
        total_len = min(piece.shape[0] for piece in pieces)
        start = min(self.clip_offset, max(0, total_len - self.max_clips))
        min_len = min(total_len - start, self.max_clips)
        if min_len <= 0:
            raise RuntimeError(f"Zero-length student clip input for {row['Id']}")
        clip_inputs = np.concatenate([piece[start : start + min_len] for piece in pieces], axis=-1)

        if self.input_config.use_text_tokens:
            text_inputs = _select_text_tokens(
                row.get("text_tokens", np.zeros((0,))),
                self.input_config,
            )
        else:
            text_inputs = _select_text_pooled(
                row.get("text_pooled", np.zeros((0,))),
                self.input_config,
            )
        teacher_temporal = _fit_2d(
            _ensure_2d(row["temporal_hidden"], 512)[start : start + min_len],
            min_len,
            512,
        )
        teacher_fusion = _fit_2d(
            _ensure_2d(row["fusion_hidden"], 512)[start : start + min_len],
            min_len,
            512,
        )
        teacher_action = _fit_2d(
            _ensure_2d(row.get("action_feature", np.zeros((0,))), 512)[start : start + min_len],
            min_len,
            512,
        )
        teacher_caption_feature = _fit_2d(
            _ensure_2d(row.get("caption_feature", np.zeros((0,))), 1024)[
                start : start + min_len
            ],
            min_len,
            1024,
        )
        teacher_clip_ecr = _fit_1d(np.asarray(row["clip_ecr"])[start : start + min_len], min_len)
        teacher_attention = _attention_vector(row, total_len)[start : start + min_len]
        dover_inputs = _ensure_2d(
            row.get(
                "dover_features",
                np.zeros((0, self.input_config.dover_feature_dim), dtype=np.float32),
            ),
            self.input_config.dover_feature_dim,
        )
        if dover_inputs.shape[0] > 1:
            dover_inputs = dover_inputs[:1]
        elif dover_inputs.size == 0:
            dover_inputs = np.zeros((1, self.input_config.dover_feature_dim), dtype=np.float32)
        quality_inputs = _repeat_or_fit_2d(
            row.get(
                "quality_features",
                np.zeros((0, self.input_config.quality_feature_dim), dtype=np.float32),
            ),
            total_len,
            self.input_config.quality_feature_dim,
        )[start : start + min_len]

        return {
            "Id": row["Id"],
            "clip_inputs": torch.from_numpy(clip_inputs.astype(np.float32, copy=False)),
            "quality_inputs": torch.from_numpy(quality_inputs.astype(np.float32, copy=False)),
            "dover_inputs": torch.from_numpy(
                dover_inputs.reshape(-1).astype(np.float32, copy=False)
            ),
            "text_inputs": torch.from_numpy(text_inputs.astype(np.float32, copy=False)),
            "ecr_true": torch.tensor(float(row["ecr_true"]), dtype=torch.float32),
            "teacher_ecr": torch.tensor(float(row["teacher_ecr"]), dtype=torch.float32),
            "pseudo_ecr": torch.tensor(
                float(row.get("pseudo_ecr", row["teacher_ecr"])),
                dtype=torch.float32,
            ),
            "teacher_temporal": torch.from_numpy(teacher_temporal.astype(np.float32, copy=False)),
            "teacher_fusion": torch.from_numpy(teacher_fusion.astype(np.float32, copy=False)),
            "teacher_action": torch.from_numpy(teacher_action.astype(np.float32, copy=False)),
            "teacher_caption_feature": torch.from_numpy(
                teacher_caption_feature.astype(np.float32, copy=False)
            ),
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
    dover_dim = int(items[0]["dover_inputs"].numel())
    dover_inputs = torch.zeros(batch_size, dover_dim)
    quality_dim = int(items[0]["quality_inputs"].shape[-1])
    quality_inputs = torch.zeros(batch_size, max_clips, quality_dim)
    teacher_temporal = torch.zeros(batch_size, max_clips, 512)
    teacher_fusion = torch.zeros(batch_size, max_clips, 512)
    teacher_action = torch.zeros(batch_size, max_clips, 512)
    teacher_caption_feature = torch.zeros(batch_size, max_clips, 1024)
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
        if dover_dim:
            dover_inputs[i] = item["dover_inputs"][:dover_dim]
        if quality_dim:
            quality_inputs[i, :n_clips] = item["quality_inputs"][:n_clips]
        teacher_temporal[i, :n_clips] = item["teacher_temporal"][:n_clips]
        teacher_fusion[i, :n_clips] = item["teacher_fusion"][:n_clips]
        teacher_action[i, :n_clips] = item["teacher_action"][:n_clips]
        teacher_caption_feature[i, :n_clips] = item["teacher_caption_feature"][:n_clips]
        teacher_clip_ecr[i, :n_clips] = item["teacher_clip_ecr"][:n_clips]
        attn = item["teacher_attention"][:n_clips]
        teacher_attention[i, :n_clips] = attn / attn.sum().clamp_min(1e-6)

    return {
        "ids": ids,
        "clip_inputs": clip_inputs,
        "clip_mask": clip_mask,
        "text_inputs": text_inputs,
        "text_mask": text_mask,
        "quality_inputs": quality_inputs,
        "dover_inputs": dover_inputs,
        "ecr_true": torch.stack([item["ecr_true"] for item in items]),
        "teacher_ecr": torch.stack([item["teacher_ecr"] for item in items]),
        "pseudo_ecr": torch.stack([item["pseudo_ecr"] for item in items]),
        "teacher_temporal": teacher_temporal,
        "teacher_fusion": teacher_fusion,
        "teacher_action": teacher_action,
        "teacher_caption_feature": teacher_caption_feature,
        "teacher_clip_ecr": teacher_clip_ecr,
        "teacher_attention": teacher_attention,
    }

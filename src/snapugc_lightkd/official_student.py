"""Student baseline and KD model for official SnapUGC teacher artifacts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(1, dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, dim // 2), 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None):
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class OfficialArtifactStudent(nn.Module):
    """Compact student trained from a reduced subset of official teacher artifacts."""

    def __init__(
        self,
        *,
        clip_input_dim: int,
        text_input_dim: int = 768,
        hidden_dim: int = 128,
        teacher_hidden_dim: int = 512,
        max_clips: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_clips, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.temporal_pool = AttentionPool(hidden_dim, dropout)

        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_pool = AttentionPool(hidden_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.clip_ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.hidden_to_teacher = nn.Linear(hidden_dim, teacher_hidden_dim)

    def forward(
        self,
        clip_inputs: torch.Tensor,
        clip_mask: torch.Tensor,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.clip_proj(clip_inputs)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.temporal_encoder(x, src_key_padding_mask=~clip_mask.bool())
        video_hidden, temporal_attention = self.temporal_pool(x, clip_mask)

        if text_inputs.size(1) > 0:
            text_tokens = self.text_proj(text_inputs)
            text_hidden, text_attention = self.text_pool(text_tokens, text_mask)
        else:
            text_hidden = torch.zeros_like(video_hidden)
            text_attention = torch.zeros(text_inputs.size(0), 0, device=text_inputs.device)

        fused_hidden = self.fusion(torch.cat([video_hidden, text_hidden], dim=-1))
        ecr = self.ecr_head(fused_hidden).squeeze(-1)
        clip_ecr = self.clip_ecr_head(x).squeeze(-1)
        teacher_space_temporal = self.hidden_to_teacher(x)
        teacher_space_hidden = self.hidden_to_teacher(fused_hidden)
        return {
            "predicted_ecr": ecr,
            "clip_ecr": clip_ecr,
            "student_temporal": x,
            "student_hidden": fused_hidden,
            "teacher_space_temporal": teacher_space_temporal,
            "teacher_space_hidden": teacher_space_hidden,
            "temporal_attention": temporal_attention,
            "text_attention": text_attention,
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    diff = (pred - target) ** 2
    diff = diff.masked_select(mask.expand_as(diff))
    if diff.numel() == 0:
        return pred.sum() * 0.0
    return diff.mean()


def masked_representation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "raw_mse",
) -> torch.Tensor:
    mask = mask.bool()
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    valid = mask.expand_as(pred)
    if valid.sum() == 0:
        return pred.sum() * 0.0

    if mode == "raw_mse":
        diff = (pred - target) ** 2
        return diff.masked_select(valid).mean()
    if mode == "normalized_mse":
        pred = F.layer_norm(pred, pred.shape[-1:])
        target = F.layer_norm(target, target.shape[-1:])
        diff = (pred - target) ** 2
        return diff.masked_select(valid).mean()
    if mode == "cosine":
        token_mask = mask.squeeze(-1)
        loss = 1.0 - F.cosine_similarity(pred, target, dim=-1)
        loss = loss.masked_select(token_mask)
        return loss.mean() if loss.numel() else pred.sum() * 0.0
    raise ValueError(f"Unknown representation loss mode: {mode}")


def attention_kl(student_attention: torch.Tensor, teacher_attention: torch.Tensor, mask: torch.Tensor):
    mask = mask.bool()
    student = student_attention.masked_fill(~mask, 0.0)
    teacher = teacher_attention.masked_fill(~mask, 0.0)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div((student + 1e-6).log(), teacher, reduction="batchmean")


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float = 0.05,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Batch pairwise logistic ranking loss for ECR order distillation."""

    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(0) - target.unsqueeze(1)
    pair_mask = target_diff.abs() > margin
    if pair_mask.sum() == 0:
        return pred.sum() * 0.0
    signs = target_diff.sign()
    logits = signs * pred_diff / max(temperature, 1e-6)
    loss = F.softplus(-logits)
    return loss.masked_select(pair_mask).mean()


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    use_kd: bool,
    weights: dict[str, float],
    repr_loss: str = "raw_mse",
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = F.mse_loss(outputs["predicted_ecr"], batch["ecr_true"])
    losses = {"hard_ecr": hard}
    total = weights.get("hard_ecr", 1.0) * hard

    hard_rank_weight = weights.get("hard_rank", 0.0)
    if hard_rank_weight:
        hard_rank = pairwise_rank_loss(outputs["predicted_ecr"], batch["ecr_true"])
        losses["hard_rank"] = hard_rank
        total = total + hard_rank_weight * hard_rank

    if use_kd:
        soft = F.mse_loss(outputs["predicted_ecr"], batch["teacher_ecr"])
        clip = masked_mse(outputs["clip_ecr"], batch["teacher_clip_ecr"], batch["clip_mask"])
        temporal = masked_representation_loss(
            outputs["teacher_space_temporal"],
            batch["teacher_temporal"],
            batch["clip_mask"],
            mode=repr_loss,
        )
        clip_mask = batch["clip_mask"].unsqueeze(-1).float()
        fusion_target = (batch["teacher_fusion"] * clip_mask).sum(dim=1) / clip_mask.sum(
            dim=1
        ).clamp_min(1.0)
        if repr_loss == "raw_mse":
            hidden = F.mse_loss(outputs["teacher_space_hidden"], fusion_target)
        elif repr_loss == "normalized_mse":
            hidden = F.mse_loss(
                F.layer_norm(outputs["teacher_space_hidden"], outputs["teacher_space_hidden"].shape[-1:]),
                F.layer_norm(fusion_target, fusion_target.shape[-1:]),
            )
        elif repr_loss == "cosine":
            hidden = (1.0 - F.cosine_similarity(outputs["teacher_space_hidden"], fusion_target, dim=-1)).mean()
        else:
            raise ValueError(f"Unknown representation loss mode: {repr_loss}")
        attn = attention_kl(
            outputs["temporal_attention"],
            batch["teacher_attention"],
            batch["clip_mask"],
        )
        teacher_rank_weight = weights.get("teacher_rank", 0.0)
        if teacher_rank_weight:
            teacher_rank = pairwise_rank_loss(outputs["predicted_ecr"], batch["teacher_ecr"])
        else:
            teacher_rank = None
        losses.update(
            {
                "soft_ecr": soft,
                "clip_ecr": clip,
                "temporal_hidden": temporal,
                "fusion_hidden": hidden,
                "attention": attn,
            }
        )
        if teacher_rank is not None:
            losses["teacher_rank"] = teacher_rank
        total = (
            total
            + weights.get("soft_ecr", 0.5) * soft
            + weights.get("clip_ecr", 0.2) * clip
            + weights.get("temporal_hidden", 0.2) * temporal
            + weights.get("fusion_hidden", 0.1) * hidden
            + weights.get("attention", 0.05) * attn
        )
        if teacher_rank is not None:
            total = total + teacher_rank_weight * teacher_rank
    losses["total"] = total
    return total, {key: float(value.detach().cpu()) for key, value in losses.items()}

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


def attention_kl(student_attention: torch.Tensor, teacher_attention: torch.Tensor, mask: torch.Tensor):
    mask = mask.bool()
    student = student_attention.masked_fill(~mask, 0.0)
    teacher = teacher_attention.masked_fill(~mask, 0.0)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div((student + 1e-6).log(), teacher, reduction="batchmean")


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    use_kd: bool,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = F.mse_loss(outputs["predicted_ecr"], batch["ecr_true"])
    losses = {"hard_ecr": hard}
    total = weights.get("hard_ecr", 1.0) * hard

    if use_kd:
        soft = F.mse_loss(outputs["predicted_ecr"], batch["teacher_ecr"])
        clip = masked_mse(outputs["clip_ecr"], batch["teacher_clip_ecr"], batch["clip_mask"])
        temporal = masked_mse(
            outputs["teacher_space_temporal"],
            batch["teacher_temporal"],
            batch["clip_mask"],
        )
        clip_mask = batch["clip_mask"].unsqueeze(-1).float()
        fusion_target = (batch["teacher_fusion"] * clip_mask).sum(dim=1) / clip_mask.sum(
            dim=1
        ).clamp_min(1.0)
        hidden = F.mse_loss(outputs["teacher_space_hidden"], fusion_target)
        attn = attention_kl(
            outputs["temporal_attention"],
            batch["teacher_attention"],
            batch["clip_mask"],
        )
        losses.update(
            {
                "soft_ecr": soft,
                "clip_ecr": clip,
                "temporal_hidden": temporal,
                "fusion_hidden": hidden,
                "attention": attn,
            }
        )
        total = (
            total
            + weights.get("soft_ecr", 0.5) * soft
            + weights.get("clip_ecr", 0.2) * clip
            + weights.get("temporal_hidden", 0.2) * temporal
            + weights.get("fusion_hidden", 0.1) * hidden
            + weights.get("attention", 0.05) * attn
        )
    losses["total"] = total
    return total, {key: float(value.detach().cpu()) for key, value in losses.items()}

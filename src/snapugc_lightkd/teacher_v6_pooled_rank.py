"""Teacher v6: pooled feature MLP with pairwise ranking regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ResidualBlock


def masked_mean(tokens, mask=None):
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def pairwise_rank_loss(pred, target, margin=0.0):
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    sign = torch.sign(diff_target)
    valid = sign != 0
    if not valid.any():
        return pred.new_tensor(0.0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    # If target_i > target_j, pred_i should be > pred_j.
    losses = F.softplus(-(sign * diff_pred - margin))
    return losses[valid].mean()


class TeacherV6PooledRank(nn.Module):
    """Low-data teacher: pooled modalities, gated fusion, optional rank loss."""

    def __init__(
        self,
        clip_dim=512,
        motion_dim=512,
        audio_dim=1024,
        text_dim=768,
        quality_dim=3,
        hidden_dim=384,
        n_blocks=2,
        n_heads=1,
        max_frames=16,
        max_motion_clips=4,
        dropout=0.35,
        text_drop_prob=0.15,
        quality_drop_prob=0.10,
        aux_weight=0.10,
        rank_weight=0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_drop_prob = float(text_drop_prob)
        self.quality_drop_prob = float(quality_drop_prob)
        self.aux_weight = float(aux_weight)
        self.rank_weight = float(rank_weight)

        def projector(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.visual_proj = projector(clip_dim)
        self.motion_proj = projector(motion_dim)
        self.audio_proj = projector(audio_dim)
        self.text_proj = projector(text_dim)
        self.caption_proj = projector(text_dim)
        self.rationale_proj = projector(text_dim)
        self.quality_proj = projector(quality_dim)

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 7),
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 7),
            nn.Linear(hidden_dim * 7, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(max(1, n_blocks))]
        )
        self.ecr_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.aesthetic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.technical_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def _drop_inputs(self, caption_emb, rationale_emb, quality_scores):
        if not self.training:
            return caption_emb, rationale_emb, quality_scores
        if torch.rand((), device=caption_emb.device).item() < self.text_drop_prob:
            caption_emb = torch.zeros_like(caption_emb)
            rationale_emb = torch.zeros_like(rationale_emb)
        if torch.rand((), device=quality_scores.device).item() < self.quality_drop_prob:
            quality_scores = torch.zeros_like(quality_scores)
        return caption_emb, rationale_emb, quality_scores

    def forward(
        self,
        clip_frames,
        audio_emb,
        text_emb,
        quality_scores,
        motion_clips=None,
        caption_emb=None,
        rationale_emb=None,
        clip_mask=None,
        motion_mask=None,
        ecr_targets=None,
        aesthetic_targets=None,
        technical_targets=None,
    ):
        bsz = clip_frames.size(0)
        device = clip_frames.device
        if motion_clips is None:
            motion_clips = torch.zeros(bsz, 1, 512, device=device)
        if caption_emb is None:
            caption_emb = torch.zeros_like(text_emb)
        if rationale_emb is None:
            rationale_emb = torch.zeros_like(text_emb)

        caption_emb, rationale_emb, quality_scores = self._drop_inputs(
            caption_emb, rationale_emb, quality_scores
        )

        visual = self.visual_proj(masked_mean(clip_frames, clip_mask))
        motion = self.motion_proj(masked_mean(motion_clips, motion_mask))
        audio = self.audio_proj(audio_emb)
        text = self.text_proj(text_emb)
        caption = self.caption_proj(caption_emb)
        rationale = self.rationale_proj(rationale_emb)
        quality = self.quality_proj(quality_scores)

        tokens = torch.stack([visual, motion, audio, text, caption, rationale, quality], dim=1)
        flat = tokens.reshape(bsz, -1)
        gates = self.gate(flat)
        hidden = self.fusion((tokens * gates.unsqueeze(-1)).reshape(bsz, -1))
        hidden = self.post(hidden)

        predicted_ecr = self.ecr_head(hidden).squeeze(-1)
        predicted_aesthetic = self.aesthetic_head(hidden).squeeze(-1)
        predicted_technical = self.technical_head(hidden).squeeze(-1)

        outputs = {
            "predicted_ecr": predicted_ecr,
            "predicted_aesthetic": predicted_aesthetic,
            "predicted_technical": predicted_technical,
            "hidden": hidden,
            "temporal_attention": clip_mask.float() / clip_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            if clip_mask is not None
            else torch.full((bsz, clip_frames.size(1)), 1.0 / clip_frames.size(1), device=device),
            "motion_attention": motion_mask.float() / motion_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            if motion_mask is not None
            else torch.full((bsz, motion_clips.size(1)), 1.0 / motion_clips.size(1), device=device),
            "modality_attention": gates,
        }

        if ecr_targets is not None:
            loss = F.mse_loss(predicted_ecr, ecr_targets)
            loss = loss + self.rank_weight * pairwise_rank_loss(predicted_ecr, ecr_targets)
            if aesthetic_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(predicted_aesthetic, aesthetic_targets)
            if technical_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(predicted_technical, technical_targets)
            outputs["loss"] = loss

        return outputs

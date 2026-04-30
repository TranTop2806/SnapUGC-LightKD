"""Teacher v5: gated late-fusion teacher.

This version removes modality self-attention from the teacher and uses a gated
late-fusion MLP. It is intentionally conservative for the 4k-train regime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AttentionPool, ResidualBlock, TemporalEncoder


class TeacherV5GatedLateFusion(nn.Module):
    """Low-data teacher with per-modality projections and learned gates."""

    def __init__(
        self,
        clip_dim=512,
        motion_dim=512,
        audio_dim=1024,
        text_dim=768,
        quality_dim=3,
        hidden_dim=256,
        n_blocks=1,
        n_heads=4,
        max_frames=16,
        max_motion_clips=4,
        dropout=0.35,
        caption_drop_prob=0.25,
        rationale_drop_prob=0.25,
        quality_drop_prob=0.10,
        aux_weight=0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.caption_drop_prob = float(caption_drop_prob)
        self.rationale_drop_prob = float(rationale_drop_prob)
        self.quality_drop_prob = float(quality_drop_prob)
        self.aux_weight = float(aux_weight)

        self.visual_encoder = TemporalEncoder(
            clip_dim,
            hidden_dim,
            n_layers=1,
            n_heads=n_heads,
            max_tokens=max_frames,
            dropout=dropout,
        )
        self.motion_encoder = TemporalEncoder(
            motion_dim,
            hidden_dim,
            n_layers=1,
            n_heads=max(1, n_heads // 2),
            max_tokens=max_motion_clips,
            dropout=dropout,
        )

        def projector(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.audio_proj = projector(audio_dim)
        self.shared_text_proj = projector(text_dim)
        self.text_pool = AttentionPool(hidden_dim, dropout)
        self.quality_proj = projector(quality_dim)

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
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

    def _regularize_inputs(self, caption_emb, rationale_emb, quality_scores):
        if not self.training:
            return caption_emb, rationale_emb, quality_scores
        if torch.rand((), device=caption_emb.device).item() < self.caption_drop_prob:
            caption_emb = torch.zeros_like(caption_emb)
        if torch.rand((), device=rationale_emb.device).item() < self.rationale_drop_prob:
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

        caption_emb, rationale_emb, quality_scores = self._regularize_inputs(
            caption_emb, rationale_emb, quality_scores
        )

        visual_token, _, temporal_attn = self.visual_encoder(clip_frames, clip_mask)
        motion_token, _, motion_attn = self.motion_encoder(motion_clips, motion_mask)
        audio_token = self.audio_proj(audio_emb)
        quality_token = self.quality_proj(quality_scores)

        text_tokens = torch.stack(
            [
                self.shared_text_proj(text_emb),
                self.shared_text_proj(caption_emb),
                self.shared_text_proj(rationale_emb),
            ],
            dim=1,
        )
        text_token, text_attn = self.text_pool(text_tokens)

        modality_tokens = torch.stack(
            [visual_token, motion_token, audio_token, text_token, quality_token],
            dim=1,
        )
        flat = modality_tokens.reshape(bsz, -1)
        gate_weights = self.gate(flat)
        gated_tokens = modality_tokens * gate_weights.unsqueeze(-1)
        hidden = self.fusion(gated_tokens.reshape(bsz, -1))
        hidden = self.post(hidden)

        predicted_ecr = self.ecr_head(hidden).squeeze(-1)
        predicted_aesthetic = self.aesthetic_head(hidden).squeeze(-1)
        predicted_technical = self.technical_head(hidden).squeeze(-1)

        outputs = {
            "predicted_ecr": predicted_ecr,
            "predicted_aesthetic": predicted_aesthetic,
            "predicted_technical": predicted_technical,
            "hidden": hidden,
            "temporal_attention": temporal_attn,
            "motion_attention": motion_attn,
            "modality_attention": gate_weights,
            "text_attention": text_attn,
        }

        if ecr_targets is not None:
            loss = F.mse_loss(predicted_ecr, ecr_targets)
            if aesthetic_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(predicted_aesthetic, aesthetic_targets)
            if technical_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(predicted_technical, technical_targets)
            outputs["loss"] = loss

        return outputs

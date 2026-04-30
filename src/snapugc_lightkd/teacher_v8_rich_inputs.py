"""Teacher v8: regularized teacher with richer audio and numeric metadata.

This keeps the stable v3 recipe, but uses two feature groups that were already
extracted and previously unused by the neural teacher:
  - YAMNet AudioSet class probabilities
  - simple numeric metadata such as duration, fps, resolution, aspect, text length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AttentionPool, ResidualBlock, TemporalEncoder


class TeacherV8RichInputs(nn.Module):
    """Reduced-capacity teacher with compact rich side-channel tokens."""

    def __init__(
        self,
        clip_dim=512,
        motion_dim=512,
        audio_dim=1024,
        audio_probs_dim=521,
        text_dim=768,
        quality_dim=3,
        numeric_dim=8,
        hidden_dim=384,
        n_blocks=1,
        n_heads=6,
        max_frames=16,
        max_motion_clips=4,
        dropout=0.35,
        text_drop_prob=0.20,
        quality_drop_prob=0.10,
        motion_drop_prob=0.10,
        audio_probs_drop_prob=0.10,
        numeric_drop_prob=0.10,
        aux_weight=0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_drop_prob = float(text_drop_prob)
        self.quality_drop_prob = float(quality_drop_prob)
        self.motion_drop_prob = float(motion_drop_prob)
        self.audio_probs_drop_prob = float(audio_probs_drop_prob)
        self.numeric_drop_prob = float(numeric_drop_prob)
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
        self.audio_probs_proj = projector(audio_probs_dim)
        self.text_proj = projector(text_dim)
        self.caption_proj = projector(text_dim)
        self.rationale_proj = projector(text_dim)
        self.quality_proj = projector(quality_dim)
        self.numeric_proj = projector(numeric_dim)

        self.modality_embed = nn.Parameter(torch.zeros(1, 9, hidden_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=n_blocks)
        self.modality_pool = AttentionPool(hidden_dim, dropout)
        self.post = nn.Sequential(ResidualBlock(hidden_dim, dropout))

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

    def _maybe_zero(self, value, prob):
        if self.training and torch.rand((), device=value.device).item() < prob:
            return torch.zeros_like(value)
        return value

    def forward(
        self,
        clip_frames,
        audio_emb,
        text_emb,
        quality_scores,
        motion_clips=None,
        caption_emb=None,
        rationale_emb=None,
        audio_probs=None,
        numeric_features=None,
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
        if audio_probs is None:
            audio_probs = torch.zeros(bsz, 521, device=device)
        if numeric_features is None:
            numeric_features = torch.zeros(bsz, 8, device=device)

        if self.training and torch.rand((), device=device).item() < self.text_drop_prob:
            caption_emb = torch.zeros_like(caption_emb)
            rationale_emb = torch.zeros_like(rationale_emb)
        motion_clips = self._maybe_zero(motion_clips, self.motion_drop_prob)
        quality_scores = self._maybe_zero(quality_scores, self.quality_drop_prob)
        audio_probs = self._maybe_zero(audio_probs, self.audio_probs_drop_prob)
        numeric_features = self._maybe_zero(numeric_features, self.numeric_drop_prob)

        visual_token, _, temporal_attn = self.visual_encoder(clip_frames, clip_mask)
        motion_token, _, motion_attn = self.motion_encoder(motion_clips, motion_mask)
        audio_token = self.audio_proj(audio_emb)
        audio_probs_token = self.audio_probs_proj(audio_probs)
        text_token = self.text_proj(text_emb)
        caption_token = self.caption_proj(caption_emb)
        rationale_token = self.rationale_proj(rationale_emb)
        quality_token = self.quality_proj(quality_scores)
        numeric_token = self.numeric_proj(numeric_features)

        tokens = torch.stack(
            [
                visual_token,
                motion_token,
                audio_token,
                audio_probs_token,
                text_token,
                caption_token,
                rationale_token,
                quality_token,
                numeric_token,
            ],
            dim=1,
        )
        tokens = tokens + self.modality_embed[:, : tokens.size(1), :]
        fused_tokens = self.fusion(tokens)
        hidden, modality_attn = self.modality_pool(fused_tokens)
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
            "modality_attention": modality_attn,
        }

        if ecr_targets is not None:
            loss = F.mse_loss(predicted_ecr, ecr_targets)
            if aesthetic_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(predicted_aesthetic, aesthetic_targets)
            if technical_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(predicted_technical, technical_targets)
            outputs["loss"] = loss

        return outputs

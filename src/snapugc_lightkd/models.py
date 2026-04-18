"""Temporal teacher/student models for KD v2 final pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ================================================================
# FINAL PAPER-ALIGNED TEMPORAL KD MODELS
# ================================================================
class AttentionPool(nn.Module):
    """Learned attention pooling for temporal or modality tokens."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, tokens, mask=None):
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TemporalEncoder(nn.Module):
    """Small Transformer encoder over sampled frame/clip tokens."""

    def __init__(self, input_dim, hidden_dim, n_layers=2, n_heads=4,
                 max_tokens=32, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = AttentionPool(hidden_dim, dropout)

    def forward(self, tokens, mask=None):
        x = self.input_proj(tokens)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x, src_key_padding_mask=(~mask.bool()) if mask is not None else None)
        pooled, attn = self.pool(x, mask)
        return pooled, x, attn


class TeacherModel(nn.Module):
    """
    Paper-aligned teacher for the final extraction schema.

    Inputs:
      - CLIP frame tokens
      - R(2+1)D motion clip tokens
      - YAMNet audio embedding
      - Sentence-T5 metadata/caption/rationale embeddings
      - DOVER technical/aesthetic/overall scores

    The teacher intentionally uses privileged features that the student does
    not receive, making the KD setup defensible.
    """

    def __init__(self, clip_dim=512, motion_dim=512, audio_dim=1024,
                 text_dim=768, quality_dim=3, hidden_dim=512,
                 n_blocks=2, n_heads=8, max_frames=16, max_motion_clips=4,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.visual_encoder = TemporalEncoder(
            clip_dim, hidden_dim, n_layers=n_blocks, n_heads=n_heads,
            max_tokens=max_frames, dropout=dropout,
        )
        self.motion_encoder = TemporalEncoder(
            motion_dim, hidden_dim, n_layers=1, n_heads=max(1, n_heads // 2),
            max_tokens=max_motion_clips, dropout=dropout,
        )

        def projector(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.audio_proj = projector(audio_dim)
        self.text_proj = projector(text_dim)
        self.caption_proj = projector(text_dim)
        self.rationale_proj = projector(text_dim)
        self.quality_proj = projector(quality_dim)

        self.modality_embed = nn.Parameter(torch.zeros(1, 7, hidden_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=n_blocks)
        self.modality_pool = AttentionPool(hidden_dim, dropout)

        self.post = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(2)]
        )
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.aesthetic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.technical_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, clip_frames, audio_emb, text_emb, quality_scores,
                motion_clips=None, caption_emb=None, rationale_emb=None,
                clip_mask=None, motion_mask=None, ecr_targets=None):
        bsz = clip_frames.size(0)
        device = clip_frames.device

        if motion_clips is None:
            motion_clips = torch.zeros(bsz, 1, 512, device=device)
        if caption_emb is None:
            caption_emb = torch.zeros_like(text_emb)
        if rationale_emb is None:
            rationale_emb = torch.zeros_like(text_emb)

        visual_token, _, temporal_attn = self.visual_encoder(clip_frames, clip_mask)
        motion_token, _, motion_attn = self.motion_encoder(motion_clips, motion_mask)
        audio_token = self.audio_proj(audio_emb)
        text_token = self.text_proj(text_emb)
        caption_token = self.caption_proj(caption_emb)
        rationale_token = self.rationale_proj(rationale_emb)
        quality_token = self.quality_proj(quality_scores)

        tokens = torch.stack([
            visual_token, motion_token, audio_token, text_token,
            caption_token, rationale_token, quality_token,
        ], dim=1)
        tokens = tokens + self.modality_embed[:, :tokens.size(1), :]
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
            outputs["loss"] = F.mse_loss(predicted_ecr, ecr_targets)

        return outputs


class StudentModel(nn.Module):
    """
    Lightweight temporal student.

    Student uses only deployable lightweight inputs:
      - CLIP frame tokens
      - YAMNet audio embedding
      - Sentence-T5 metadata embedding

    It learns quality heads and teacher attention via KD, but does not receive
    DOVER/motion/Qwen features directly.
    """

    def __init__(self, clip_dim=512, audio_dim=1024, text_dim=768,
                 hidden_dim=256, teacher_hidden_dim=512, n_heads=4,
                 max_frames=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.visual_encoder = TemporalEncoder(
            clip_dim, hidden_dim, n_layers=2, n_heads=n_heads,
            max_tokens=max_frames, dropout=dropout,
        )

        def projector(dim):
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.audio_proj = projector(audio_dim)
        self.text_proj = projector(text_dim)
        self.modality_embed = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=2)
        self.modality_pool = AttentionPool(hidden_dim, dropout)
        self.post = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )

        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.aesthetic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.technical_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.kd_projector = nn.Sequential(
            nn.Linear(hidden_dim, teacher_hidden_dim),
            nn.LayerNorm(teacher_hidden_dim),
        )

    def forward(self, clip_frames, audio_emb, text_emb, clip_mask=None,
                ecr_targets=None, aesthetic_targets=None, technical_targets=None,
                teacher_ecr=None, teacher_hidden=None, teacher_temporal_attention=None,
                loss_weights=None):
        if loss_weights is None:
            loss_weights = {
                "ecr_hard": 1.0, "ecr_soft": 0.5, "kd_repr": 0.3,
                "temporal_attn": 0.1, "aesthetic": 0.2, "technical": 0.2,
            }

        visual_token, _, temporal_attn = self.visual_encoder(clip_frames, clip_mask)
        audio_token = self.audio_proj(audio_emb)
        text_token = self.text_proj(text_emb)

        tokens = torch.stack([visual_token, audio_token, text_token], dim=1)
        tokens = tokens + self.modality_embed[:, :tokens.size(1), :]
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
            "fused_hidden": hidden,
            "temporal_attention": temporal_attn,
            "modality_attention": modality_attn,
        }

        loss = torch.tensor(0.0, device=clip_frames.device)
        losses = {}

        if ecr_targets is not None:
            l = F.mse_loss(predicted_ecr, ecr_targets)
            losses["ecr_hard"] = l
            loss = loss + loss_weights.get("ecr_hard", 1.0) * l

        if teacher_ecr is not None:
            l = F.mse_loss(predicted_ecr, teacher_ecr)
            losses["ecr_soft"] = l
            loss = loss + loss_weights.get("ecr_soft", 0.0) * l

        if teacher_hidden is not None:
            projected = self.kd_projector(hidden)
            l = 1.0 - F.cosine_similarity(projected, teacher_hidden, dim=-1).mean()
            losses["kd_repr"] = l
            loss = loss + loss_weights.get("kd_repr", 0.0) * l

        if teacher_temporal_attention is not None:
            target = teacher_temporal_attention[:, :temporal_attn.size(1)]
            target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
            pred = temporal_attn[:, :target.size(1)]
            pred = pred / (pred.sum(dim=-1, keepdim=True) + 1e-8)
            l = F.kl_div((pred + 1e-8).log(), target, reduction="batchmean")
            losses["temporal_attn"] = l
            loss = loss + loss_weights.get("temporal_attn", 0.0) * l

        if aesthetic_targets is not None:
            l = F.mse_loss(predicted_aesthetic, aesthetic_targets)
            losses["aesthetic"] = l
            loss = loss + loss_weights.get("aesthetic", 0.0) * l

        if technical_targets is not None:
            l = F.mse_loss(predicted_technical, technical_targets)
            losses["technical"] = l
            loss = loss + loss_weights.get("technical", 0.0) * l

        outputs["loss"] = loss
        outputs["losses"] = losses
        return outputs


FinalTeacherModel = TeacherModel
FinalStudentModel = StudentModel

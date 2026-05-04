"""Teacher v14-v17: V8-style teachers with selected input groups removed.

These variants test whether opening-5s features benefit from a smaller fusion
set. The removed modalities are not inserted as zero tokens; they are excluded
from the Transformer token sequence.
"""

import torch
import torch.nn.functional as F

from .teacher_v8_rich_inputs import TeacherV8RichInputs


class _SelectiveInputTeacher(TeacherV8RichInputs):
    token_names = (
        "visual",
        "motion",
        "audio",
        "audio_probs",
        "text",
        "caption",
        "rationale",
        "quality",
        "numeric",
    )
    include_tokens = token_names

    def _sample_zero(self, value, prob):
        if not self.training or prob <= 0:
            return value
        keep_shape = [value.size(0)] + [1] * (value.ndim - 1)
        keep = torch.rand(keep_shape, device=value.device) >= prob
        return value * keep.to(value.dtype)

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

        if self.training and self.text_drop_prob > 0:
            keep = (torch.rand((bsz, 1), device=device) >= self.text_drop_prob).to(text_emb.dtype)
            caption_emb = caption_emb * keep
            rationale_emb = rationale_emb * keep
        motion_clips = self._sample_zero(motion_clips, self.motion_drop_prob)
        quality_scores = self._sample_zero(quality_scores, self.quality_drop_prob)
        audio_probs = self._sample_zero(audio_probs, self.audio_probs_drop_prob)
        numeric_features = self._sample_zero(numeric_features, self.numeric_drop_prob)

        visual_token, _, temporal_attn = self.visual_encoder(clip_frames, clip_mask)
        motion_token, _, motion_attn = self.motion_encoder(motion_clips, motion_mask)
        token_map = {
            "visual": visual_token,
            "motion": motion_token,
            "audio": self.audio_proj(audio_emb),
            "audio_probs": self.audio_probs_proj(audio_probs),
            "text": self.text_proj(text_emb),
            "caption": self.caption_proj(caption_emb),
            "rationale": self.rationale_proj(rationale_emb),
            "quality": self.quality_proj(quality_scores),
            "numeric": self.numeric_proj(numeric_features),
        }
        token_index = {name: idx for idx, name in enumerate(self.token_names)}
        tokens = torch.stack([token_map[name] for name in self.include_tokens], dim=1)
        embed_indices = [token_index[name] for name in self.include_tokens]
        tokens = tokens + self.modality_embed[:, embed_indices, :]

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


class TeacherV14NoGeneratedText(_SelectiveInputTeacher):
    """Drop BLIP caption and rationale tokens; keep measured modalities."""

    include_tokens = ("visual", "motion", "audio", "audio_probs", "text", "quality", "numeric")


class TeacherV15OpeningCore(_SelectiveInputTeacher):
    """Drop generated text and YAMNet class probabilities."""

    include_tokens = ("visual", "motion", "audio", "text", "quality", "numeric")


class TeacherV16VisualTextQuality(_SelectiveInputTeacher):
    """Keep visual/motion, metadata text, DOVER quality, and numeric metadata only."""

    include_tokens = ("visual", "motion", "text", "quality", "numeric")


class TeacherV17NoQuality(_SelectiveInputTeacher):
    """Drop generated text, DOVER quality, and numeric metadata."""

    include_tokens = ("visual", "motion", "audio", "audio_probs", "text")

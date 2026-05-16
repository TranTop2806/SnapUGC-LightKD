"""Compact student model and KD losses for official SnapUGC artifacts."""

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

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(tokens).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


def _identity_temporal_conv(
    channels: int,
    *,
    depthwise: bool,
) -> nn.Conv1d:
    groups = channels if depthwise else 1
    conv = nn.Conv1d(
        channels,
        channels,
        kernel_size=3,
        padding=1,
        groups=groups,
    )
    with torch.no_grad():
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
        if depthwise:
            conv.weight[:, 0, 1] = 1.0
        else:
            idx = torch.arange(channels)
            conv.weight[idx, idx, 1] = 1.0
    return conv


class OfficialArtifactStudent(nn.Module):
    """Small source-aware student used for both deployable and upper-bound runs."""

    def __init__(
        self,
        *,
        clip_input_dim: int,
        text_input_dim: int = 768,
        hidden_dim: int = 96,
        teacher_hidden_dim: int = 512,
        max_clips: int = 16,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.22,
        fusion_mode: str = "concat",
        projection_head: str = "linear",
        use_hallucination: bool = False,
        temporal_conv: str = "none",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.use_hallucination = use_hallucination
        if fusion_mode not in {"concat", "cross_attention"}:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        if projection_head not in {"linear", "mlp"}:
            raise ValueError(f"Unknown projection_head: {projection_head}")
        if temporal_conv not in {"none", "depthwise", "full"}:
            raise ValueError(f"Unknown temporal_conv: {temporal_conv}")

        if temporal_conv == "depthwise":
            self.temporal_conv = _identity_temporal_conv(clip_input_dim, depthwise=True)
        elif temporal_conv == "full":
            self.temporal_conv = _identity_temporal_conv(clip_input_dim, depthwise=False)
        else:
            self.temporal_conv = None
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
        self.text_type_embed = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        self.text_pool = AttentionPool(hidden_dim, dropout)

        if fusion_mode == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)
            self.cross_pool = AttentionPool(hidden_dim, dropout)
        else:
            self.cross_attention = None
            self.cross_norm = None
            self.cross_pool = None

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
        if projection_head == "linear":
            self.hidden_to_teacher = nn.Linear(hidden_dim, teacher_hidden_dim)
        else:
            self.hidden_to_teacher = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, teacher_hidden_dim),
            )
        if use_hallucination:
            self.action_hallucination_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 512),
            )
            self.caption_hallucination_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 1024),
            )
        else:
            self.action_hallucination_head = None
            self.caption_hallucination_head = None

    @staticmethod
    def _fit_type_embed(type_embed: torch.Tensor, length: int) -> torch.Tensor:
        if length <= type_embed.size(1):
            return type_embed[:, :length, :]
        repeats = (length + type_embed.size(1) - 1) // type_embed.size(1)
        return type_embed.repeat(1, repeats, 1)[:, :length, :]

    def forward(
        self,
        clip_inputs: torch.Tensor,
        clip_mask: torch.Tensor,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.temporal_conv is not None:
            conv_inputs = clip_inputs.transpose(1, 2)
            clip_inputs = self.temporal_conv(conv_inputs).transpose(1, 2)
        clip_hidden = self.clip_proj(clip_inputs)
        clip_hidden = clip_hidden + self.pos_embed[:, : clip_hidden.size(1), :]
        clip_hidden = self.temporal_encoder(
            clip_hidden,
            src_key_padding_mask=~clip_mask.bool(),
        )
        video_hidden, temporal_attention = self.temporal_pool(clip_hidden, clip_mask)

        if text_inputs.size(1) > 0:
            text_tokens = self.text_proj(text_inputs)
            text_tokens = text_tokens + self._fit_type_embed(
                self.text_type_embed,
                text_tokens.size(1),
            )
            text_hidden, text_attention = self.text_pool(text_tokens, text_mask)
        else:
            text_hidden = torch.zeros_like(video_hidden)
            text_attention = torch.zeros(text_inputs.size(0), 0, device=text_inputs.device)

        if self.fusion_mode == "cross_attention" and text_inputs.size(1) > 0:
            attended_clip_hidden, _ = self.cross_attention(
                query=clip_hidden,
                key=text_tokens,
                value=text_tokens,
                key_padding_mask=~text_mask.bool(),
                need_weights=False,
            )
            cross_hidden = self.cross_norm(clip_hidden + attended_clip_hidden)
            video_hidden, temporal_attention = self.cross_pool(cross_hidden, clip_mask)
        else:
            cross_hidden = clip_hidden

        fused_hidden = self.fusion(torch.cat([video_hidden, text_hidden], dim=-1))
        predicted_ecr = self.ecr_head(fused_hidden).squeeze(-1)
        clip_ecr = self.clip_ecr_head(cross_hidden).squeeze(-1)
        outputs = {
            "predicted_ecr": predicted_ecr,
            "clip_ecr": clip_ecr,
            "student_temporal": cross_hidden,
            "student_hidden": fused_hidden,
            "teacher_space_temporal": self.hidden_to_teacher(cross_hidden),
            "teacher_space_hidden": self.hidden_to_teacher(fused_hidden),
            "temporal_attention": temporal_attention,
            "text_attention": text_attention,
        }
        if self.use_hallucination:
            outputs["pred_action_feature"] = self.action_hallucination_head(cross_hidden)
            outputs["pred_caption_feature"] = self.caption_hallucination_head(cross_hidden)
        return outputs


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
    mode: str = "cosine",
) -> torch.Tensor:
    mask = mask.bool()
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    valid = mask.expand_as(pred)
    if valid.sum() == 0:
        return pred.sum() * 0.0

    if mode == "raw_mse":
        return ((pred - target) ** 2).masked_select(valid).mean()
    if mode == "normalized_mse":
        pred = F.layer_norm(pred, pred.shape[-1:])
        target = F.layer_norm(target, target.shape[-1:])
        return ((pred - target) ** 2).masked_select(valid).mean()
    if mode == "cosine":
        token_mask = mask.squeeze(-1)
        loss = 1.0 - F.cosine_similarity(pred, target, dim=-1)
        loss = loss.masked_select(token_mask)
        return loss.mean() if loss.numel() else pred.sum() * 0.0
    raise ValueError(f"Unknown representation loss mode: {mode}")


def attention_kl(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
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
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(0) - target.unsqueeze(1)
    pair_mask = target_diff.abs() > margin
    if pair_mask.sum() == 0:
        return pred.sum() * 0.0
    signs = target_diff.sign()
    logits = signs * pred_diff / max(temperature, 1e-6)
    return F.softplus(-logits).masked_select(pair_mask).mean()


def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred = pred - pred.mean()
    target = target.detach() - target.detach().mean()
    pred_std = pred.pow(2).mean().sqrt()
    target_std = target.pow(2).mean().sqrt()
    if pred_std.detach() < 1e-6 or target_std.detach() < 1e-6:
        return pred.sum() * 0.0
    corr = (pred * target).mean() / pred_std.clamp_min(1e-6) / target_std.clamp_min(1e-6)
    return 1.0 - corr.clamp(-1.0, 1.0)


def soft_rank(scores: torch.Tensor, *, temperature: float = 0.05) -> torch.Tensor:
    if scores.numel() < 2:
        return torch.ones_like(scores)
    pairwise = scores.unsqueeze(0) - scores.unsqueeze(1)
    return 1.0 + torch.sigmoid(pairwise / max(temperature, 1e-6)).sum(dim=1)


def soft_spearman_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_rank = soft_rank(pred, temperature=temperature)
    with torch.no_grad():
        target_rank = soft_rank(target.detach(), temperature=temperature)
    return pearson_correlation_loss(pred_rank, target_rank)


def listwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.15,
) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    teacher_probs = torch.softmax(target.detach() / max(temperature, 1e-6), dim=0)
    student_log_probs = torch.log_softmax(pred / max(temperature, 1e-6), dim=0)
    return F.kl_div(student_log_probs, teacher_probs, reduction="sum")


def relational_distance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.size(0) < 2:
        return student.sum() * 0.0
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    student_dist_sq = (
        student.pow(2).sum(dim=1, keepdim=True) + student.pow(2).sum(dim=1).unsqueeze(0)
    ) - 2.0 * (student @ student.T)
    teacher_dist_sq = (
        teacher.pow(2).sum(dim=1, keepdim=True) + teacher.pow(2).sum(dim=1).unsqueeze(0)
    ) - 2.0 * (teacher @ teacher.T)
    student_dist = torch.sqrt(student_dist_sq.clamp_min(1e-12))
    teacher_dist = torch.sqrt(teacher_dist_sq.clamp_min(1e-12))
    pair_mask = torch.triu(
        torch.ones(student.size(0), student.size(0), device=student.device, dtype=torch.bool),
        diagonal=1,
    )
    student_dist = student_dist.masked_select(pair_mask)
    teacher_dist = teacher_dist.masked_select(pair_mask)
    teacher_positive = teacher_dist[teacher_dist > 0]
    student_positive = student_dist[student_dist > 0]
    if teacher_positive.numel() == 0 or student_positive.numel() == 0:
        return student.sum() * 0.0
    teacher_mean = teacher_positive.mean()
    student_mean = student_positive.mean()
    teacher_dist = teacher_dist / teacher_mean.clamp_min(1e-6)
    student_dist = student_dist / student_mean.clamp_min(1e-6)
    return F.smooth_l1_loss(student_dist, teacher_dist)


def contrastive_hidden_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if student.size(0) < 2:
        return student.sum() * 0.0
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    logits = student @ teacher.T / max(temperature, 1e-6)
    labels = torch.arange(student.size(0), device=student.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    use_kd: bool,
    weights: dict[str, float],
    repr_loss: str = "cosine",
    rank_temperature: float = 0.15,
    soft_rank_temperature: float = 0.05,
    contrastive_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = F.mse_loss(outputs["predicted_ecr"], batch["ecr_true"])
    losses = {"hard_ecr": hard}
    total = weights.get("hard_ecr", 1.0) * hard

    hard_rank_weight = weights.get("hard_rank", 0.0)
    if hard_rank_weight:
        hard_rank = pairwise_rank_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
            temperature=rank_temperature,
        )
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
                F.layer_norm(
                    outputs["teacher_space_hidden"], outputs["teacher_space_hidden"].shape[-1:]
                ),
                F.layer_norm(fusion_target, fusion_target.shape[-1:]),
            )
        elif repr_loss == "cosine":
            hidden = (
                1.0 - F.cosine_similarity(outputs["teacher_space_hidden"], fusion_target, dim=-1)
            ).mean()
        else:
            raise ValueError(f"Unknown representation loss mode: {repr_loss}")
        attn = attention_kl(
            outputs["temporal_attention"],
            batch["teacher_attention"],
            batch["clip_mask"],
        )

        optional_losses: dict[str, torch.Tensor] = {}
        if weights.get("teacher_rank", 0.0):
            optional_losses["teacher_rank"] = pairwise_rank_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
                temperature=rank_temperature,
            )
        if weights.get("teacher_pearson", 0.0):
            optional_losses["teacher_pearson"] = pearson_correlation_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
            )
        if weights.get("teacher_spearman", 0.0):
            optional_losses["teacher_spearman"] = soft_spearman_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
                temperature=soft_rank_temperature,
            )
        if weights.get("teacher_listwise", 0.0):
            optional_losses["teacher_listwise"] = listwise_rank_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
                temperature=rank_temperature,
            )
        if weights.get("rkd_distance", 0.0):
            optional_losses["rkd_distance"] = relational_distance_loss(
                outputs["teacher_space_hidden"],
                fusion_target,
            )
        if weights.get("contrastive_hidden", 0.0):
            optional_losses["contrastive_hidden"] = contrastive_hidden_loss(
                outputs["teacher_space_hidden"],
                fusion_target,
                temperature=contrastive_temperature,
            )
        if weights.get("action_hallucination", 0.0) and "pred_action_feature" in outputs:
            optional_losses["action_hallucination"] = masked_representation_loss(
                outputs["pred_action_feature"],
                batch["teacher_action"],
                batch["clip_mask"],
                mode=repr_loss,
            )
        if weights.get("caption_hallucination", 0.0) and "pred_caption_feature" in outputs:
            optional_losses["caption_hallucination"] = masked_representation_loss(
                outputs["pred_caption_feature"],
                batch["teacher_caption_feature"],
                batch["clip_mask"],
                mode=repr_loss,
            )

        losses.update(
            {
                "soft_ecr": soft,
                "clip_ecr": clip,
                "temporal_hidden": temporal,
                "fusion_hidden": hidden,
                "attention": attn,
                **optional_losses,
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
        for name, loss in optional_losses.items():
            total = total + weights.get(name, 0.0) * loss

    losses["total"] = total
    return total, {key: float(value.detach().cpu()) for key, value in losses.items()}

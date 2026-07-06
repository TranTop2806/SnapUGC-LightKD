"""Compact student model and KD losses for official SnapUGC artifacts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class DropPathTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN
        x = self.norm1(src)
        if src_key_padding_mask is not None:
            attn_output, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        else:
            attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        src = src + self.drop_path(self.dropout1(attn_output))

        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.drop_path(self.dropout2(x))
        return src


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
    """Small source-aware deployable student."""

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
        quality_input_dim: int = 0,
        quality_fusion: str = "none",
        dover_input_dim: int = 0,
        dover_fusion: str = "none",
        ecr_bins: int = 0,
        temporal_aggregation: str = "attention",
        local_clips: int = 5,
        semantic_gated_fusion: bool = False,
        shared_distill_dim: int = 0,
        fusion_experts: int = 1,
        temporal_conv: str = "none",
        shared_transformer_weights: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.quality_fusion = quality_fusion
        self.dover_fusion = dover_fusion
        self.temporal_aggregation = temporal_aggregation
        self.local_clips = local_clips
        self.semantic_gated_fusion = semantic_gated_fusion
        self.shared_distill_dim = shared_distill_dim
        self.fusion_experts = fusion_experts
        if fusion_mode not in {"concat", "cross_attention"}:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        if temporal_aggregation not in {"attention", "global_local"}:
            raise ValueError(f"Unknown temporal_aggregation: {temporal_aggregation}")
        if projection_head not in {"linear", "mlp"}:
            raise ValueError(f"Unknown projection_head: {projection_head}")
        if dover_fusion not in {"none", "late_add", "late_concat"}:
            raise ValueError(f"Unknown dover_fusion: {dover_fusion}")
        if quality_fusion not in {"none", "clip_add"}:
            raise ValueError(f"Unknown quality_fusion: {quality_fusion}")
        if temporal_conv not in {"none", "depthwise", "full"}:
            raise ValueError(f"Unknown temporal_conv: {temporal_conv}")
        if dover_fusion != "none" and dover_input_dim <= 0:
            raise ValueError("dover_input_dim must be positive when late DOVER fusion is enabled")
        if quality_fusion != "none" and quality_input_dim <= 0:
            raise ValueError("quality_input_dim must be positive when quality fusion is enabled")
        if ecr_bins and ecr_bins < 2:
            raise ValueError("ecr_bins must be 0 or at least 2")
        if fusion_experts < 1:
            raise ValueError("fusion_experts must be at least 1")

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
        self.quality_proj = None
        if quality_fusion == "clip_add":
            self.quality_proj = nn.Sequential(
                nn.Linear(quality_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_clips, hidden_dim))
        self.shared_transformer_weights = shared_transformer_weights
        self.drop_path = drop_path
        self.n_layers = n_layers

        if shared_transformer_weights:
            self.temporal_layer = DropPathTransformerLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                drop_path=drop_path,
            )
            self.temporal_layers = None
            self.temporal_encoder = None
        elif drop_path > 0.0:
            self.temporal_layer = None
            self.temporal_layers = nn.ModuleList(
                [
                    DropPathTransformerLayer(
                        d_model=hidden_dim,
                        nhead=n_heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                        drop_path=drop_path,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.temporal_encoder = None
        elif n_layers > 0:
            self.temporal_layer = None
            self.temporal_layers = None
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
        else:
            self.temporal_layer = None
            self.temporal_layers = None
            self.temporal_encoder = None
        self.temporal_pool = AttentionPool(hidden_dim, dropout)
        self.local_pool = None
        self.global_pool = None
        video_context_dim = hidden_dim
        if temporal_aggregation == "global_local":
            self.local_pool = AttentionPool(hidden_dim, dropout)
            self.global_pool = AttentionPool(hidden_dim, dropout)
            video_context_dim = hidden_dim * 2

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

        self.text_gate = None
        if semantic_gated_fusion:
            self.text_gate = nn.Sequential(
                nn.Linear(hidden_dim, video_context_dim),
                nn.Sigmoid(),
            )
        fusion_input_dim = video_context_dim + hidden_dim
        if fusion_experts == 1:
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.fusion_expert_layers = None
            self.fusion_router = None
        else:
            self.fusion = None
            self.fusion_expert_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(fusion_input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                    )
                    for _ in range(fusion_experts)
                ]
            )
            self.fusion_router = nn.Linear(fusion_input_dim, fusion_experts)
        self.dover_proj = None
        self.late_fusion = None
        if dover_fusion != "none":
            self.dover_proj = nn.Sequential(
                nn.Linear(dover_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if dover_fusion == "late_concat":
                self.late_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.ecr_distribution_head = None
        if ecr_bins:
            self.ecr_distribution_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, ecr_bins),
            )
            self.register_buffer("ecr_bin_centers", torch.linspace(0.0, 1.0, ecr_bins))
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
        self.student_to_shared = None
        self.teacher_to_shared = None
        if shared_distill_dim:
            self.student_to_shared = nn.Linear(hidden_dim, shared_distill_dim)
            self.teacher_to_shared = nn.Linear(teacher_hidden_dim, shared_distill_dim)
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
        dover_inputs: torch.Tensor | None = None,
        quality_inputs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.temporal_conv is not None:
            conv_inputs = clip_inputs.transpose(1, 2)
            clip_inputs = self.temporal_conv(conv_inputs).transpose(1, 2)
        clip_hidden = self.clip_proj(clip_inputs)
        if self.quality_proj is not None:
            if quality_inputs is None or quality_inputs.numel() == 0:
                raise ValueError("quality_inputs are required when quality_fusion is enabled")
            clip_hidden = clip_hidden + self.quality_proj(quality_inputs)
        clip_hidden = clip_hidden + self.pos_embed[:, : clip_hidden.size(1), :]
        if self.shared_transformer_weights:
            for _ in range(self.n_layers):
                clip_hidden = self.temporal_layer(
                    clip_hidden,
                    src_key_padding_mask=~clip_mask.bool(),
                )
        elif self.temporal_layers is not None:
            for layer in self.temporal_layers:
                clip_hidden = layer(
                    clip_hidden,
                    src_key_padding_mask=~clip_mask.bool(),
                )
        elif self.temporal_encoder is not None:
            clip_hidden = self.temporal_encoder(
                clip_hidden,
                src_key_padding_mask=~clip_mask.bool(),
            )
        if self.temporal_aggregation == "global_local":
            n_local = min(self.local_clips, clip_hidden.size(1))
            local_hidden, _ = self.local_pool(
                clip_hidden[:, :n_local, :],
                clip_mask[:, :n_local],
            )
            global_hidden, temporal_attention = self.global_pool(clip_hidden, clip_mask)
            video_hidden = torch.cat([local_hidden, global_hidden], dim=-1)
        else:
            video_hidden, temporal_attention = self.temporal_pool(clip_hidden, clip_mask)

        if text_inputs.size(1) > 0:
            text_embeddings = self.text_proj(text_inputs)
            text_embeddings = text_embeddings + self._fit_type_embed(
                self.text_type_embed,
                text_embeddings.size(1),
            )
            text_hidden, text_attention = self.text_pool(text_embeddings, text_mask)
        else:
            text_hidden = torch.zeros_like(video_hidden)
            text_attention = torch.zeros(text_inputs.size(0), 0, device=text_inputs.device)

        if self.fusion_mode == "cross_attention" and text_inputs.size(1) > 0:
            attended_clip_hidden, _ = self.cross_attention(
                query=clip_hidden,
                key=text_embeddings,
                value=text_embeddings,
                key_padding_mask=~text_mask.bool(),
                need_weights=False,
            )
            cross_hidden = self.cross_norm(clip_hidden + attended_clip_hidden)
            if self.temporal_aggregation == "global_local":
                n_local = min(self.local_clips, cross_hidden.size(1))
                local_hidden, _ = self.local_pool(
                    cross_hidden[:, :n_local, :],
                    clip_mask[:, :n_local],
                )
                global_hidden, temporal_attention = self.global_pool(cross_hidden, clip_mask)
                video_hidden = torch.cat([local_hidden, global_hidden], dim=-1)
            else:
                video_hidden, temporal_attention = self.cross_pool(cross_hidden, clip_mask)
        else:
            cross_hidden = clip_hidden

        if self.text_gate is not None:
            video_hidden = video_hidden * self.text_gate(text_hidden)
        fusion_inputs = torch.cat([video_hidden, text_hidden], dim=-1)
        router_probs = None
        if self.fusion_expert_layers is None:
            fused_hidden = self.fusion(fusion_inputs)
        else:
            router_logits = self.fusion_router(fusion_inputs)
            router_probs = torch.softmax(router_logits, dim=-1)
            top1 = router_probs.argmax(dim=-1)
            hard_gate = F.one_hot(top1, num_classes=router_probs.size(-1)).type_as(router_probs)
            gate = hard_gate - router_probs.detach() + router_probs
            expert_outputs = torch.stack(
                [expert(fusion_inputs) for expert in self.fusion_expert_layers],
                dim=1,
            )
            fused_hidden = torch.sum(expert_outputs * gate.unsqueeze(-1), dim=1)
        teacher_aligned_hidden = fused_hidden
        ecr_hidden = fused_hidden
        if self.dover_fusion != "none":
            if dover_inputs is None or dover_inputs.numel() == 0:
                raise ValueError("Late DOVER fusion requires non-empty dover_inputs")
            dover_hidden = self.dover_proj(dover_inputs)
            if self.dover_fusion == "late_add":
                ecr_hidden = fused_hidden + dover_hidden
            else:
                ecr_hidden = self.late_fusion(torch.cat([fused_hidden, dover_hidden], dim=-1))
        dist_logits = None
        if self.ecr_distribution_head is not None:
            dist_logits = self.ecr_distribution_head(ecr_hidden)
            dist_probs = torch.softmax(dist_logits, dim=-1)
            predicted_ecr = (dist_probs * self.ecr_bin_centers.view(1, -1)).sum(dim=-1)
        else:
            predicted_ecr = self.ecr_head(ecr_hidden).squeeze(-1)
        clip_ecr = self.clip_ecr_head(cross_hidden).squeeze(-1)
        outputs = {
            "predicted_ecr": predicted_ecr,
            "clip_ecr": clip_ecr,
            "student_temporal": cross_hidden,
            "student_hidden": teacher_aligned_hidden,
            "teacher_space_temporal": self.hidden_to_teacher(cross_hidden),
            "teacher_space_hidden": self.hidden_to_teacher(teacher_aligned_hidden),
            "temporal_attention": temporal_attention,
            "text_attention": text_attention,
        }
        if router_probs is not None:
            outputs["router_probs"] = router_probs
        if self.student_to_shared is not None and self.teacher_to_shared is not None:
            outputs["student_to_shared"] = self.student_to_shared
            outputs["teacher_to_shared"] = self.teacher_to_shared
        if dist_logits is not None:
            outputs["ecr_dist_logits"] = dist_logits
            outputs["ecr_bin_centers"] = self.ecr_bin_centers
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
    embeddings: torch.Tensor | None = None,
    hard_similarity: float = 0.0,
    hard_target_margin: float = 0.2,
) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(0) - target.unsqueeze(1)
    pair_mask = target_diff.abs() > margin
    if embeddings is not None and hard_similarity > 0.0:
        emb = F.normalize(embeddings.detach(), dim=-1)
        sim = emb @ emb.T
        pair_mask = pair_mask & (sim > hard_similarity) & (target_diff.abs() > hard_target_margin)
    if pair_mask.sum() == 0:
        return pred.sum() * 0.0
    signs = target_diff.sign()
    logits = signs * pred_diff / max(temperature, 1e-6)
    return F.softplus(-logits).masked_select(pair_mask).mean()


def focal_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.0,
) -> torch.Tensor:
    if alpha <= 0.0:
        return F.mse_loss(pred, target)
    weights = torch.exp(alpha * target.detach().clamp(0.0, 1.0))
    weights = weights / weights.mean().clamp_min(1e-6)
    return (weights * (pred - target).pow(2)).mean()


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


def concordance_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus Lin's CCC for batch-level continuous regression.

    Unlike Pearson loss, CCC also penalizes mismatched means and variances. This
    makes it a useful metric-aligned companion to pointwise ECR regression.
    """
    if pred.numel() < 2:
        return pred.sum() * 0.0
    target = target.detach()
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_var = pred_centered.pow(2).mean()
    target_var = target_centered.pow(2).mean()
    covariance = (pred_centered * target_centered).mean()
    denominator = pred_var + target_var + (pred_mean - target_mean).pow(2)
    if denominator.detach() < 1e-8:
        return pred.sum() * 0.0
    ccc = 2.0 * covariance / denominator.clamp_min(1e-8)
    return 1.0 - ccc.clamp(-1.0, 1.0)


def std_match_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_std = pred.std(unbiased=False)
    target_std = target.detach().std(unbiased=False)
    return F.smooth_l1_loss(pred_std, target_std)


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


def gaussian_label_distribution(
    target: torch.Tensor,
    centers: torch.Tensor,
    *,
    sigma: float = 0.06,
) -> torch.Tensor:
    target = target.detach().clamp(0.0, 1.0).unsqueeze(-1)
    centers = centers.to(target.device).view(1, -1)
    dist = torch.exp(-0.5 * ((centers - target) / max(sigma, 1e-6)).pow(2))
    return dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def label_distribution_kl(
    logits: torch.Tensor,
    target: torch.Tensor,
    centers: torch.Tensor,
    *,
    sigma: float = 0.06,
) -> torch.Tensor:
    target_dist = gaussian_label_distribution(target, centers, sigma=sigma)
    return F.kl_div(F.log_softmax(logits, dim=-1), target_dist, reduction="batchmean")


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


def score_relation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = target.detach().unsqueeze(0) - target.detach().unsqueeze(1)
    pred_scale = pred_diff.detach().abs().mean().clamp_min(1e-6)
    target_scale = target_diff.abs().mean().clamp_min(1e-6)
    return F.smooth_l1_loss(pred_diff / pred_scale, target_diff / target_scale)


def prototype_kl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    centers: torch.Tensor | None = None,
    sigma: float = 0.10,
    temperature: float = 0.08,
) -> torch.Tensor:
    if centers is None:
        centers = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=pred.device)
    centers = centers.to(pred.device).view(1, -1)
    pred_logits = -torch.abs(pred.unsqueeze(-1) - centers) / max(temperature, 1e-6)
    target_dist = torch.exp(
        -0.5 * ((target.detach().clamp(0.0, 1.0).unsqueeze(-1) - centers) / max(sigma, 1e-6)).pow(2)
    )
    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(F.log_softmax(pred_logits, dim=-1), target_dist, reduction="batchmean")


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


def similarity_preserving_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    if student.size(0) < 2:
        return student.sum() * 0.0
    if student.ndim > 2:
        student = student.flatten(start_dim=1)
    if teacher.ndim > 2:
        teacher = teacher.flatten(start_dim=1)
    
    s_norm = F.normalize(student, p=2, dim=-1)
    t_norm = F.normalize(teacher.detach(), p=2, dim=-1)
    
    G_s = s_norm @ s_norm.T
    G_t = t_norm @ t_norm.T
    
    G_s_norm = F.normalize(G_s, p=2, dim=-1)
    G_t_norm = F.normalize(G_t, p=2, dim=-1)
    
    return F.mse_loss(G_s_norm, G_t_norm, reduction="mean")


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
    ldl_sigma: float = 0.06,
    focal_teacher_alpha: float = 0.0,
    teacher_score_loss: str = "mse",
    teacher_huber_beta: float = 0.05,
    hard_pair_similarity: float = 0.0,
    hard_pair_target_margin: float = 0.2,
    kd_transfer_beta: float = 0.0,
    prototype_sigma: float = 0.10,
    prototype_temperature: float = 0.08,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard = F.mse_loss(outputs["predicted_ecr"], batch["ecr_true"])
    losses = {"hard_ecr": hard}
    total = weights.get("hard_ecr", 1.0) * hard

    if weights.get("hard_ldl", 0.0) and "ecr_dist_logits" in outputs:
        hard_ldl = label_distribution_kl(
            outputs["ecr_dist_logits"],
            batch["ecr_true"],
            outputs["ecr_bin_centers"],
            sigma=ldl_sigma,
        )
        losses["hard_ldl"] = hard_ldl
        total = total + weights.get("hard_ldl", 0.0) * hard_ldl

    hard_rank_weight = weights.get("hard_rank", 0.0)
    if hard_rank_weight:
        hard_rank = pairwise_rank_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
            temperature=rank_temperature,
            embeddings=outputs.get("student_hidden"),
            hard_similarity=hard_pair_similarity,
            hard_target_margin=hard_pair_target_margin,
        )
        losses["hard_rank"] = hard_rank
        total = total + hard_rank_weight * hard_rank
    if weights.get("hard_pearson", 0.0):
        hard_pearson = pearson_correlation_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
        )
        losses["hard_pearson"] = hard_pearson
        total = total + weights.get("hard_pearson", 0.0) * hard_pearson
    if weights.get("hard_ccc", 0.0):
        hard_ccc = concordance_correlation_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
        )
        losses["hard_ccc"] = hard_ccc
        total = total + weights.get("hard_ccc", 0.0) * hard_ccc
    if weights.get("hard_spearman", 0.0):
        hard_spearman = soft_spearman_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
            temperature=soft_rank_temperature,
        )
        losses["hard_spearman"] = hard_spearman
        total = total + weights.get("hard_spearman", 0.0) * hard_spearman
    if weights.get("hard_listwise", 0.0):
        hard_listwise = listwise_rank_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
            temperature=rank_temperature,
        )
        losses["hard_listwise"] = hard_listwise
        total = total + weights.get("hard_listwise", 0.0) * hard_listwise
    if weights.get("hard_std", 0.0):
        hard_std = std_match_loss(
            outputs["predicted_ecr"],
            batch["ecr_true"],
        )
        losses["hard_std"] = hard_std
        total = total + weights.get("hard_std", 0.0) * hard_std

    if use_kd:
        if kd_transfer_beta > 0.0:
            transfer_weight = torch.exp(
                -kd_transfer_beta * (batch["teacher_ecr"] - batch["ecr_true"]).detach().abs()
            )
            transfer_weight = transfer_weight / transfer_weight.mean().clamp_min(1e-6)
            soft_base = (outputs["predicted_ecr"] - batch["teacher_ecr"]).pow(2)
            soft = (transfer_weight * soft_base).mean()
        elif teacher_score_loss == "huber":
            soft = F.smooth_l1_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
                beta=teacher_huber_beta,
            )
        else:
            soft = focal_mse_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
                alpha=focal_teacher_alpha,
            )
        clip = masked_mse(outputs["clip_ecr"], batch["teacher_clip_ecr"], batch["clip_mask"])
        use_shared_distill = "student_to_shared" in outputs and "teacher_to_shared" in outputs
        if use_shared_distill:
            temporal = masked_representation_loss(
                outputs["student_to_shared"](outputs["student_temporal"]),
                outputs["teacher_to_shared"](batch["teacher_temporal"].detach()),
                batch["clip_mask"],
                mode=repr_loss,
            )
        else:
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
        if use_shared_distill:
            student_hidden = outputs["student_to_shared"](outputs["student_hidden"])
            teacher_hidden = outputs["teacher_to_shared"](fusion_target.detach())
            if repr_loss == "raw_mse":
                hidden = F.mse_loss(student_hidden, teacher_hidden)
            elif repr_loss == "normalized_mse":
                hidden = F.mse_loss(
                    F.layer_norm(student_hidden, student_hidden.shape[-1:]),
                    F.layer_norm(teacher_hidden, teacher_hidden.shape[-1:]),
                )
            elif repr_loss == "cosine":
                hidden = (
                    1.0 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1)
                ).mean()
            else:
                raise ValueError(f"Unknown representation loss mode: {repr_loss}")
        elif repr_loss == "raw_mse":
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
                embeddings=outputs.get("student_hidden"),
                hard_similarity=hard_pair_similarity,
                hard_target_margin=hard_pair_target_margin,
            )
        if weights.get("teacher_pearson", 0.0):
            optional_losses["teacher_pearson"] = pearson_correlation_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
            )
        if weights.get("teacher_ccc", 0.0):
            optional_losses["teacher_ccc"] = concordance_correlation_loss(
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
        if weights.get("teacher_score_relation", 0.0):
            optional_losses["teacher_score_relation"] = score_relation_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
            )
        if weights.get("hard_score_relation", 0.0):
            optional_losses["hard_score_relation"] = score_relation_loss(
                outputs["predicted_ecr"],
                batch["ecr_true"],
            )
        if weights.get("teacher_prototype", 0.0):
            optional_losses["teacher_prototype"] = prototype_kl_loss(
                outputs["predicted_ecr"],
                batch["teacher_ecr"],
                sigma=prototype_sigma,
                temperature=prototype_temperature,
            )
        if weights.get("hard_prototype", 0.0):
            optional_losses["hard_prototype"] = prototype_kl_loss(
                outputs["predicted_ecr"],
                batch["ecr_true"],
                sigma=prototype_sigma,
                temperature=prototype_temperature,
            )
        if weights.get("rkd_distance", 0.0):
            optional_losses["rkd_distance"] = relational_distance_loss(
                outputs["teacher_space_hidden"],
                fusion_target,
            )
        if weights.get("student_teacher_relation", 0.0):
            optional_losses["student_teacher_relation"] = relational_distance_loss(
                outputs["student_hidden"],
                fusion_target,
            )
        if weights.get("contrastive_hidden", 0.0):
            optional_losses["contrastive_hidden"] = contrastive_hidden_loss(
                outputs["teacher_space_hidden"],
                fusion_target,
                temperature=contrastive_temperature,
            )
        if weights.get("spkd", 0.0):
            optional_losses["spkd"] = similarity_preserving_loss(
                outputs["student_hidden"],
                fusion_target,
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
        if weights.get("pseudo_ecr", 0.0):
            losses["pseudo_ecr"] = F.mse_loss(outputs["predicted_ecr"], batch["pseudo_ecr"])
        if "ecr_dist_logits" in outputs:
            if weights.get("teacher_ldl", 0.0):
                losses["teacher_ldl"] = label_distribution_kl(
                    outputs["ecr_dist_logits"],
                    batch["teacher_ecr"],
                    outputs["ecr_bin_centers"],
                    sigma=ldl_sigma,
                )
            if weights.get("pseudo_ldl", 0.0):
                losses["pseudo_ldl"] = label_distribution_kl(
                    outputs["ecr_dist_logits"],
                    batch["pseudo_ecr"],
                    outputs["ecr_bin_centers"],
                    sigma=ldl_sigma,
                )
        total = (
            total
            + weights.get("soft_ecr", 0.5) * soft
            + weights.get("pseudo_ecr", 0.0) * losses.get("pseudo_ecr", soft * 0.0)
            + weights.get("teacher_ldl", 0.0) * losses.get("teacher_ldl", soft * 0.0)
            + weights.get("pseudo_ldl", 0.0) * losses.get("pseudo_ldl", soft * 0.0)
            + weights.get("clip_ecr", 0.2) * clip
            + weights.get("temporal_hidden", 0.2) * temporal
            + weights.get("fusion_hidden", 0.1) * hidden
            + weights.get("attention", 0.05) * attn
        )
        for name, loss in optional_losses.items():
            total = total + weights.get(name, 0.0) * loss

    losses["total"] = total
    return total, {key: float(value.detach().cpu()) for key, value in losses.items()}

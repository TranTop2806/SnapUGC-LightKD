"""Teacher v9: rich-input teacher with correlation and ranking losses."""

import torch.nn.functional as F

from .teacher_v6_pooled_rank import pairwise_rank_loss
from .teacher_v7_corr_rank import pearson_loss
from .teacher_v8_rich_inputs import TeacherV8RichInputs


class TeacherV9RichCorrRank(TeacherV8RichInputs):
    """V8 architecture plus losses aligned with PLCC/SRCC validation metrics."""

    def __init__(
        self,
        *args,
        corr_weight=0.03,
        rank_weight=0.03,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.corr_weight = float(corr_weight)
        self.rank_weight = float(rank_weight)

    def forward(self, *args, **kwargs):
        ecr_targets = kwargs.get("ecr_targets")
        aesthetic_targets = kwargs.get("aesthetic_targets")
        technical_targets = kwargs.get("technical_targets")
        kwargs["ecr_targets"] = None
        kwargs["aesthetic_targets"] = None
        kwargs["technical_targets"] = None
        outputs = super().forward(*args, **kwargs)

        if ecr_targets is not None:
            predicted_ecr = outputs["predicted_ecr"]
            loss = F.mse_loss(predicted_ecr, ecr_targets)
            loss = loss + self.corr_weight * pearson_loss(predicted_ecr, ecr_targets)
            loss = loss + self.rank_weight * pairwise_rank_loss(predicted_ecr, ecr_targets)
            if aesthetic_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(
                    outputs["predicted_aesthetic"], aesthetic_targets
                )
            if technical_targets is not None:
                loss = loss + self.aux_weight * F.mse_loss(
                    outputs["predicted_technical"], technical_targets
                )
            outputs["loss"] = loss

        return outputs

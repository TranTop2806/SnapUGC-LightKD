"""Teacher v12: rich-input teacher with an unconstrained ECR head."""

import torch.nn as nn

from .teacher_v8_rich_inputs import TeacherV8RichInputs


class TeacherV12RichLinearHead(TeacherV8RichInputs):
    """V8 architecture with no sigmoid on ECR to avoid compressing extremes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.hidden_dim
        dropout = 0.35
        for module in self.ecr_head.modules():
            if isinstance(module, nn.Dropout):
                dropout = module.p
                break
        self.ecr_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

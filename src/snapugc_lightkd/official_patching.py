"""Small, idempotent source patches for the pinned official SnapUGC code."""

from __future__ import annotations

from pathlib import Path

EXPORT_IMPORT = """from snapugc_lightkd.teacher_export import (
    flush_teacher_artifacts as _snapugc_flush_teacher_artifacts,
    save_teacher_artifact as _snapugc_save_teacher_artifact,
)
"""


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    """Replace a pinned source fragment and fail loudly if upstream drifted."""
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Cannot apply {label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def patch_teacher_export(ecr_dir: Path) -> None:
    """Add env-gated tensor capture and artifact export hooks."""
    _patch_evqa(ecr_dir / "modules" / "EVQA.py")
    _patch_inference_script(ecr_dir / "test_SnapUGC_baseline.py")


def _patch_evqa(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "import os\n" not in text.splitlines(keepends=True)[:10]:
        text = replace_once(
            text, "import torch\n", "import os\nimport torch\n", label="EVQA os import"
        )

    text = replace_once(
        text,
        """        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)""",
        """        attn = attn.softmax(dim=-1)
        if os.environ.get("SNAPUGC_EXPORT_ARTIFACTS") == "1":
            self.last_attn = attn.detach().to(torch.float16).cpu()
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)""",
        label="EVQA attention capture",
    )
    text = replace_once(
        text,
        """        feat_4 = feat4
        feat_3_embed = self.feat3_preprocess(feat_3)
        feat_3 = self.fc30(feat_3)
        feat_4 = self.fc4(feat_4)""",
        """        caption_feature_raw = feat4
        feat_4 = feat4
        feat_3_embed = self.feat3_preprocess(feat_3)
        feat_3 = self.fc30(feat_3)
        feat_4 = self.fc4(feat_4)""",
        label="EVQA caption capture",
    )
    text = replace_once(
        text,
        """        feat123 = self.fc_merge123(torch.cat((feat_12, feat_3, feat_4, music_feature1, music_feature2_, music_feature3_, music_feature4), dim=1))
        feat123 = feat123.unsqueeze(0)
        out = self.block1(feat123).squeeze(0)
        temp_out = self.out(out)
        temp_out = torch.mean(temp_out, dim=0)
        return temp_out""",
        """        fusion_hidden = self.fc_merge123(torch.cat((feat_12, feat_3, feat_4, music_feature1, music_feature2_, music_feature3_, music_feature4), dim=1))
        feat123 = fusion_hidden.unsqueeze(0)
        out = self.block1(feat123).squeeze(0)
        clip_ecr = self.out(out).view(-1)
        temp_out = torch.mean(clip_ecr, dim=0).view(1)
        if os.environ.get("SNAPUGC_EXPORT_ARTIFACTS") == "1":
            layer_attention = []
            for block in self.block1:
                attn = getattr(block.attn, "last_attn", None)
                if attn is not None:
                    layer_attention.append(attn.squeeze(0).mean(dim=0).to(torch.float16))
            attention_importance = (
                torch.stack(layer_attention).mean(dim=1)
                if layer_attention else torch.empty(0, dtype=torch.float16)
            )
            text_pooled = torch.stack((
                text_embedding1[0].mean(dim=0),
                text_embedding2[0].mean(dim=0),
                text_embedding3[0].mean(dim=0),
            ))
            self.last_artifacts = {
                "fusion_hidden": fusion_hidden.detach().to(torch.float16).cpu(),
                "temporal_hidden": out.detach().to(torch.float16).cpu(),
                "clip_ecr": clip_ecr.detach().to(torch.float16).cpu(),
                "caption_feature": caption_feature_raw.detach().to(torch.float16).cpu(),
                "action_feature": feat_3_embed.detach().to(torch.float16).cpu(),
                "frame_fusion_feature": feat_12.detach().to(torch.float16).cpu(),
                "text_pooled": text_pooled.detach().to(torch.float16).cpu(),
                "attention_importance": attention_importance.cpu(),
            }
        return temp_out""",
        label="EVQA artifact outputs",
    )
    path.write_text(text, encoding="utf-8")


def _patch_inference_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if EXPORT_IMPORT not in text:
        text = replace_once(
            text,
            "from pathlib import Path\n",
            f"from pathlib import Path\n{EXPORT_IMPORT}",
            label="teacher export import",
        )
    text = replace_once(
        text,
        "        out0_val = out0_mean.clamp(0.0, 1.0).item()\n",
        "        out0_val = out0_mean.clamp(0.0, 1.0).item()\n"
        "        _snapugc_save_teacher_artifact(idx, video_id, out0_val, model, caption, music1_text, text1[0], text2[0], path)\n",
        label="teacher artifact save call",
    )
    text = replace_once(
        text,
        '    with open("submission_baseline.csv", "w", newline="") as csvfile:',
        "    _snapugc_flush_teacher_artifacts(force=True)\n"
        '    with open("submission_baseline.csv", "w", newline="") as csvfile:',
        label="teacher artifact final flush",
    )
    path.write_text(text, encoding="utf-8")

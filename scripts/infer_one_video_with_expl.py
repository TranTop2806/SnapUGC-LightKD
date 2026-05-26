#!/usr/bin/env python3
"""Infer one SnapUGC video artifact row and return an NLA-inspired explanation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.explanations import (  # noqa: E402
    explain_student_prediction,
    load_captions,
    load_metadata,
    move_batch,
)
from snapugc_lightkd.official_artifacts import (  # noqa: E402
    OfficialTeacherArtifactDataset,
    StudentInputConfig,
    artifact_keys_for_input_config,
    collate_student_batch,
    load_official_artifact_rows,
)
from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory with official_teacher_artifacts_*.npz",
    )
    parser.add_argument(
        "--labels-csv",
        required=True,
        help="CSV containing Id/ECR and preferably Title/Description; ECR is optional here",
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--report-json",
        required=True,
        help="official_student_kd_report.json from training",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to student_kd_best.pth next to report-json",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional CSV for Title/Description; defaults to labels-csv",
    )
    parser.add_argument(
        "--captions-dir",
        default=None,
        help="Optional directory with *_captions.jsonl; defaults to artifact-dir",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    input_preset = report["input_preset"]
    model_kwargs = report["model_kwargs"]
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (report_path.parent / "student_kd_best.pth")
    )

    input_cfg = StudentInputConfig.from_preset(input_preset)
    ragged_keys = artifact_keys_for_input_config(input_cfg)
    rows = load_official_artifact_rows(
        args.artifact_dir,
        args.labels_csv,
        require_complete_labels=False,
        ragged_keys=ragged_keys,
    )

    matched = [row for row in rows if str(row["Id"]) == str(args.video_id)]
    if not matched:
        raise ValueError(
            f"Khong tim thay video_id={args.video_id} trong artifact rows. "
            "Hay chay official teacher voi --export-artifacts cho video nay truoc."
        )

    dataset = OfficialTeacherArtifactDataset(
        matched,
        input_cfg,
        max_clips=int(model_kwargs.get("max_clips", 16)),
    )
    batch = collate_student_batch([dataset[0]])

    device = torch.device(args.device)
    model = OfficialArtifactStudent(**model_kwargs).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    with torch.no_grad():
        moved = move_batch(batch, device)
        outputs = model(
            moved["clip_inputs"],
            moved["clip_mask"],
            moved["text_inputs"],
            moved["text_mask"],
        )

    teacher_ecr = None
    if "teacher_ecr" in batch and isinstance(batch["teacher_ecr"], torch.Tensor):
        teacher_ecr = float(batch["teacher_ecr"][0].item())

    metadata_csv = args.metadata_csv or args.labels_csv
    captions_dir = args.captions_dir or args.artifact_dir
    metadata = load_metadata(metadata_csv).get(str(args.video_id), {})
    caption = load_captions(captions_dir).get(str(args.video_id))
    reference_ecr_values = [
        float(row["ecr_true"])
        for row in rows
        if "ecr_true" in row and math.isfinite(float(row["ecr_true"]))
    ]

    result = explain_student_prediction(
        model=model,
        batch=moved,
        outputs=outputs,
        input_config=input_cfg,
        video_id=str(args.video_id),
        metadata=metadata,
        caption=caption,
        reference_ecr_values=reference_ecr_values,
        teacher_ecr=teacher_ecr,
        topk=args.topk,
    )
    result["meta"] = {
        "input_preset": input_preset,
        "checkpoint": str(ckpt_path),
        "report_json": str(report_path),
        "repr_loss": report.get("repr_loss"),
        "nla_source": (
            "Fraser-Taliente, Kantamneni, Ong et al. (2026), "
            "Natural Language Autoencoders; adapted as activation verbalization "
            "plus ablation reconstruction for SnapUGC artifacts."
        ),
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

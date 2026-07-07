from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from snapugc_lightkd.official_artifacts import RAGGED_KEYS, load_official_artifact_rows
from snapugc_lightkd.official_patching import patch_teacher_export
from snapugc_lightkd.official_student import OfficialArtifactStudent
from snapugc_lightkd.teacher_export import ARTIFACT_KEYS, TeacherArtifactExporter

ROOT = Path(__file__).resolve().parents[1]


class TeacherExportSmokeTest(unittest.TestCase):
    def test_exported_shard_loads_through_training_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exporter = TeacherArtifactExporter(directory, shard_size=2)
            for idx in range(2):
                artifacts = {
                    key: np.full((idx + 1, 3), idx + 1, dtype=np.float32) for key in ARTIFACT_KEYS
                }
                output = exporter.add(
                    idx=idx,
                    video_id=f"video-{idx}",
                    teacher_ecr=0.2 + idx * 0.1,
                    artifacts=artifacts,
                )

            self.assertIsNotNone(output)
            labels_path = Path(directory) / "labels.csv"
            with labels_path.open("w", encoding="utf-8", newline="") as labels_file:
                writer = csv.DictWriter(labels_file, fieldnames=["Id", "ECR"])
                writer.writeheader()
                writer.writerows([{"Id": "video-0", "ECR": 0.2}, {"Id": "video-1", "ECR": 0.3}])
            rows = load_official_artifact_rows(Path(directory), labels_path)
            self.assertEqual([row["Id"] for row in rows], ["video-0", "video-1"])
            for key in RAGGED_KEYS:
                self.assertEqual(rows[1][key].shape, (2, 3))

    def test_official_patch_is_idempotent_and_imports_package_helper(self) -> None:
        source = ROOT / "third_party" / "SnapUGC_Engagement" / "ECR_inference"
        if not source.exists():
            self.skipTest("official SnapUGC source is optional and has not been cloned")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ECR_inference"
            (target / "modules").mkdir(parents=True)
            shutil.copy2(source / "modules" / "EVQA.py", target / "modules" / "EVQA.py")
            shutil.copy2(source / "test_SnapUGC_baseline.py", target / "test_SnapUGC_baseline.py")

            patch_teacher_export(target)
            first = (target / "test_SnapUGC_baseline.py").read_text(encoding="utf-8")
            patch_teacher_export(target)
            second = (target / "test_SnapUGC_baseline.py").read_text(encoding="utf-8")

            self.assertEqual(first, second)
            self.assertIn("from snapugc_lightkd.teacher_export import", second)
            self.assertEqual(second.count("_snapugc_save_teacher_artifact(idx"), 1)


class StudentSmokeTest(unittest.TestCase):
    def test_minimal_forward_contract(self) -> None:
        model = OfficialArtifactStudent(
            clip_input_dim=12,
            text_input_dim=8,
            hidden_dim=16,
            teacher_hidden_dim=32,
            max_clips=4,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
        ).eval()
        with torch.inference_mode():
            outputs = model(
                clip_inputs=torch.randn(2, 4, 12),
                clip_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
                text_inputs=torch.randn(2, 3, 8),
                text_mask=torch.ones(2, 3, dtype=torch.bool),
            )

        self.assertEqual(outputs["predicted_ecr"].shape, (2,))
        self.assertEqual(outputs["clip_ecr"].shape, (2, 4))
        self.assertTrue(torch.isfinite(outputs["predicted_ecr"]).all())


if __name__ == "__main__":
    unittest.main()

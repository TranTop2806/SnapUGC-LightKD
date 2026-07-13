from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import infer_new_video_with_student_expl as infer  # noqa: E402

import demo_app.app as demo  # noqa: E402
from snapugc_lightkd.explanations import explain_student_prediction  # noqa: E402
from snapugc_lightkd.video_editing import _apply_operations, _mux_original_audio  # noqa: E402


class TinyStudent(torch.nn.Module):
    def forward(
        self,
        clip_inputs: torch.Tensor,
        clip_mask: torch.Tensor,
        text_inputs: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, n_clips, _ = clip_inputs.shape
        n_text = text_inputs.shape[1]
        score = 0.5 + 0.01 * clip_inputs.sum(dim=(1, 2)) + 0.01 * text_inputs.sum(
            dim=(1, 2)
        )
        return {
            "predicted_ecr": score,
            "temporal_attention": torch.full(
                (batch_size, n_clips), 1.0 / n_clips, device=clip_inputs.device
            ),
            "clip_ecr": torch.full(
                (batch_size, n_clips), 0.5, device=clip_inputs.device
            ),
            "text_attention": torch.full(
                (batch_size, n_text), 1.0 / n_text, device=clip_inputs.device
            ),
        }


class InferenceSafetyTest(unittest.TestCase):
    def test_report_and_complete_checkpoint_are_required(self) -> None:
        with self.assertRaises(FileNotFoundError):
            infer.require_report_path(None)

        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "partial.pth"
            torch.save({"weight": model.weight.detach().clone()}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "incomplete or incompatible"):
                infer.load_required_checkpoint(model, checkpoint, torch.device("cpu"))

    def test_health_does_not_report_missing_checkpoint_as_loaded(self) -> None:
        with (
            patch.object(demo, "resolve_report_path", return_value=None),
            patch.object(demo, "resolve_checkpoint_path", return_value=None),
            patch.object(demo, "resolve_efficientnet_path", return_value=None),
        ):
            status = demo.health()

        self.assertFalse(status["model_ready"])
        self.assertIsNone(status["report_json"])
        self.assertIsNone(status["checkpoint"])

    def test_ui_inference_rejects_missing_model_artifacts(self) -> None:
        with patch.object(demo, "resolve_report_path", return_value=None):
            with self.assertRaises(demo.HTTPException) as raised:
                demo.run_student_inference(
                    video_path=Path("video.mp4"),
                    title="title",
                    description="",
                    device="cpu",
                    topk=1,
                    out_json=Path("result.json"),
                    assets_dir=Path("assets"),
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("report JSON is missing", str(raised.exception.detail))

    def test_empty_text_sources_are_not_ranked_as_evidence(self) -> None:
        model = TinyStudent()
        batch = {
            "clip_inputs": torch.ones((1, 2, 1)),
            "clip_mask": torch.ones((1, 2), dtype=torch.bool),
            "text_inputs": torch.tensor([[[0.0], [1.0], [0.0]]]),
            "text_mask": torch.ones((1, 3), dtype=torch.bool),
        }
        result = explain_student_prediction(
            model=model,
            batch=batch,
            outputs=model(
                batch["clip_inputs"],
                batch["clip_mask"],
                batch["text_inputs"],
                batch["text_mask"],
            ),
            input_config=SimpleNamespace(
                use_sound_text=True,
                use_title_text=True,
                use_description_text=True,
                use_caption_text=False,
            ),
            video_id="sample",
            metadata={"sound": "  ", "title": "Golf practice", "description": None},
            topk=1,
        )

        self.assertEqual(
            [row["stream"] for row in result["evidence"]["text_streams"]],
            ["title"],
        )
        self.assertEqual(
            result["nla_style_explanation"]["natural_language_bottleneck"][
                "selected_text_streams"
            ],
            ["title"],
        )

    def test_negative_brightness_clamps_instead_of_reflecting(self) -> None:
        frame = np.full((2, 2, 3), 5, dtype=np.uint8)
        edited = _apply_operations(frame, {"brightness": -10.0})
        self.assertTrue(np.array_equal(edited, np.zeros_like(frame)))

    def test_audio_mux_fails_when_ffmpeg_is_unavailable(self) -> None:
        with patch("snapugc_lightkd.video_editing.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "required to preserve"):
                _mux_original_audio(
                    video_without_audio=Path("silent.mp4"),
                    original_video=Path("original.mp4"),
                    output_video=Path("output.mp4"),
                )


if __name__ == "__main__":
    unittest.main()

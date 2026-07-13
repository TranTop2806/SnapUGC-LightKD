from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.llm_explainer import (  # noqa: E402
    _apply_grounding_guard,
    _local_generation_token_budgets,
    _normalize_llm_output,
    _prompt,
)


def sample_payload() -> dict:
    return {
        "prediction": {
            "student_ecr": 0.878,
            "band": "high",
            "band_vi": "cao",
        },
        "input_context": {"title": "funny memes", "description": None},
        "top_clips": [
            {
                "clip_index": 14,
                "time": "88-94% video",
                "semantic_label": "chuyển động rõ, màu sắc nổi bật",
                "semantic_profile": {
                    "attributes": [
                        {"name": "motion", "label": "strong"},
                        {"name": "lighting", "label": "balanced"},
                    ]
                },
                "contribution_to_score": 0.008,
            }
        ],
        "text_streams": [
            {
                "stream": "title",
                "source_text": "funny memes",
                "attention": 0.99,
            }
        ],
        "semantic_attributes": [],
        "recommendations": ["Metadata: thêm description ngắn và cụ thể."],
    }


class LocalLlmGuardTest(unittest.TestCase):
    def test_default_and_retry_token_budgets(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_local_generation_token_budgets(), [800, 1200])
        with patch.dict(
            os.environ,
            {
                "SNAPUGC_LOCAL_LLM_MAX_NEW_TOKENS": "420",
                "SNAPUGC_LOCAL_LLM_RETRY_TOKENS": "900",
            },
            clear=True,
        ):
            self.assertEqual(_local_generation_token_budgets(), [420, 900])

    def test_grounding_guard_rejects_unsupported_and_contradictory_claims(self) -> None:
        payload = sample_payload()
        generated = _normalize_llm_output(
            {
                "summary": "Video này sẽ thu hút người xem và giữ sự chú ý.",
                "claims": [
                    "Không có clip nào có chuyển động mạnh.",
                    "Clip 99 là bằng chứng chính.",
                    "Clip 14 có chuyển động mạnh.",
                ],
                "top_evidence_rationales": [
                    "Clip 14 có chuyển động mạnh và màu sắc nổi bật."
                ],
                "recommendations": ["Hãy làm video viral hơn."],
            },
            provider="local_transformers",
            model="test-model",
            used_llm=True,
            fallback_reason=None,
        )

        guarded = _apply_grounding_guard(generated, payload=payload, language="vi")

        self.assertEqual(guarded["summary"], "Mô hình dự đoán ECR=0.878, thuộc nhóm cao.")
        claims = " ".join(guarded["claims"])
        self.assertIn("Clip 14 có chuyển động mạnh.", claims)
        self.assertNotIn("Không có clip nào", claims)
        self.assertNotIn("Clip 99", claims)
        self.assertEqual(guarded["recommendations"], payload["recommendations"])
        self.assertTrue(guarded["llm"]["grounding_guard_applied"])
        self.assertGreaterEqual(guarded["llm"]["discarded_items"], 4)

    def test_prompt_requires_compact_grounded_json(self) -> None:
        prompt = _prompt(sample_payload(), language="vi")
        self.assertIn("at most two short sentences", prompt)
        self.assertIn("copy the supplied recommendations verbatim", prompt)
        self.assertIn("Grounded draft to preserve", prompt)


if __name__ == "__main__":
    unittest.main()

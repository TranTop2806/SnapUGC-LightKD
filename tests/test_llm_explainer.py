from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd import llm_explainer  # noqa: E402
from snapugc_lightkd.llm_explainer import (  # noqa: E402
    DEFAULT_LOCAL_LLM_MODEL,
    _apply_grounding_guard,
    _local_generation_token_budgets,
    _normalize_llm_output,
    _prompt,
    generate_semantic_explanation,
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
    def test_qwen35_is_the_default_local_model(self) -> None:
        self.assertEqual(DEFAULT_LOCAL_LLM_MODEL, "Qwen/Qwen3.5-4B")

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

    def test_local_failure_falls_back_to_openai_when_configured(self) -> None:
        local_failure = {
            "llm": {
                "provider": "template",
                "model": None,
                "used_llm": False,
                "fallback_reason": "local_llm_failed:RuntimeError",
            }
        }
        remote_success = {
            "summary": "remote ok",
            "claims": ["remote ok"],
            "top_evidence_rationales": [],
            "recommendations": [],
            "llm": {
                "provider": "openai_compatible",
                "model": "gpt-4o-mini",
                "used_llm": True,
                "fallback_reason": None,
            },
        }
        with (
            patch.dict(
                os.environ,
                {
                    "SNAPUGC_LLM_BACKEND": "local",
                    "SNAPUGC_LLM_API_KEY": "test-key",
                },
                clear=True,
            ),
            patch.object(
                llm_explainer,
                "generate_local_transformers_explanation",
                return_value=local_failure,
            ),
            patch.object(
                llm_explainer,
                "_generate_openai_compatible_explanation",
                return_value=remote_success,
            ) as remote_call,
        ):
            result = generate_semantic_explanation(sample_payload())

        remote_call.assert_called_once()
        self.assertTrue(result["llm"]["used_llm"])
        self.assertEqual(result["llm"]["provider"], "openai_compatible")
        self.assertEqual(result["llm"]["fallback_from"]["model"], "Qwen/Qwen3.5-4B")
        self.assertIn("local_llm_failed", result["llm"]["fallback_from"]["reason"])

    def test_openai_fallback_can_be_disabled(self) -> None:
        local_failure = {
            "llm": {
                "provider": "template",
                "model": None,
                "used_llm": False,
                "fallback_reason": "local_llm_failed:RuntimeError",
            }
        }
        with (
            patch.dict(
                os.environ,
                {
                    "SNAPUGC_LLM_BACKEND": "local",
                    "SNAPUGC_LLM_FALLBACK_TO_OPENAI": "0",
                    "SNAPUGC_LLM_API_KEY": "test-key",
                },
                clear=True,
            ),
            patch.object(
                llm_explainer,
                "generate_local_transformers_explanation",
                return_value=local_failure,
            ),
            patch.object(llm_explainer, "_generate_openai_compatible_explanation") as remote_call,
        ):
            result = generate_semantic_explanation(sample_payload())

        remote_call.assert_not_called()
        self.assertFalse(result["llm"]["used_llm"])
        self.assertIn("local_llm_failed", result["llm"]["fallback_reason"])


if __name__ == "__main__":
    unittest.main()

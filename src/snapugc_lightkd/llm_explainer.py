"""Semantic-labeling to natural-language explanation helpers.

The LLM layer is intentionally optional. If no API key is configured, the
module falls back to a deterministic template so demo inference stays fully
student-only and offline-capable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def build_semantic_llm_input(
    result: dict[str, Any],
    *,
    semantic_attributes: list[dict[str, Any]],
    recommendations: list[str],
    max_clips: int = 3,
) -> dict[str, Any]:
    """Create a compact, grounded package for LLM explanation."""

    faith = result.get("nla_style_explanation", {}).get("faithfulness", {})
    scores = result.get("scores", {})
    band = scores.get("band", {})
    top_clips = []
    for row in result.get("evidence", {}).get("top_clips", [])[:max_clips]:
        top_clips.append(
            {
                "clip_index": row.get("clip_index"),
                "time": row.get("relative_time", {}).get("label"),
                "semantic_label": row.get("semantic_label"),
                "semantic_profile": row.get("semantic_profile"),
                "student_attention": row.get("student_attention"),
                "contribution_to_score": row.get("contribution_to_score"),
                "direction": row.get("direction"),
            }
        )

    text_streams = []
    for row in result.get("evidence", {}).get("text_streams", [])[:2]:
        text_streams.append(
            {
                "stream": row.get("stream"),
                "source_text": row.get("source_text"),
                "attention": row.get("attention"),
                "contribution_to_score": row.get("contribution_to_score"),
                "direction": row.get("direction"),
            }
        )

    return {
        "task": "Explain a short-form social video engagement prediction.",
        "constraints": [
            "Use only the supplied evidence.",
            "Do not invent unseen objects, events, or audience reactions.",
            "Mention faithfulness/ablation only as model evidence, not absolute causality.",
            "Prefer clear language for a non-technical reader.",
        ],
        "prediction": {
            "student_ecr": scores.get("student_ecr"),
            "band": band.get("label"),
            "band_vi": band.get("label_vi"),
            "raw_checkpoint_ecr": scores.get("student_ecr_raw_checkpoint"),
            "native_heuristic_ecr": scores.get("native_heuristic_ecr"),
        },
        "input_context": result.get("input_context", {}),
        "top_clips": top_clips,
        "text_streams": text_streams,
        "semantic_attributes": semantic_attributes,
        "faithfulness": {
            "keep_only_selected_ecr": faith.get("keep_only_selected_ecr"),
            "remove_selected_ecr": faith.get("remove_selected_ecr"),
            "reconstruction_error_abs": faith.get("reconstruction_error_abs"),
            "necessity_delta": faith.get("necessity_delta_student_minus_removed"),
        },
        "recommendations": recommendations,
    }


def generate_semantic_explanation(
    payload: dict[str, Any],
    *,
    language: str = "vi",
    enabled: bool = True,
) -> dict[str, Any]:
    """Generate a natural-language explanation from structured evidence.

    Environment variables:
    - SNAPUGC_LLM_API_KEY or OPENAI_API_KEY
    - SNAPUGC_LLM_BASE_URL, default https://api.openai.com/v1
    - SNAPUGC_LLM_MODEL, default gpt-4o-mini
    """

    if not enabled:
        return template_semantic_explanation(payload, language=language, fallback_reason="llm_disabled")

    api_key = os.environ.get("SNAPUGC_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return template_semantic_explanation(payload, language=language, fallback_reason="missing_api_key")

    base_url = os.environ.get("SNAPUGC_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("SNAPUGC_LLM_MODEL", "gpt-4o-mini")
    timeout = float(os.environ.get("SNAPUGC_LLM_TIMEOUT", "45"))
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an explanation writer for a video engagement model. "
                    "You must stay grounded in the provided JSON evidence and return JSON only."
                ),
            },
            {
                "role": "user",
                "content": _prompt(payload, language=language),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        row = json.loads(raw)
        content = row["choices"][0]["message"]["content"]
        parsed = _parse_json_object(content)
        return _normalize_llm_output(
            parsed,
            provider="openai_compatible",
            model=model,
            used_llm=True,
            fallback_reason=None,
        )
    except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as exc:
        return template_semantic_explanation(
            payload,
            language=language,
            fallback_reason=f"llm_call_failed: {type(exc).__name__}",
        )


def template_semantic_explanation(
    payload: dict[str, Any],
    *,
    language: str = "vi",
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    prediction = payload.get("prediction", {})
    score = _safe_float(prediction.get("student_ecr"), default=0.0)
    band = prediction.get("band_vi") or prediction.get("band") or "unknown"
    clips = payload.get("top_clips", [])
    text_rows = payload.get("text_streams", [])
    faith = payload.get("faithfulness", {})

    if language.lower().startswith("en"):
        summary = f"The student predicts ECR={score:.3f}, in the {prediction.get('band', band)} range."
        claims = [summary]
        if clips:
            bits = []
            for clip in clips:
                delta = _safe_float(clip.get("contribution_to_score"), default=0.0)
                verb = "supports a higher score" if delta >= 0 else "pulls the score down"
                bits.append(f"{clip.get('time')}: {clip.get('semantic_label')} ({verb}, delta={delta:.3f})")
            claims.append("The main temporal evidence is: " + "; ".join(bits) + ".")
        if text_rows and text_rows[0].get("source_text"):
            claims.append(
                f"The strongest text evidence is {text_rows[0].get('stream')}: "
                f"{text_rows[0].get('source_text')}"
            )
        claims.append(_faithfulness_sentence(faith, english=True))
    else:
        summary = f"Student dự đoán ECR={score:.3f}, thuộc nhóm {band}."
        claims = [summary]
        if clips:
            bits = []
            for clip in clips:
                delta = _safe_float(clip.get("contribution_to_score"), default=0.0)
                verb = "hỗ trợ tăng score" if delta >= 0 else "kìm score"
                bits.append(f"{clip.get('time')}: {clip.get('semantic_label')} ({verb}, delta={delta:.3f})")
            claims.append("Evidence thời gian chính: " + "; ".join(bits) + ".")
        if text_rows and text_rows[0].get("source_text"):
            claims.append(
                f"Text quan trọng nhất là {text_rows[0].get('stream')}: "
                f"{text_rows[0].get('source_text')}"
            )
        claims.append(_faithfulness_sentence(faith, english=False))

    return _normalize_llm_output(
        {
            "summary": summary,
            "claims": claims,
            "top_evidence_rationales": claims[1:2],
            "recommendations": payload.get("recommendations", []),
        },
        provider="template",
        model=None,
        used_llm=False,
        fallback_reason=fallback_reason,
    )


def _prompt(payload: dict[str, Any], *, language: str) -> str:
    if language.lower().startswith("en"):
        language_instruction = "Write in English."
    else:
        language_instruction = "Write in Vietnamese."
    return (
        f"{language_instruction}\n"
        "Return a JSON object with keys: summary, claims, top_evidence_rationales, recommendations.\n"
        "- summary: one short paragraph for a normal user.\n"
        "- claims: 3-5 grounded bullet-like strings.\n"
        "- top_evidence_rationales: explain why the selected clips/text streams matter.\n"
        "- recommendations: reuse or lightly rewrite supplied recommendations.\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_llm_output(
    value: dict[str, Any],
    *,
    provider: str,
    model: str | None,
    used_llm: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    summary = str(value.get("summary") or "").strip()
    claims = _string_list(value.get("claims"))
    rationales = _string_list(value.get("top_evidence_rationales"))
    recommendations = _string_list(value.get("recommendations"))
    if summary and not claims:
        claims = [summary]
    return {
        "method": "semantic_labeling_to_llm",
        "summary": summary or (claims[0] if claims else ""),
        "claims": claims,
        "top_evidence_rationales": rationales,
        "recommendations": recommendations,
        "llm": {
            "provider": provider,
            "model": model,
            "used_llm": used_llm,
            "fallback_reason": fallback_reason,
        },
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _faithfulness_sentence(faith: dict[str, Any], *, english: bool) -> str:
    keep = _safe_float(faith.get("keep_only_selected_ecr"), default=float("nan"))
    remove = _safe_float(faith.get("remove_selected_ecr"), default=float("nan"))
    err = _safe_float(faith.get("reconstruction_error_abs"), default=float("nan"))
    if english:
        return f"Faithfulness check: keep-only={keep:.3f}, remove-selected={remove:.3f}, error={err:.3f}."
    return f"Kiểm chứng faithfulness: keep-only={keep:.3f}, remove-selected={remove:.3f}, error={err:.3f}."


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

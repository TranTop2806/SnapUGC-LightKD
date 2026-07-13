"""Semantic-labeling to natural-language explanation helpers.

The LLM layer is intentionally optional. If no API key is configured, the
module falls back to a deterministic template so demo inference stays fully
student-only and offline-capable.
"""

from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any


DEFAULT_LOCAL_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS = 800
DEFAULT_LOCAL_LLM_RETRY_TOKENS = 1200


def build_semantic_llm_input(
    result: dict[str, Any],
    *,
    semantic_attributes: list[dict[str, Any]],
    recommendations: list[str],
    max_clips: int = 3,
) -> dict[str, Any]:
    """Create a compact, grounded package for LLM explanation."""

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
            "Prefer clear language for a non-technical reader.",
            "Final output must preserve: ECR prediction, why top clips/text matter, and actionable suggestions.",
            "If multiple selected clips have similar labels, group them instead of repeating the same wording.",
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
    - SNAPUGC_LLM_BACKEND: auto, local, openai, template. Default auto.
    - SNAPUGC_LOCAL_LLM_MODEL, default Qwen/Qwen2.5-3B-Instruct.
    - SNAPUGC_LOCAL_LLM_CACHE, optional Hugging Face cache directory.
    - SNAPUGC_LLM_API_KEY or OPENAI_API_KEY
    - SNAPUGC_LLM_BASE_URL, default https://api.openai.com/v1
    - SNAPUGC_LLM_MODEL, default gpt-4o-mini
    """

    if not enabled:
        return template_semantic_explanation(payload, language=language, fallback_reason="llm_disabled")

    backend = os.environ.get("SNAPUGC_LLM_BACKEND", "auto").strip().lower()
    if backend == "template":
        return template_semantic_explanation(payload, language=language, fallback_reason="template_backend")
    if backend in {"local", "transformers", "hf"}:
        return generate_local_transformers_explanation(payload, language=language)
    if backend == "auto" and _local_model_available():
        local = generate_local_transformers_explanation(payload, language=language)
        if local["llm"]["used_llm"]:
            return local

    if backend not in {"auto", "openai", "api", "remote"}:
        return template_semantic_explanation(
            payload,
            language=language,
            fallback_reason=f"unknown_backend:{backend}",
        )

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
        return _apply_grounding_guard(
            _normalize_llm_output(
                parsed,
                provider="openai_compatible",
                model=model,
                used_llm=True,
                fallback_reason=None,
            ),
            payload=payload,
            language=language,
        )
    except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as exc:
        return template_semantic_explanation(
            payload,
            language=language,
            fallback_reason=f"llm_call_failed: {type(exc).__name__}",
        )


def generate_local_transformers_explanation(
    payload: dict[str, Any],
    *,
    language: str = "vi",
) -> dict[str, Any]:
    model_id = os.environ.get("SNAPUGC_LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL)
    try:
        tokenizer, model = _load_local_transformers_model(model_id)
        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise, grounded explanations for a student video engagement model. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": _prompt(payload, language=language)},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        parsed, attempts, token_budget = _generate_local_json(
            tokenizer=tokenizer,
            model=model,
            inputs=inputs,
            torch_module=torch,
        )
        result = _apply_grounding_guard(
            _normalize_llm_output(
                parsed,
                provider="local_transformers",
                model=model_id,
                used_llm=True,
                fallback_reason=None,
            ),
            payload=payload,
            language=language,
        )
        result["llm"]["generation_attempts"] = attempts
        result["llm"]["max_new_tokens_used"] = token_budget
        return result
    except Exception as exc:
        return template_semantic_explanation(
            payload,
            language=language,
            fallback_reason=f"local_llm_failed:{type(exc).__name__}",
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

    if language.lower().startswith("en"):
        summary = f"The student predicts ECR={score:.3f}, in the {prediction.get('band', band)} range."
        claims = [summary]
        if clips:
            bits = []
            for clip in clips:
                delta = _safe_float(clip.get("contribution_to_score"), default=0.0)
                verb = "supports a higher score" if delta >= 0 else "pulls the score down"
                bits.append(f"{clip.get('time')}: {clip.get('semantic_label')} ({verb}, delta={delta:.3f})")
            claims.append(_join_evidence_bits(bits, english=True))
        if text_rows and text_rows[0].get("source_text"):
            claims.append(
                f"The strongest text evidence is {text_rows[0].get('stream')}: "
                f"{text_rows[0].get('source_text')}"
            )
    else:
        summary = f"Mô hình dự đoán ECR={score:.3f}, thuộc nhóm {band}."
        claims = [summary]
        if clips:
            bits = []
            for clip in clips:
                delta = _safe_float(clip.get("contribution_to_score"), default=0.0)
                verb = "hỗ trợ tăng điểm" if delta >= 0 else "kìm điểm"
                bits.append(
                    f"{clip.get('time')}: {clip.get('semantic_label')} "
                    f"({verb}, mức ảnh hưởng={delta:.3f})"
                )
            claims.append(_join_evidence_bits(bits, english=False))
        if text_rows and text_rows[0].get("source_text"):
            stream = _stream_label(text_rows[0].get("stream"), language="vi")
            claims.append(
                f"Văn bản quan trọng nhất là {stream}; tín hiệu này được dùng để bổ sung "
                "ngữ cảnh cho dự đoán."
            )

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
        language_instruction = (
            "Write entirely in Vietnamese. Do not use English section labels. "
            "Translate technical labels into natural Vietnamese when possible."
        )
    grounded = template_semantic_explanation(payload, language=language)
    grounded_draft = {
        "summary": grounded["summary"],
        "claims": grounded["claims"],
        "top_evidence_rationales": grounded["top_evidence_rationales"],
        "recommendations": payload.get("recommendations", []),
    }
    return (
        f"{language_instruction}\n"
        "Return a JSON object with keys: summary, claims, top_evidence_rationales, recommendations.\n"
        "Use valid JSON with double quotes only. Do not wrap the JSON in Markdown.\n"
        "Keep the whole response concise enough to finish the JSON.\n"
        "- summary: at most two short sentences and must preserve the ECR score and band.\n"
        "- claims: 2-4 short strings based only on supplied evidence.\n"
        "- top_evidence_rationales: 1-3 short strings; group clips with similar labels.\n"
        "- recommendations: copy the supplied recommendations verbatim; do not add new advice.\n"
        "Do not claim actual audience reactions, virality, causality, or unseen events.\n"
        "Do not negate a strong/balanced attribute that appears in the evidence.\n\n"
        "Grounded draft to preserve and lightly rewrite:\n"
        f"{json.dumps(grounded_draft, ensure_ascii=False, indent=2)}\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _generate_local_json(
    *,
    tokenizer: Any,
    model: Any,
    inputs: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], int, int]:
    """Generate parseable JSON, retrying once when the first output is truncated."""

    last_error: Exception | None = None
    for attempt, token_budget in enumerate(_local_generation_token_budgets(), start=1):
        with torch_module.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        content = tokenizer.decode(generated, skip_special_tokens=True)
        try:
            return _parse_json_object(content), attempt, token_budget
        except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No local LLM generation budget configured")


def _local_generation_token_budgets() -> list[int]:
    initial = max(
        128,
        int(
            os.environ.get(
                "SNAPUGC_LOCAL_LLM_MAX_NEW_TOKENS",
                str(DEFAULT_LOCAL_LLM_MAX_NEW_TOKENS),
            )
        ),
    )
    retry = max(
        initial,
        int(
            os.environ.get(
                "SNAPUGC_LOCAL_LLM_RETRY_TOKENS",
                str(max(DEFAULT_LOCAL_LLM_RETRY_TOKENS, initial + 400)),
            )
        ),
    )
    return [initial] if retry == initial else [initial, retry]


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, dict):
                return value
        except (SyntaxError, ValueError) as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", text, 0)


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


def _apply_grounding_guard(
    result: dict[str, Any],
    *,
    payload: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    """Keep LLM prose while anchoring critical claims to deterministic evidence."""

    fallback = template_semantic_explanation(payload, language=language)
    fallback_summary = fallback["summary"]
    discarded = 0

    generated_summary = str(result.get("summary") or "").strip()
    if generated_summary and _grounding_rejection_reason(generated_summary, payload) is None:
        if generated_summary == fallback_summary or "ecr" in generated_summary.lower():
            summary = generated_summary
        else:
            summary = f"{fallback_summary} {generated_summary}"
    else:
        summary = fallback_summary
        discarded += int(bool(generated_summary))

    claims = [fallback_summary]
    for claim in _string_list(result.get("claims")):
        if _grounding_rejection_reason(claim, payload) is not None:
            discarded += 1
            continue
        if claim not in claims:
            claims.append(claim)
        if len(claims) >= 4:
            break
    for claim in fallback["claims"][1:]:
        if claim not in claims and len(claims) < 4:
            claims.append(claim)

    rationales = []
    for rationale in _string_list(result.get("top_evidence_rationales")):
        if _grounding_rejection_reason(rationale, payload) is not None:
            discarded += 1
            continue
        if rationale not in rationales:
            rationales.append(rationale)
        if len(rationales) >= 3:
            break
    if not rationales:
        rationales = list(fallback["top_evidence_rationales"])

    supplied_recommendations = _string_list(payload.get("recommendations"))
    generated_recommendations = _string_list(result.get("recommendations"))
    if generated_recommendations != supplied_recommendations:
        discarded += len(generated_recommendations)

    result["summary"] = summary
    result["claims"] = claims
    result["top_evidence_rationales"] = rationales
    result["recommendations"] = supplied_recommendations
    result["llm"]["grounding_guard_applied"] = True
    result["llm"]["discarded_items"] = discarded
    return result


def _grounding_rejection_reason(text: str, payload: dict[str, Any]) -> str | None:
    lowered = " ".join(text.lower().split())
    unsupported_phrases = (
        "khán giả",
        "thu hút người xem",
        "giữ sự chú ý",
        "sẽ tăng ecr",
        "viral",
        "audience reaction",
        "attract viewers",
        "hold attention",
        "will increase ecr",
        "guarantee",
    )
    if any(phrase in lowered for phrase in unsupported_phrases):
        return "unsupported_audience_or_causal_claim"

    allowed_clip_indices = {
        int(row["clip_index"])
        for row in payload.get("top_clips", [])
        if row.get("clip_index") is not None
    }
    for match in re.finditer(r"\bclip(?:\s+số)?\s*#?\s*(\d+)\b", lowered):
        if int(match.group(1)) not in allowed_clip_indices:
            return "unknown_clip_reference"

    if _payload_has_attribute(payload, name="motion", label="strong"):
        contradiction_patterns = (
            r"không có[^.]{0,60}chuyển động mạnh",
            r"không[^.]{0,40}clip[^.]{0,40}chuyển động mạnh",
            r"no[^.]{0,40}strong motion",
        )
        if any(re.search(pattern, lowered) for pattern in contradiction_patterns):
            return "contradicts_strong_motion_evidence"
    return None


def _payload_has_attribute(payload: dict[str, Any], *, name: str, label: str) -> bool:
    for clip in payload.get("top_clips", []):
        profile = clip.get("semantic_profile") or {}
        for attribute in profile.get("attributes", []):
            if attribute.get("name") == name and attribute.get("label") == label:
                return True
    return False


@lru_cache(maxsize=2)
def _load_local_transformers_model(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = os.environ.get("SNAPUGC_LOCAL_LLM_CACHE") or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    kwargs: dict[str, Any] = {"cache_dir": cache_dir}
    try:
        import torch

        if torch.cuda.is_available():
            kwargs.update({"torch_dtype": "auto", "device_map": "auto"})
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            kwargs.update({"torch_dtype": torch.float16})
        else:
            kwargs.update({"torch_dtype": torch.float32})
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    try:
        import torch

        if (
            not torch.cuda.is_available()
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            model = model.to("mps")
    except Exception:
        pass
    model.eval()
    return tokenizer, model


def _local_model_available() -> bool:
    model_id = os.environ.get("SNAPUGC_LOCAL_LLM_MODEL")
    cache_dir = os.environ.get("SNAPUGC_LOCAL_LLM_CACHE")
    if not model_id:
        return False
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        return True
    except Exception:
        return False


def _join_evidence_bits(bits: list[str], *, english: bool) -> str:
    if not bits:
        return ""
    if len(bits) <= 2:
        prefix = "The main temporal evidence is: " if english else "Bằng chứng thời gian chính: "
        return prefix + "; ".join(bits) + "."
    prefix = (
        "The selected temporal evidence appears in several high-scoring moments: "
        if english
        else "Bằng chứng thời gian chính xuất hiện ở nhiều đoạn quan trọng: "
    )
    return prefix + "; ".join(bits[:3]) + "."


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stream_label(value: Any, *, language: str) -> str:
    stream = str(value or "")
    if language.lower().startswith("vi"):
        return {
            "title": "tiêu đề",
            "description": "mô tả",
            "caption": "caption thị giác",
            "sound": "âm thanh",
        }.get(stream, stream or "văn bản")
    return stream or "text"

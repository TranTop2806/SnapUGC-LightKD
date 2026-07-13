#!/usr/bin/env python3
"""Download/cache a local instruction LLM for offline demo explanations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.llm_explainer import (  # noqa: E402
    DEFAULT_LOCAL_LLM_MODEL,
    _load_local_transformers_model,
    generate_local_transformers_explanation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("SNAPUGC_LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM_MODEL),
        help=(
            "Hugging Face model id. Qwen/Qwen3.5-4B is the default for the "
            "student-only explanation demo."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("SNAPUGC_LOCAL_LLM_CACHE"),
        help="Optional Hugging Face cache directory. Defaults to HF cache.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate a tiny JSON response after downloading.",
    )
    args = parser.parse_args()

    cache_dir = str(Path(args.cache_dir).expanduser()) if args.cache_dir else None
    os.environ["SNAPUGC_LOCAL_LLM_MODEL"] = args.model
    if cache_dir:
        os.environ["SNAPUGC_LOCAL_LLM_CACHE"] = cache_dir

    print(f"Downloading processor and model: {args.model}")
    try:
        _load_local_transformers_model(args.model)
    except ImportError as exc:
        raise SystemExit(
            "Local Qwen dependencies are missing. Install requirements-local-llm.txt "
            "or configure the OpenAI API fallback."
        ) from exc
    print("LOCAL_LLM_READY=1")
    print("SNAPUGC_LLM_BACKEND=auto")
    print(f"SNAPUGC_LOCAL_LLM_MODEL={args.model}")
    if cache_dir:
        print(f"SNAPUGC_LOCAL_LLM_CACHE={cache_dir}")

    if args.smoke_test:
        payload = {
            "prediction": {"student_ecr": 0.5, "band": "medium", "band_vi": "trung bình"},
            "input_context": {"title": "Smoke test", "description": "Local Qwen check"},
            "top_clips": [],
            "text_streams": [],
            "semantic_attributes": [],
            "recommendations": ["Keep the metadata concise."],
        }
        result = generate_local_transformers_explanation(payload, language="en")
        if not result["llm"]["used_llm"]:
            raise RuntimeError(result["llm"].get("fallback_reason") or "local smoke test failed")
        print("LOCAL_LLM_SMOKE_TEST=1")
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

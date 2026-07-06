#!/usr/bin/env python3
"""Download/cache a local instruction LLM for offline demo explanations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("SNAPUGC_LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        help=(
            "Hugging Face model id. Qwen/Qwen2.5-3B-Instruct is the default for "
            "Apple Silicon / 64GB RAM demos; use 0.5B for very small machines."
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = str(Path(args.cache_dir).expanduser()) if args.cache_dir else None
    print(f"Downloading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    print(f"Downloading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=cache_dir)
    model.eval()
    print("LOCAL_LLM_READY=1")
    print("SNAPUGC_LLM_BACKEND=local")
    print(f"SNAPUGC_LOCAL_LLM_MODEL={args.model}")
    if cache_dir:
        print(f"SNAPUGC_LOCAL_LLM_CACHE={cache_dir}")

    if args.smoke_test:
        prompt = (
            "Return JSON only: {\"summary\":\"ok\",\"claims\":[\"local model works\"],"
            "\"top_evidence_rationales\":[],\"recommendations\":[]}"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI wrapper for robust Qwen caption extraction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.qwen_extraction import main

if __name__ == "__main__":
    main()

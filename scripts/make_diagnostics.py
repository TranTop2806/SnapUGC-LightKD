#!/usr/bin/env python3
"""Write diagnostic plots and tables for a completed run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapugc_lightkd.diagnostics import write_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature JSON path")
    parser.add_argument("--report", required=True, help="Final experiment report JSON path")
    parser.add_argument("--subset-csv", required=True, help="Subset CSV path")
    parser.add_argument("--out-dir", required=True, help="Directory for plots and tables")
    parser.add_argument("--bins", type=int, default=10, help="Number of ECR quantile bins")
    args = parser.parse_args()

    paths = write_diagnostics(
        features_path=args.features,
        report_path=args.report,
        subset_csv=args.subset_csv,
        out_dir=args.out_dir,
        bins=args.bins,
    )
    print("Diagnostic files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

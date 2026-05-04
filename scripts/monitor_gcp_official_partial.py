#!/usr/bin/env python3
"""Checkpoint a running official SnapUGC inference from its live log.

The authors' inference script prints one prediction per processed video but only
writes the final submission CSV after all videos finish. For cost-controlled GCP
runs, this monitor parses the live log and saves partial CSV/reports every N
predictions. It can optionally stop the official processes after a target count.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


PRED_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s+(?P<id>[0-9a-f]{32})\s+(?P<pred>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def parse_predictions(log_path: Path):
    rows = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = PRED_RE.match(line)
            if not match:
                continue
            rows.append(
                {
                    "idx": int(match.group("idx")),
                    "Id": match.group("id"),
                    "ECR_pred": float(match.group("pred")),
                }
            )
    return rows


def read_seed_predictions(path: Path | None):
    if path is None:
        return []
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pred_text = row.get("ECR_pred", row.get("ECR", ""))
            if pred_text in (None, ""):
                continue
            rows.append(
                {
                    "idx": int(row.get("idx", i) or i),
                    "Id": str(row["Id"]),
                    "ECR_pred": float(pred_text),
                }
            )
    return rows


def merge_unique_predictions(seed_rows, live_rows):
    merged = []
    seen = set()
    for row in list(seed_rows) + list(live_rows):
        vid = row["Id"]
        if vid in seen:
            continue
        seen.add(vid)
        merged.append(
            {
                "idx": len(merged),
                "Id": vid,
                "ECR_pred": float(row["ECR_pred"]),
            }
        )
    return merged


def read_labels(csv_path: Path):
    labels = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ECR") not in (None, ""):
                labels[str(row["Id"])] = float(row["ECR"])
    return labels


def evaluate(rows, labels):
    ids = [row["Id"] for row in rows if row["Id"] in labels]
    if len(ids) < 3:
        return {"n_eval": len(ids)}
    pred_by_id = {row["Id"]: row["ECR_pred"] for row in rows}
    pred = np.array([pred_by_id[vid] for vid in ids], dtype=np.float64)
    true = np.array([labels[vid] for vid in ids], dtype=np.float64)
    plcc = pearsonr(pred, true)[0] if pred.std() > 0 and true.std() > 0 else 0.0
    srcc = spearmanr(pred, true).correlation
    ktau = kendalltau(pred, true).correlation
    metrics = {
        "n_eval": len(ids),
        "plcc": 0.0 if np.isnan(plcc) else float(plcc),
        "srcc": 0.0 if np.isnan(srcc) else float(srcc),
        "ktau": 0.0 if np.isnan(ktau) else float(ktau),
        "mse": float(np.mean((pred - true) ** 2)),
        "mae": float(np.mean(np.abs(pred - true))),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "true_mean": float(true.mean()),
        "true_std": float(true.std()),
    }
    metrics["final_score_srcc06_plcc04"] = 0.6 * metrics["srcc"] + 0.4 * metrics["plcc"]
    metrics["final_score_mean_plcc_srcc"] = 0.5 * (metrics["plcc"] + metrics["srcc"])
    return metrics


def save_outputs(rows, labels, out_dir: Path, target_n: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = rows[:target_n]
    pred_csv = out_dir / f"official_partial_{len(selected)}_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "Id", "ECR_pred", "ECR_true"])
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "ECR_true": labels.get(row["Id"], "")})

    metrics = evaluate(selected, labels)
    report = {
        "n_predictions": len(selected),
        "target_n": target_n,
        "prediction_csv": str(pred_csv),
        "metrics": metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_path = out_dir / f"official_partial_{len(selected)}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return pred_csv, report_path


def kill_matching_processes(patterns):
    own_pid = os.getpid()
    ps = subprocess.run(
        ["ps", "-eo", "pid,args"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    killed = []
    for line in ps[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, cmd = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        if any(pattern in cmd for pattern in patterns):
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                pass
    time.sleep(10)
    for pid in killed:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return killed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-n", type=int, default=5000)
    parser.add_argument("--every-n", type=int, default=500)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--seed-predictions",
        default=None,
        help="Existing partial prediction CSV to prepend for resume runs.",
    )
    parser.add_argument("--no-stop", action="store_true", help="Save partial outputs without killing the run.")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    labels = read_labels(Path(args.labels_csv))
    out_dir = Path(args.out_dir)
    seed_rows = read_seed_predictions(Path(args.seed_predictions)) if args.seed_predictions else []

    print(
        f"Monitoring {log_path} for {args.target_n} predictions. "
        f"every_n={args.every_n} seed_n={len(seed_rows)} shutdown={args.shutdown}",
        flush=True,
    )
    saved_targets = set()
    for checkpoint_n in range(args.every_n, min(len(seed_rows), args.target_n) + 1, args.every_n):
        save_outputs(seed_rows, labels, out_dir, checkpoint_n)
        saved_targets.add(checkpoint_n)
    while True:
        live_rows = parse_predictions(log_path)
        rows = merge_unique_predictions(seed_rows, live_rows)
        last = rows[-1] if rows else None
        print(
            f"progress n={len(rows)}/{args.target_n}"
            + (f" last_idx={last['idx']} last_id={last['Id']}" if last else ""),
            flush=True,
        )
        max_checkpoint = min(len(rows), args.target_n)
        for checkpoint_n in range(args.every_n, max_checkpoint + 1, args.every_n):
            if checkpoint_n not in saved_targets:
                save_outputs(rows, labels, out_dir, checkpoint_n)
                saved_targets.add(checkpoint_n)
        if len(rows) >= args.target_n:
            if args.target_n not in saved_targets:
                save_outputs(rows, labels, out_dir, args.target_n)
            if args.no_stop:
                print("no_stop=True; leaving official inference running.", flush=True)
                return
            killed = kill_matching_processes(
                [
                    "test_SnapUGC_baseline.py",
                    "run_official_snapugc_evqa.py",
                    "run_gcp_official_balanced_5k_from_links.sh",
                ]
            )
            print(f"killed_processes={killed}", flush=True)
            if args.shutdown:
                subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

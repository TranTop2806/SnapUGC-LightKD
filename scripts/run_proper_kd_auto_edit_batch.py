#!/usr/bin/env python3
"""Run Proper KD analyze + auto-edit on a normal-shaped ECR sample."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import NormalDist
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("SNAPUGC_LLM_BACKEND", "template")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import infer_new_video_with_student_expl as infer  # noqa: E402

from snapugc_lightkd.official_student import OfficialArtifactStudent  # noqa: E402
from snapugc_lightkd.video_editing import apply_feasible_video_edits  # noqa: E402

OUTPUT_COLUMNS = [
    "Id",
    "Title",
    "Description",
    "True ECR",
    "Predicted ECR",
    "Suggested Title",
    "Suggested Description",
    "After-edit ECR",
]


class ProperKDAnalyzer:
    def __init__(
        self,
        *,
        report_json: Path | None,
        checkpoint: Path | None,
        labels_csv: Path | None,
        device: str,
        max_clips: int | None,
        topk: int,
        text_encoder_model: str,
    ) -> None:
        self.device = infer.resolve_device(device)
        self.report_path = infer.require_report_path(
            infer.resolve_report_path(str(report_json) if report_json else None)
        )
        self.report = infer.load_report(self.report_path)
        self.model_kwargs = dict(
            self.report.get(
                "model_kwargs",
                {
                    "clip_input_dim": 1664,
                    "text_input_dim": 768,
                    "hidden_dim": 192,
                    "teacher_hidden_dim": 512,
                    "max_clips": 16,
                    "n_layers": 2,
                    "n_heads": 4,
                    "dropout": 0.22,
                    "projection_head": "mlp",
                },
            )
        )
        self.max_clips = int(max_clips or self.model_kwargs.get("max_clips", 16))
        self.model_kwargs["max_clips"] = self.max_clips
        self.input_preset = infer.resolve_native_input_preset(
            self.report.get("input_preset") or "clip_mobilenet_text",
            int(self.model_kwargs.get("clip_input_dim", 1664)),
        )
        self.text_encoder_model = text_encoder_model
        self.reference_values = infer.load_reference_ecr_values(str(labels_csv) if labels_csv else None)
        self.topk = topk

        self.model = OfficialArtifactStudent(**self.model_kwargs).to(self.device)
        self.checkpoint_path = infer.require_checkpoint_path(
            infer.resolve_checkpoint_path(
                str(checkpoint) if checkpoint else None,
                self.report_path,
            )
        )
        infer.load_required_checkpoint(self.model, self.checkpoint_path, self.device)
        self.checkpoint_loaded = True
        self.model.eval()

    def analyze(self, video_path: Path, *, title: str, description: str, video_id: str) -> dict[str, Any]:
        native = infer.build_native_student_inputs(
            video_path,
            title=title,
            description=description,
            max_clips=self.max_clips,
            clip_dim=int(self.model_kwargs.get("clip_input_dim", 1664)),
            device=self.device,
            no_visual_encoder=False,
            input_preset=self.input_preset,
            text_encoder_model=self.text_encoder_model,
        )
        batch = infer.move_batch(native.as_batch(), self.device)

        with torch.no_grad():
            outputs = self.model(
                batch["clip_inputs"],
                batch["clip_mask"],
                batch["text_inputs"],
                batch["text_mask"],
            )

        raw_student_score = float(outputs["predicted_ecr"][0].detach().cpu().item())
        student_ecr = raw_student_score
        outputs["predicted_ecr"] = torch.tensor(
            [student_ecr],
            device=self.device,
            dtype=torch.float32,
        )

        result = infer.explain_student_prediction(
            model=self.model,
            batch=batch,
            outputs=outputs,
            input_config=infer.make_input_config(native.text_streams),
            video_id=video_id,
            metadata=native.metadata,
            caption=None,
            reference_ecr_values=self.reference_values,
            teacher_ecr=None,
            topk=self.topk,
        )

        clip_metrics = [clip.metrics for clip in native.clips]
        clip_semantics = {clip.index: infer.semantic_clip_label(clip.metrics) for clip in native.clips}
        clip_profiles = {clip.index: infer.semantic_clip_profile(clip.metrics) for clip in native.clips}
        for section in ("all_clips", "top_clips"):
            for row in result["evidence"][section]:
                idx = int(row["clip_index"])
                row["semantic_label"] = clip_semantics.get(idx)
                row["semantic_profile"] = clip_profiles.get(idx)
                if 0 <= idx < len(clip_metrics):
                    row["native_visual_metrics"] = clip_metrics[idx]

        result["scores"]["student_ecr_raw_checkpoint"] = raw_student_score
        result["scores"]["native_heuristic_ecr"] = native.heuristic_score
        result["scores"]["student_ecr"] = student_ecr
        result["scores"]["band"] = infer.engagement_band(student_ecr, self.reference_values)

        recommendations = infer.build_recommendations(
            score=student_ecr,
            title=native.metadata.get("title"),
            description=native.metadata.get("description"),
            clip_rows=result["evidence"]["all_clips"],
            clip_metrics=clip_metrics,
        )
        semantic_attributes = infer.build_semantic_attributes(
            title=native.metadata.get("title"),
            description=native.metadata.get("description"),
            clip_rows=result["evidence"]["all_clips"],
            clip_metrics=clip_metrics,
        )
        llm_payload = infer.build_semantic_llm_input(
            result,
            semantic_attributes=semantic_attributes,
            recommendations=recommendations,
        )
        semantic_explanation = infer.generate_semantic_explanation(
            llm_payload,
            language="vi",
            enabled=False,
        )
        if not semantic_explanation["summary"]:
            semantic_explanation["summary"] = infer.build_student_summary(result)
        if not semantic_explanation["claims"]:
            semantic_explanation["claims"] = infer.build_student_claims(result)
        if not semantic_explanation["recommendations"]:
            semantic_explanation["recommendations"] = recommendations

        recommendation_groups = infer.build_recommendation_groups(
            title=native.metadata.get("title"),
            description=native.metadata.get("description"),
            clip_rows=result["evidence"]["all_clips"],
            clip_metrics=clip_metrics,
            semantic_attributes=semantic_attributes,
            fallback_recommendations=semantic_explanation["recommendations"],
        )
        metadata_suggestion = infer.build_metadata_suggestion(
            title=native.metadata.get("title"),
            description=native.metadata.get("description"),
            top_clips=result["evidence"]["top_clips"],
            semantic_attributes=semantic_attributes,
        )

        result["recommendations"] = [
            *recommendation_groups["post_production"],
            *recommendation_groups["content_reshoot"],
        ]
        result["recommendations_grouped"] = recommendation_groups
        result["metadata_suggestion"] = metadata_suggestion
        result["semantic_attributes"] = {
            "type": "posthoc_semantic_attributes",
            "attributes": semantic_attributes,
            "note": (
                "These attributes are deterministic semantic labels over video/text evidence. "
                "They are not a separately trained concept bottleneck model."
            ),
        }
        result["semantic_explanation"] = {
            **semantic_explanation,
            "input_package": llm_payload,
        }
        result["nla_style_explanation"]["summary"] = semantic_explanation["summary"]
        result["nla_style_explanation"]["claims"] = semantic_explanation["claims"]
        result["nla_style_explanation"]["natural_language_bottleneck"]["verbalizer"] = (
            "semantic-labeling evidence package followed by deterministic template explanation"
        )
        result["meta"] = {
            "inference_mode": "student_only_native_video",
            "teacher_called_at_inference": False,
            "device": str(self.device),
            "explanation_pipeline": (
                "student_attribution_ablation -> semantic_labeling -> template_explanation"
            ),
            "llm_used": False,
            "llm_provider": "template",
            "video_path": str(video_path.resolve()),
            "report_json": str(self.report_path) if self.report_path else None,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "checkpoint_loaded": self.checkpoint_loaded,
            "native_input_preset": self.input_preset,
            "student_score_policy": "raw_checkpoint_score",
            "text_encoder_model": self.text_encoder_model,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "data/train_data.csv"),
    )
    parser.add_argument("--official-dir", default=str(ROOT / "data/official_5k_split"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/proper_kd_auto_edit_100_normal"))
    parser.add_argument("--target-successes", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--normal-mean", type=float, default=0.50)
    parser.add_argument("--normal-std", type=float, default=0.18)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--report-json",
        default=str(
            ROOT
            / "results/final_4000_500_500_2026/proper_kd_seed42/official_student_kd_report.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "results/final_4000_500_500_2026/proper_kd_seed42/student_kd_best.pth"
        ),
    )
    parser.add_argument("--labels-csv", default=str(ROOT / "data/official_5k_split/split_all_5000.csv"))
    parser.add_argument("--text-encoder-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--download-timeout", type=int, default=240)
    parser.add_argument("--delete-videos", action="store_true")
    parser.add_argument("--force-reselect", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry IDs already present in the failure log instead of skipping them.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    videos_dir = out_dir / "videos"
    edited_dir = out_dir / "edited_videos"
    json_dir = out_dir / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    output_csv = out_dir / "proper_kd_auto_edit_100_normal_results.csv"
    output_xlsx = out_dir / "proper_kd_auto_edit_100_normal_results.xlsx"
    failed_jsonl = out_dir / "proper_kd_auto_edit_100_normal_failures.jsonl"
    selection_csv = out_dir / "proper_kd_auto_edit_100_normal_candidates.csv"

    if args.force_reselect or not selection_csv.exists():
        selected = select_candidates(
            train_csv=Path(args.train_csv),
            official_dir=Path(args.official_dir),
            n_candidates=args.candidate_count,
            seed=args.seed,
            normal_mean=args.normal_mean,
            normal_std=args.normal_std,
        )
        selected.to_csv(selection_csv, index=False)
        write_selection_summary(selected, out_dir, args)
    else:
        selected = pd.read_csv(selection_csv)

    rows = load_existing_rows(output_csv)
    done_ids = {str(row["Id"]) for row in rows}
    failed_ids = set() if args.retry_failures else load_failed_ids(failed_jsonl)
    print(
        f"batch_start candidates={len(selected)} existing_success={len(rows)} "
        f"target_successes={args.target_successes} previous_failures_skipped={len(failed_ids)} "
        f"out_dir={out_dir}",
        flush=True,
    )

    analyzer = ProperKDAnalyzer(
        report_json=Path(args.report_json) if args.report_json else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        labels_csv=Path(args.labels_csv) if args.labels_csv else None,
        device=args.device,
        max_clips=args.max_clips,
        topk=args.topk,
        text_encoder_model=args.text_encoder_model,
    )
    print(
        f"model_ready device={analyzer.device} checkpoint_loaded={analyzer.checkpoint_loaded} "
        f"preset={analyzer.input_preset}",
        flush=True,
    )

    start_time = time.time()
    for i, row in selected.iterrows():
        if len(rows) >= args.target_successes:
            break
        video_id = str(row["Id"])
        if video_id in done_ids:
            continue
        if video_id in failed_ids:
            continue
        title = clean_text_value(row.get("Title"))
        description = clean_text_value(row.get("Description"))
        true_ecr = safe_float(row.get("ECR"))
        video_path = videos_dir / f"{video_id}.mp4"
        edited_path = edited_dir / f"{video_id}_auto_edited.mp4"

        try:
            print(
                f"[{len(rows)+1}/{args.target_successes}] candidate={i+1}/{len(selected)} "
                f"id={video_id} true_ecr={true_ecr:.6f}",
                flush=True,
            )
            download_video(
                url=str(row["Download_link"]),
                target=video_path,
                timeout=args.download_timeout,
            )
            original = analyzer.analyze(
                video_path,
                title=title,
                description=description,
                video_id=video_id,
            )
            (json_dir / f"{video_id}_analysis.json").write_text(
                json.dumps(original, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            predicted = float(original["scores"]["student_ecr"])
            suggestion = original.get("metadata_suggestion", {})
            suggested_title = clean_text_value(suggestion.get("title")) or title
            suggested_description = clean_text_value(suggestion.get("description")) or description

            edit_plan = apply_feasible_video_edits(
                input_video=video_path,
                output_video=edited_path,
                result=original,
            )
            (json_dir / f"{video_id}_edit_plan.json").write_text(
                json.dumps(edit_plan, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            edited = analyzer.analyze(
                edited_path,
                title=suggested_title,
                description=suggested_description,
                video_id=f"{video_id}_auto_edited",
            )
            (json_dir / f"{video_id}_auto_edited_analysis.json").write_text(
                json.dumps(edited, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            after_ecr = float(edited["scores"]["student_ecr"])

            rows.append(
                {
                    "Id": video_id,
                    "Title": title,
                    "Description": description,
                    "True ECR": true_ecr,
                    "Predicted ECR": predicted,
                    "Suggested Title": suggested_title,
                    "Suggested Description": suggested_description,
                    "After-edit ECR": after_ecr,
                }
            )
            done_ids.add(video_id)
            write_outputs(rows, output_csv, output_xlsx)

            elapsed = max(time.time() - start_time, 1e-6)
            rate = (len(rows)) / elapsed
            remaining = max(args.target_successes - len(rows), 0)
            eta_min = remaining / rate / 60 if rate > 0 else 0.0
            print(
                f"success id={video_id} pred={predicted:.6f} after={after_ecr:.6f} "
                f"completed={len(rows)}/{args.target_successes} eta_min={eta_min:.1f}",
                flush=True,
            )
            if args.delete_videos:
                video_path.unlink(missing_ok=True)
                edited_path.unlink(missing_ok=True)
            clear_torch_cache()
        except Exception as exc:
            failure = {
                "Id": video_id,
                "candidate_index": int(i),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            }
            with failed_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            failed_ids.add(video_id)
            print(f"failed id={video_id} {type(exc).__name__}: {exc}", flush=True)
            clear_torch_cache()

    write_outputs(rows, output_csv, output_xlsx)
    print(
        f"batch_done successes={len(rows)} csv={output_csv} xlsx={output_xlsx} failures={failed_jsonl}",
        flush=True,
    )
    if len(rows) < args.target_successes:
        raise SystemExit(
            f"Only collected {len(rows)} successful rows. Increase --candidate-count and rerun."
        )


def select_candidates(
    *,
    train_csv: Path,
    official_dir: Path,
    n_candidates: int,
    seed: int,
    normal_mean: float,
    normal_std: float,
) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    df["Id"] = df["Id"].astype(str)
    df["ECR"] = pd.to_numeric(df["ECR"], errors="coerce")
    official_ids = read_official_ids(official_dir)
    eligible = df[
        df["ECR"].notna()
        & df["Download_link"].notna()
        & (df["Download_link"].astype(str).str.len() > 0)
        & ~df["Id"].isin(official_ids)
    ].copy()
    if len(eligible) < n_candidates:
        raise RuntimeError(f"Not enough eligible rows: {len(eligible)} < {n_candidates}")

    rng = np.random.default_rng(seed)
    eligible["_rand"] = rng.random(len(eligible))
    sorted_df = eligible.sort_values(["ECR", "_rand"]).reset_index(drop=True)
    ecr_values = sorted_df["ECR"].to_numpy(dtype=float)
    normal = NormalDist(mu=normal_mean, sigma=normal_std)
    targets = np.array(
        [
            min(1.0, max(0.0, normal.inv_cdf((i + 0.5) / n_candidates)))
            for i in range(n_candidates)
        ],
        dtype=float,
    )
    used: set[int] = set()
    picked_positions: list[int] = []
    for target in targets:
        pos = nearest_unused_position(ecr_values, target, used)
        used.add(pos)
        picked_positions.append(pos)

    selected = sorted_df.iloc[picked_positions].copy()
    selected["target_normal_ecr"] = targets
    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return selected.drop(columns=["_rand"], errors="ignore")


def nearest_unused_position(values: np.ndarray, target: float, used: set[int]) -> int:
    n = len(values)
    right = int(np.searchsorted(values, target, side="left"))
    left = right - 1
    while left >= 0 or right < n:
        left_ok = left >= 0 and left not in used
        right_ok = right < n and right not in used
        if left_ok and right_ok:
            if abs(float(values[left]) - target) <= abs(float(values[right]) - target):
                return left
            return right
        if left_ok:
            return left
        if right_ok:
            return right
        left -= 1
        right += 1
    raise RuntimeError("No unused candidate remains")


def read_official_ids(official_dir: Path) -> set[str]:
    ids: set[str] = set()
    for name in ["split_all_5000.csv", "train_4000.csv", "test_1000.csv"]:
        path = official_dir / name
        if path.exists():
            df = pd.read_csv(path, usecols=["Id"])
            ids.update(df["Id"].astype(str).tolist())
    for name in ["train_ids_4000.txt", "test_ids_1000.txt"]:
        path = official_dir / name
        if path.exists():
            ids.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return ids


def write_selection_summary(selected: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    summary = {
        "seed": args.seed,
        "candidate_count": int(len(selected)),
        "normal_mean": args.normal_mean,
        "normal_std": args.normal_std,
        "selected_ecr_summary": {
            key: float(value)
            for key, value in selected["ECR"].describe(
                percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            ).to_dict().items()
        },
    }
    (out_dir / "proper_kd_auto_edit_100_normal_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def download_video(*, url: str, target: Path, timeout: int) -> None:
    if target.exists() and target.stat().st_size > 1024:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "20",
        "--max-time",
        str(timeout),
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--retry-connrefused",
        "-A",
        "Mozilla/5.0",
        url,
        "-o",
        str(part),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError((result.stderr or f"curl_exit_{result.returncode}").strip())
    if not part.exists() or part.stat().st_size <= 1024:
        part.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is missing or too small")
    part.replace(target)


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


def load_failed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = str(row.get("Id") or "").strip()
            if video_id:
                ids.add(video_id)
    return ids


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, xlsx_path: Path) -> None:
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)


def clean_text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return " ".join(str(value).split())


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def clear_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()

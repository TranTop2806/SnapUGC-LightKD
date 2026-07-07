#!/usr/bin/env python3
"""Run the official SnapUGC EVQA baseline on a local/Kaggle video subset.

This intentionally uses the authors' released repository instead of a local
reimplementation, because the original SnapUGC pipeline depends on several
feature extractors and an EVQA architecture that are bundled there:

  - EfficientNetV2 semantic frame features
  - UVQ-style distortion frame features
  - ResNet3D-18 Kinetics action clip features
  - mPLUG-2 video caption and mid-layer features
  - YAMNet top-5 sound labels
  - Stable Diffusion tokenizer/text encoder for cross-attention text features
  - EVQA.pth official engagement checkpoint

The official script writes `submission_baseline.csv`; this wrapper prepares the
CSV, runs the script, copies the predictions, and evaluates them against ECR
when labels are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OFFICIAL_REPO = "https://github.com/dasongli1/SnapUGC_Engagement.git"
OFFICIAL_COMMIT = "4e0ce3154225cfdf1d036e5b8b1d3874615a04f7"
CHECKPOINT_DRIVE = (
    "https://drive.google.com/drive/folders/19_s6Z4R-iTaQHkRWFRn2Aby1FOy2cHes?usp=share_link"
)
REQUIRED_CHECKPOINTS = [
    "EVQA.pth",
    "net_distort6_g_latest.pth",
    "r3d18_K_200ep.pth",
    "mPLUG2_MSRVTT_Caption.pth",
    "ViT-L-14.tar",
]
EXTERNAL_PRETRAINED_WEIGHTS = [
    "efficientnet_v2_s_21k_ft1k-dbb43f38.pth",
]


def run(cmd, *, cwd=None, env=None):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def ensure_repo(repo_dir: Path, repo_url: str):
    if (repo_dir / "ECR_inference" / "test_SnapUGC_baseline.py").exists():
        if (repo_dir / ".git").exists():
            run(["git", "checkout", OFFICIAL_COMMIT], cwd=repo_dir)
        else:
            print(f"Using vendored official SnapUGC source: {repo_dir}", flush=True)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, repo_dir])
    run(["git", "checkout", OFFICIAL_COMMIT], cwd=repo_dir)


def maybe_download_checkpoints(ecr_dir: Path, enabled: bool):
    checkpoint_dir = ecr_dir / "checkpoints"
    missing = [name for name in REQUIRED_CHECKPOINTS if not (checkpoint_dir / name).exists()]
    if not missing:
        return
    if not enabled:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            "Missing official SnapUGC checkpoints in "
            f"{checkpoint_dir}: {missing_text}\n"
            f"Download them from {CHECKPOINT_DRIVE} and place them in that folder, "
            "or rerun with --download-checkpoints."
        )

    run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            CHECKPOINT_DRIVE,
            "-O",
            checkpoint_dir,
        ]
    )
    missing = [name for name in REQUIRED_CHECKPOINTS if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint download finished but these files are still missing: " + ", ".join(missing)
        )


def ensure_external_pretrained_weights(ecr_dir: Path):
    """Put external pretrained weights expected by the official code in torch hub cache."""
    checkpoint_dir = ecr_dir / "checkpoints"
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    cache_dir = torch_home / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in EXTERNAL_PRETRAINED_WEIGHTS:
        cached = cache_dir / name
        if cached.exists() and cached.stat().st_size > 0:
            continue
        src = checkpoint_dir / name
        if src.exists() and src.stat().st_size > 0:
            try:
                cached.symlink_to(src)
            except OSError:
                shutil.copy2(src, cached)
            print(f"Cached external pretrained weight: {src} -> {cached}", flush=True)
            continue
        missing.append(name)

    if missing:
        raise FileNotFoundError(
            "The official EfficientNetV2 semantic extractor needs an external ImageNet "
            "pretrained weight whose original OneDrive URL now returns 404. Add these "
            "file(s) to ECR_inference/checkpoints or the torch hub cache before rerunning: "
            + ", ".join(missing)
        )


def patch_official_code(
    ecr_dir: Path,
    *,
    light_sd_text_encoder: bool = True,
):
    """Patch only runtime compatibility issues in the official inference code."""
    from snapugc_lightkd.official_patching import patch_teacher_export

    script_path = ecr_dir / "test_SnapUGC_baseline.py"
    text = script_path.read_text()
    if light_sd_text_encoder and "StableDiffusionPipeline.from_pretrained" in text:
        text = text.replace(
            'from diffusers import StableDiffusionPipeline\npipe = StableDiffusionPipeline.from_pretrained(\n        "CompVis/stable-diffusion-v1-4"\n    )\nmodel = EVQA(3, 16, pipe.tokenizer, pipe.text_encoder)',
            'from transformers import CLIPTokenizer, CLIPTextModel\n_sd_model_id = "CompVis/stable-diffusion-v1-4"\n_sd_tokenizer = CLIPTokenizer.from_pretrained(_sd_model_id, subfolder="tokenizer")\n_sd_text_encoder = CLIPTextModel.from_pretrained(_sd_model_id, subfolder="text_encoder").cuda().eval()\nmodel = EVQA(3, 16, _sd_tokenizer, _sd_text_encoder)',
        )

    replacements = {
        'torch.load("checkpoints/EVQA.pth")': 'torch.load("checkpoints/EVQA.pth", weights_only=False)',
        "torch.load(\"checkpoints/mPLUG2_MSRVTT_Caption.pth\", map_location='cpu')": "torch.load(\"checkpoints/mPLUG2_MSRVTT_Caption.pth\", map_location='cpu', weights_only=False)",
        'torch.load("checkpoints/net_distort6_g_latest.pth")': 'torch.load("checkpoints/net_distort6_g_latest.pth", weights_only=False)',
        'torch.load("checkpoints/r3d18_K_200ep.pth")': 'torch.load("checkpoints/r3d18_K_200ep.pth", weights_only=False)',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Batch size for frame-level feature extraction. Lowering this does not
    # change the architecture or weights; it only reduces peak VRAM on L4/T4.
    text = text.replace(
        "num_one_running = 48",
        "num_one_running = int(os.environ.get('SNAPUGC_OFFICIAL_FRAME_BATCH', '24'))",
    )
    text = text.replace(
        "bs = 4",
        "bs = int(os.environ.get('SNAPUGC_MPLUG_CLIP_BATCH', '4'))",
    )
    text = text.replace(
        "array_ = read_data(path, num_frame=32)",
        'caption_num_frames = int(os.environ.get("SNAPUGC_CAPTION_NUM_FRAMES", "8"))\n'
        "        array_ = read_data(path, num_frame=caption_num_frames)",
    )
    text = text.replace(
        "DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,collate_fn=None,pin_memory=False)",
        "DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(os.environ.get('SNAPUGC_DATALOADER_WORKERS', '1')), collate_fn=None, pin_memory=False)",
    )
    if "@torch.inference_mode()\ndef calculate" not in text:
        text = text.replace(
            "def calculate(videos_dir, videos_files):",
            "@torch.inference_mode()\ndef calculate(videos_dir, videos_files):",
        )
    if "torch.cuda.empty_cache()" not in text:
        text = text.replace(
            "        print(idx, video_id, out0_val)",
            "        print(idx, video_id, out0_val)\n        torch.cuda.empty_cache()",
        )

    # Recent CLIPTextModel versions may not expose position_ids as a loadable
    # buffer, while the official checkpoint contains it. This does not change
    # learned weights, but avoids a strict-loading compatibility crash.
    text = text.replace(
        "model.load_state_dict(torch.load(\"checkpoints/EVQA.pth\", weights_only=False)['params'])",
        "model.load_state_dict(torch.load(\"checkpoints/EVQA.pth\", weights_only=False)['params'], strict=False)",
    )
    text = text.replace(
        "model.load_state_dict(torch.load(\"checkpoints/EVQA.pth\")['params'])",
        "model.load_state_dict(torch.load(\"checkpoints/EVQA.pth\", weights_only=False)['params'], strict=False)",
    )
    script_path.write_text(text)

    modeling_path = ecr_dir / "mPLUG_2" / "models" / "modeling_mplug2.py"
    modeling_text = modeling_path.read_text()
    old_import = """from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)"""
    new_import = """from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.modeling_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
except ImportError:
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )"""
    if old_import in modeling_text:
        modeling_path.write_text(modeling_text.replace(old_import, new_import))

    tokenizer_path = ecr_dir / "mPLUG_2" / "models" / "tokenization_bert.py"
    tokenizer_text = tokenizer_path.read_text()
    old_tokenizer_init = """        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])"""
    new_tokenizer_init = """        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )"""
    if old_tokenizer_init in tokenizer_text:
        tokenizer_path.write_text(tokenizer_text.replace(old_tokenizer_init, new_tokenizer_init))

    patch_teacher_export(ecr_dir)


def prepare_official_csv(input_csv: Path, output_csv: Path, max_samples: int | None = None):
    rows = []
    labels = {}
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Id", "Title", "Description"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")
        for row in reader:
            if max_samples is not None and len(rows) >= max_samples:
                break
            vid = str(row["Id"])
            rows.append(
                {
                    "Id": vid,
                    "Title": row.get("Title", "") or "",
                    "Description": row.get("Description", "") or "",
                    "Download_link": row.get("Download_link", "") or "",
                }
            )
            if row.get("ECR") not in (None, ""):
                labels[vid] = float(row["ECR"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Id", "Title", "Description", "Download_link"])
        writer.writeheader()
        writer.writerows(rows)
    return rows, labels


def count_existing_videos(rows, videos_dir: Path):
    found = 0
    missing = []
    for row in rows:
        path = videos_dir / f"{row['Id']}.mp4"
        if path.exists():
            found += 1
        elif len(missing) < 10:
            missing.append(str(path))
    return found, missing


def load_predictions(path: Path):
    preds = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            preds[str(row["Id"])] = float(row["ECR"])
    return preds


def evaluate(preds, labels):
    ids = [vid for vid in preds if vid in labels]
    pred = np.array([preds[vid] for vid in ids], dtype=np.float64)
    true = np.array([labels[vid] for vid in ids], dtype=np.float64)
    if len(ids) < 3:
        return {"n_eval": len(ids)}
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
    metrics["final_score"] = metrics["final_score_srcc06_plcc04"]
    metrics["final_score_formula"] = "0.6*SRCC + 0.4*PLCC"
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run official SnapUGC EVQA inference and evaluation."
    )
    parser.add_argument("--official-repo-dir", default="/kaggle/working/SnapUGC_Engagement")
    parser.add_argument("--repo-url", default=OFFICIAL_REPO)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument(
        "--csv-file", required=True, help="CSV with Id, Title, Description, optional ECR."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--no-light-sd-text-encoder", action="store_true")
    parser.add_argument(
        "--export-artifacts",
        action="store_true",
        help="Export hidden states, clip outputs, text/caption features, and attention shards.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for teacher artifact shards. Defaults to OUT_DIR/teacher_artifacts.",
    )
    parser.add_argument("--artifact-shard-size", type=int, default=500)
    parser.add_argument("--skip-run", action="store_true", help="Only prepare CSV/checks.")
    args = parser.parse_args()

    repo_dir = Path(args.official_repo_dir).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    csv_file = Path(args.csv_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    artifact_dir = (
        Path(args.artifact_dir).resolve() if args.artifact_dir else out_dir / "teacher_artifacts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_repo(repo_dir, args.repo_url)
    ecr_dir = repo_dir / "ECR_inference"
    if not args.skip_run:
        maybe_download_checkpoints(ecr_dir, args.download_checkpoints)
        ensure_external_pretrained_weights(ecr_dir)
        patch_official_code(
            ecr_dir,
            light_sd_text_encoder=not args.no_light_sd_text_encoder,
        )

    official_csv = out_dir / "official_input.csv"
    rows, labels = prepare_official_csv(csv_file, official_csv, args.max_samples)
    found, missing = count_existing_videos(rows, videos_dir)
    print(f"Prepared {len(rows)} rows. Found videos: {found}/{len(rows)}", flush=True)
    if missing:
        print("First missing videos:", flush=True)
        for path in missing:
            print(f"  {path}", flush=True)
    if found == 0 and not args.skip_run:
        raise FileNotFoundError(f"No mp4 videos found under {videos_dir}")

    if not args.skip_run:
        submission_path = ecr_dir / "submission_baseline.csv"
        if submission_path.exists():
            submission_path.unlink()
        run_env = os.environ.copy()
        run_env["SNAPUGC_REPO_ROOT"] = str(ROOT_DIR.resolve())
        existing_pythonpath = run_env.get("PYTHONPATH")
        run_env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(SRC_DIR), existing_pythonpath) if value
        )
        if args.export_artifacts:
            run_env["SNAPUGC_EXPORT_ARTIFACTS"] = "1"
            run_env["SNAPUGC_ARTIFACT_DIR"] = str(artifact_dir)
            run_env["SNAPUGC_ARTIFACT_SHARD_SIZE"] = str(args.artifact_shard_size)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            print(f"Teacher artifact export enabled: {artifact_dir}", flush=True)
        run(
            [
                args.python,
                "test_SnapUGC_baseline.py",
                "--videos_dir",
                videos_dir,
                "--csv_file",
                official_csv,
            ],
            cwd=ecr_dir,
            env=run_env,
        )
        if not submission_path.exists():
            raise FileNotFoundError(f"Official script did not create {submission_path}")
        out_submission = out_dir / "official_submission_baseline.csv"
        shutil.copy2(submission_path, out_submission)
        preds = load_predictions(out_submission)
        input_ids = {row["Id"] for row in rows}
        pred_ids = set(preds)
        if input_ids != pred_ids:
            missing_ids = sorted(input_ids - pred_ids)[:10]
            extra_ids = sorted(pred_ids - input_ids)[:10]
            raise RuntimeError(
                "Official prediction IDs do not match the requested subset. "
                f"missing={missing_ids}, extra={extra_ids}"
            )
        metrics = evaluate(preds, labels)
        report = {
            "source": "official dasongli1/SnapUGC_Engagement ECR_inference",
            "official_repo_dir": str(repo_dir),
            "videos_dir": str(videos_dir),
            "csv_file": str(csv_file),
            "official_input_csv": str(official_csv),
            "submission": str(out_submission),
            "n_rows": len(rows),
            "n_videos_found": found,
            "export_artifacts": bool(args.export_artifacts),
            "artifact_dir": str(artifact_dir) if args.export_artifacts else None,
            "artifact_shard_size": args.artifact_shard_size if args.export_artifacts else None,
            "metrics": metrics,
        }
        with (out_dir / "official_evqa_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report["metrics"], indent=2), flush=True)
        print(f"Saved report: {out_dir / 'official_evqa_report.json'}", flush=True)


if __name__ == "__main__":
    main()

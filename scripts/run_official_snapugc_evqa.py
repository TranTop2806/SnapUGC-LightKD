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


OFFICIAL_REPO = "https://github.com/dasongli1/SnapUGC_Engagement.git"
OFFICIAL_COMMIT = "4e0ce3154225cfdf1d036e5b8b1d3874615a04f7"
CHECKPOINT_DRIVE = "https://drive.google.com/drive/folders/19_s6Z4R-iTaQHkRWFRn2Aby1FOy2cHes?usp=share_link"
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
        run(["git", "checkout", OFFICIAL_COMMIT], cwd=repo_dir)
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
    run([
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        CHECKPOINT_DRIVE,
        "-O",
        checkpoint_dir,
    ])
    missing = [name for name in REQUIRED_CHECKPOINTS if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint download finished but these files are still missing: "
            + ", ".join(missing)
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
    script_path = ecr_dir / "test_SnapUGC_baseline.py"
    text = script_path.read_text()
    if light_sd_text_encoder and "StableDiffusionPipeline.from_pretrained" in text:
        text = text.replace(
            'from diffusers import StableDiffusionPipeline\npipe = StableDiffusionPipeline.from_pretrained(\n        "CompVis/stable-diffusion-v1-4"\n    )\nmodel = EVQA(3, 16, pipe.tokenizer, pipe.text_encoder)',
            'from transformers import CLIPTokenizer, CLIPTextModel\n_sd_model_id = "CompVis/stable-diffusion-v1-4"\n_sd_tokenizer = CLIPTokenizer.from_pretrained(_sd_model_id, subfolder="tokenizer")\n_sd_text_encoder = CLIPTextModel.from_pretrained(_sd_model_id, subfolder="text_encoder").cuda().eval()\nmodel = EVQA(3, 16, _sd_tokenizer, _sd_text_encoder)',
        )

    replacements = {
        'torch.load("checkpoints/EVQA.pth")':
            'torch.load("checkpoints/EVQA.pth", weights_only=False)',
        'torch.load("checkpoints/mPLUG2_MSRVTT_Caption.pth", map_location=\'cpu\')':
            'torch.load("checkpoints/mPLUG2_MSRVTT_Caption.pth", map_location=\'cpu\', weights_only=False)',
        'torch.load("checkpoints/net_distort6_g_latest.pth")':
            'torch.load("checkpoints/net_distort6_g_latest.pth", weights_only=False)',
        'torch.load("checkpoints/r3d18_K_200ep.pth")':
            'torch.load("checkpoints/r3d18_K_200ep.pth", weights_only=False)',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Batch size for frame-level feature extraction. Lowering this does not
    # change the architecture or weights; it only reduces peak VRAM on L4/T4.
    text = text.replace(
        "num_one_running = 48",
        "num_one_running = int(os.environ.get('SNAPUGC_OFFICIAL_FRAME_BATCH', '24'))",
    )

    # Recent CLIPTextModel versions may not expose position_ids as a loadable
    # buffer, while the official checkpoint contains it. This does not change
    # learned weights, but avoids a strict-loading compatibility crash.
    text = text.replace(
        'model.load_state_dict(torch.load("checkpoints/EVQA.pth", weights_only=False)[\'params\'])',
        'model.load_state_dict(torch.load("checkpoints/EVQA.pth", weights_only=False)[\'params\'], strict=False)',
    )
    text = text.replace(
        'model.load_state_dict(torch.load("checkpoints/EVQA.pth")[\'params\'])',
        'model.load_state_dict(torch.load("checkpoints/EVQA.pth", weights_only=False)[\'params\'], strict=False)',
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
    old_tokenizer_init = '''        super().__init__(
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
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])'''
    new_tokenizer_init = '''        if not os.path.isfile(vocab_file):
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
        )'''
    if old_tokenizer_init in tokenizer_text:
        tokenizer_path.write_text(tokenizer_text.replace(old_tokenizer_init, new_tokenizer_init))

    patch_official_artifact_export(ecr_dir)


def patch_official_artifact_export(ecr_dir: Path):
    """Add env-gated artifact export hooks without changing scalar predictions."""
    evqa_path = ecr_dir / "modules" / "EVQA.py"
    evqa_text = evqa_path.read_text()
    if "import os\n" not in evqa_text.splitlines()[:10]:
        evqa_text = evqa_text.replace("import torch\n", "import os\nimport torch\n", 1)

    attention_old = """        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)"""
    attention_new = """        attn = attn.softmax(dim=-1)
        if os.environ.get("SNAPUGC_EXPORT_ARTIFACTS") == "1":
            self.last_attn = attn.detach().to(torch.float16).cpu()
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)"""
    if attention_old in evqa_text and "self.last_attn = attn.detach()" not in evqa_text:
        evqa_text = evqa_text.replace(attention_old, attention_new, 1)

    forward_old = """        feat_4 = feat4
        feat_3_embed = self.feat3_preprocess(feat_3)
        feat_3 = self.fc30(feat_3)
        feat_4 = self.fc4(feat_4)"""
    forward_new = """        caption_feature_raw = feat4
        feat_4 = feat4
        feat_3_embed = self.feat3_preprocess(feat_3)
        feat_3 = self.fc30(feat_3)
        feat_4 = self.fc4(feat_4)"""
    if forward_old in evqa_text and "caption_feature_raw = feat4" not in evqa_text:
        evqa_text = evqa_text.replace(forward_old, forward_new, 1)

    output_old = """        feat123 = self.fc_merge123(torch.cat((feat_12, feat_3, feat_4, music_feature1, music_feature2_, music_feature3_, music_feature4), dim=1))
        feat123 = feat123.unsqueeze(0)
        out = self.block1(feat123).squeeze(0)
        temp_out = self.out(out)
        temp_out = torch.mean(temp_out, dim=0)
        return temp_out"""
    output_new = """        fusion_hidden = self.fc_merge123(torch.cat((feat_12, feat_3, feat_4, music_feature1, music_feature2_, music_feature3_, music_feature4), dim=1))
        feat123 = fusion_hidden.unsqueeze(0)
        out = self.block1(feat123).squeeze(0)
        clip_ecr = self.out(out).view(-1)
        temp_out = torch.mean(clip_ecr, dim=0).view(1)
        if os.environ.get("SNAPUGC_EXPORT_ARTIFACTS") == "1":
            attention_mean = []
            for block in self.block1:
                attn = getattr(block.attn, "last_attn", None)
                if attn is not None:
                    attention_mean.append(attn.squeeze(0).mean(dim=0).to(torch.float16))
            if attention_mean:
                attention_mean = torch.stack(attention_mean, dim=0)
                attention_importance = attention_mean.mean(dim=1)
            else:
                attention_mean = torch.empty(0, dtype=torch.float16)
                attention_importance = torch.empty(0, dtype=torch.float16)
            text_tokens = torch.stack(
                (
                    text_embedding1[0],
                    text_embedding2[0],
                    text_embedding3[0],
                    text_embedding4[0],
                ),
                dim=0,
            )
            self.last_artifacts = {
                "fusion_hidden": fusion_hidden.detach().to(torch.float16).cpu(),
                "temporal_hidden": out.detach().to(torch.float16).cpu(),
                "clip_ecr": clip_ecr.detach().to(torch.float16).cpu(),
                "caption_feature": caption_feature_raw.detach().to(torch.float16).cpu(),
                "action_feature": feat_3_embed.detach().to(torch.float16).cpu(),
                "frame_fusion_feature": feat_12.detach().to(torch.float16).cpu(),
                "text_tokens": text_tokens.detach().to(torch.float16).cpu(),
                "text_pooled": text_tokens.mean(dim=1).detach().to(torch.float16).cpu(),
                "attention_mean": attention_mean.cpu(),
                "attention_importance": attention_importance.cpu(),
            }
        return temp_out"""
    if output_old in evqa_text and "self.last_artifacts" not in evqa_text:
        evqa_text = evqa_text.replace(output_old, output_new, 1)
    evqa_path.write_text(evqa_text)

    script_path = ecr_dir / "test_SnapUGC_baseline.py"
    text = script_path.read_text()
    helper_marker = "def _snapugc_save_teacher_artifact"
    helper = r'''
SNAPUGC_ARTIFACT_DIR = os.environ.get("SNAPUGC_ARTIFACT_DIR")
SNAPUGC_ARTIFACT_SHARD_SIZE = int(os.environ.get("SNAPUGC_ARTIFACT_SHARD_SIZE", "500"))
SNAPUGC_ARTIFACT_ROWS = []

def _snapugc_to_numpy(value, dtype=np.float16):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)

def _snapugc_pack_ragged(rows, key):
    arrays = []
    offsets = [0]
    shapes = []
    for row in rows:
        arr = row.get(key)
        if arr is None:
            arr = np.zeros((0,), dtype=np.float16)
        arr = np.asarray(arr)
        arrays.append(arr.reshape(-1))
        shapes.append(arr.shape)
        offsets.append(offsets[-1] + arrays[-1].size)
    flat = np.concatenate(arrays, axis=0) if arrays else np.zeros((0,), dtype=np.float16)
    return flat, np.asarray(offsets, dtype=np.int64), np.asarray(shapes, dtype=np.int32)

def _snapugc_flush_teacher_artifacts(force=False):
    if not SNAPUGC_ARTIFACT_DIR or not SNAPUGC_ARTIFACT_ROWS:
        return
    if not force and len(SNAPUGC_ARTIFACT_ROWS) < SNAPUGC_ARTIFACT_SHARD_SIZE:
        return
    rows = list(SNAPUGC_ARTIFACT_ROWS)
    SNAPUGC_ARTIFACT_ROWS.clear()
    out_dir = Path(SNAPUGC_ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_idx = int(rows[0]["idx"])
    end_idx = int(rows[-1]["idx"])
    prefix = f"official_teacher_artifacts_{start_idx:04d}_{end_idx:04d}"
    payload = {
        "ids": np.asarray([row["Id"] for row in rows], dtype="<U32"),
        "order_idx": np.asarray([row["idx"] for row in rows], dtype=np.int32),
        "teacher_ecr": np.asarray([row["teacher_ecr"] for row in rows], dtype=np.float32),
    }
    for key in (
        "clip_ecr",
        "fusion_hidden",
        "temporal_hidden",
        "caption_feature",
        "action_feature",
        "frame_fusion_feature",
        "text_tokens",
        "text_pooled",
        "attention_mean",
        "attention_importance",
    ):
        flat, offsets, shapes = _snapugc_pack_ragged(rows, key)
        payload[f"{key}_flat"] = flat
        payload[f"{key}_offsets"] = offsets
        payload[f"{key}_shapes"] = shapes
    np.savez_compressed(out_dir / f"{prefix}.npz", **payload)
    with (out_dir / f"{prefix}_captions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(
                {
                    "idx": int(row["idx"]),
                    "Id": row["Id"],
                    "caption": row.get("caption", ""),
                },
                ensure_ascii=False,
            ) + "\n")
    print(f"saved_teacher_artifact_shard {prefix} n={len(rows)} dir={out_dir}", flush=True)

def _snapugc_save_teacher_artifact(idx, video_id, teacher_ecr, model, caption):
    if not SNAPUGC_ARTIFACT_DIR:
        return
    artifacts = getattr(model, "last_artifacts", None)
    if artifacts is None:
        print(f"missing_teacher_artifacts {idx} {video_id}", flush=True)
        return
    row = {
        "idx": int(idx),
        "Id": str(video_id),
        "teacher_ecr": float(teacher_ecr),
        "caption": caption,
    }
    for key, value in artifacts.items():
        row[key] = _snapugc_to_numpy(value)
    SNAPUGC_ARTIFACT_ROWS.append(row)
    _snapugc_flush_teacher_artifacts(force=False)
'''
    if helper_marker not in text:
        text = text.replace("import csv\n", "import csv\nimport json\n", 1)
        text = text.replace("from pathlib import Path\n", "from pathlib import Path\n" + helper + "\n", 1)

    save_old = """        out0_val = out0_mean.clamp(0.0, 1.0).item()
        # mos0_val = mos_label[0].clamp(0.0,1.0).item() # / 20.0"""
    save_new = """        out0_val = out0_mean.clamp(0.0, 1.0).item()
        _snapugc_save_teacher_artifact(idx, video_id, out0_val, model, caption)
        # mos0_val = mos_label[0].clamp(0.0,1.0).item() # / 20.0"""
    save_call_marker = "_snapugc_save_teacher_artifact(idx, video_id, out0_val"
    if save_old in text and save_call_marker not in text:
        text = text.replace(save_old, save_new, 1)
    if save_call_marker not in text:
        text = text.replace(
            "        out0_val = out0_mean.clamp(0.0, 1.0).item()\n",
            "        out0_val = out0_mean.clamp(0.0, 1.0).item()\n"
            "        _snapugc_save_teacher_artifact(idx, video_id, out0_val, model, caption)\n",
            1,
        )

    flush_old = """    with open("submission_baseline.csv", "w", newline="") as csvfile:"""
    flush_new = """    _snapugc_flush_teacher_artifacts(force=True)
    with open("submission_baseline.csv", "w", newline="") as csvfile:"""
    if flush_old in text and "_snapugc_flush_teacher_artifacts(force=True)" not in text:
        text = text.replace(flush_old, flush_new, 1)
    script_path.write_text(text)


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
            rows.append({
                "Id": vid,
                "Title": row.get("Title", "") or "",
                "Description": row.get("Description", "") or "",
                "Download_link": row.get("Download_link", "") or "",
            })
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
    parser = argparse.ArgumentParser(description="Run official SnapUGC EVQA inference and evaluation.")
    parser.add_argument("--official-repo-dir", default="/kaggle/working/SnapUGC_Engagement")
    parser.add_argument("--repo-url", default=OFFICIAL_REPO)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--csv-file", required=True, help="CSV with Id, Title, Description, optional ECR.")
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
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else out_dir / "teacher_artifacts"
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
        if args.export_artifacts:
            run_env["SNAPUGC_EXPORT_ARTIFACTS"] = "1"
            run_env["SNAPUGC_ARTIFACT_DIR"] = str(artifact_dir)
            run_env["SNAPUGC_ARTIFACT_SHARD_SIZE"] = str(args.artifact_shard_size)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            print(f"Teacher artifact export enabled: {artifact_dir}", flush=True)
        run([
            args.python,
            "test_SnapUGC_baseline.py",
            "--videos_dir",
            videos_dir,
            "--csv_file",
            official_csv,
        ], cwd=ecr_dir, env=run_env)
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

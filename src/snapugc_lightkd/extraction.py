"""
Final paper-aligned feature extraction for Distil-ShortVU.

Feature groups:
  - CLIP ViT-B/16 frame tokens: visual semantics
  - R(2+1)D-18 Kinetics-400 clip tokens: motion/action
  - DOVER scores from a precomputed CSV: technical/aesthetic/overall video quality
  - YAMNet AudioSet embedding/probabilities: background sound
  - Sentence-T5-base embeddings: metadata, caption, rationale text
  - BLIP-base lightweight frame captioning

This script is designed for Kaggle first, but keeps fallbacks so the pipeline can
smoke-test locally when heavy dependencies are unavailable.
"""

import argparse
import gc
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class VideoItem:
    video_id: str
    video_path: str
    ecr: Optional[float]
    title: str
    description: str
    hashtags: str


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


class VideoReaderHelper:
    @staticmethod
    def _window_frame_count(total: int, fps: float, opening_seconds: Optional[float]) -> int:
        if opening_seconds is None:
            return total
        try:
            opening_seconds = float(opening_seconds)
        except (TypeError, ValueError):
            return total
        if opening_seconds <= 0 or fps <= 0:
            return total
        return max(1, min(total, int(np.ceil(opening_seconds * fps))))

    def read_frames(
        self,
        video_path: str,
        num_frames: int,
        opening_seconds: Optional[float] = None,
    ) -> Tuple[List[Image.Image], Dict]:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return [], {"num_frames": 0, "fps": 0.0, "duration": 0.0, "width": 0, "height": 0}

        fps = float(vr.get_avg_fps() or 0.0)
        window_total = self._window_frame_count(total, fps, opening_seconds)
        indices = np.linspace(0, window_total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in frames]
        h, w = frames[0].shape[:2]
        duration = float(total / fps) if fps > 0 else 0.0
        meta = {
            "num_frames": int(total),
            "feature_num_frames": int(window_total),
            "fps": fps,
            "duration": duration,
            "feature_window_seconds": float(opening_seconds) if opening_seconds is not None else duration,
            "width": int(w),
            "height": int(h),
        }
        return pil_frames, meta

    def read_motion_clips(
        self,
        video_path: str,
        num_clips: int,
        frames_per_clip: int,
        opening_seconds: Optional[float] = None,
    ) -> List[np.ndarray]:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return []
        fps = float(vr.get_avg_fps() or 0.0)
        total = self._window_frame_count(total, fps, opening_seconds)

        if total < frames_per_clip:
            base = np.linspace(0, total - 1, frames_per_clip, dtype=int)
            return [vr.get_batch(base).asnumpy()]

        centers = np.linspace(frames_per_clip // 2, total - frames_per_clip // 2 - 1, num_clips, dtype=int)
        clips = []
        for center in centers:
            start = max(0, int(center) - frames_per_clip // 2)
            end = min(total, start + frames_per_clip)
            start = max(0, end - frames_per_clip)
            idx = np.arange(start, end, dtype=int)
            if len(idx) < frames_per_clip:
                idx = np.pad(idx, (0, frames_per_clip - len(idx)), mode="edge")
            clips.append(vr.get_batch(idx).asnumpy())
        return clips


class CLIPFrameExtractor:
    def __init__(self, model_id="openai/clip-vit-base-patch16", device=DEVICE):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        print(f"[CLIP] loaded {model_id} on {device}")

    @torch.no_grad()
    def encode(self, frames: Sequence[Image.Image]) -> np.ndarray:
        if not frames:
            return np.zeros((0, 512), dtype=np.float32)
        inputs = self.processor(images=list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.get_image_features(**inputs)
        if hasattr(output, "image_embeds"):
            output = output.image_embeds
        elif hasattr(output, "pooler_output"):
            output = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state[:, 0]
        if not torch.is_tensor(output):
            raise TypeError(f"Unexpected CLIP output type: {type(output)}")
        feats = output.detach().cpu().float().numpy()
        return l2_normalize(feats).astype(np.float32)

    def unload(self):
        del self.model, self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MotionExtractor:
    def __init__(self, device=DEVICE):
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

        self.device = device
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        self.model = r2plus1d_18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
        print(f"[Motion] R(2+1)D-18 Kinetics-400 loaded on {device}")

    def _preprocess_clip(self, clip: np.ndarray) -> torch.Tensor:
        import torch.nn.functional as F

        # clip: T,H,W,C -> 1,C,T,H,W
        x = torch.from_numpy(clip).float() / 255.0
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        top = (128 - 112) // 2
        left = (128 - 112) // 2
        x = x[:, :, top:top + 112, left:left + 112]
        x = x.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def encode(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.zeros((0, 512), dtype=np.float32)
        outputs = []
        for clip in clips:
            x = self._preprocess_clip(clip).to(self.device)
            feat = self.model(x).detach().cpu().float().numpy()
            outputs.append(feat[0])
        return l2_normalize(np.stack(outputs, axis=0)).astype(np.float32)

    def unload(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DoverScoreLookup:
    """Reads precomputed DOVER scores. Missing rows fall back to neutral values."""

    def __init__(self, csv_path: Optional[str] = None):
        self.by_id: Dict[str, Dict[str, float]] = {}
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            id_col = self._find_col(df, ["video_id", "id", "name", "filename", "path"])
            if id_col is None:
                print(f"[DOVER] Could not find id column in {csv_path}; using neutral scores")
                return
            for _, row in df.iterrows():
                vid = str(row[id_col])
                vid = os.path.splitext(os.path.basename(vid))[0]
                self.by_id[vid] = {
                    "technical": self._value(row, ["technical score", "technical", "tech"], 0.5),
                    "aesthetic": self._value(row, ["aesthetic score", "aesthetic", "aes"], 0.5),
                    "overall": self._value(row, ["overall/final score", "overall", "final"], 0.5),
                    "found": True,
                }
            print(f"[DOVER] loaded {len(self.by_id)} score rows from {csv_path}")
        else:
            print("[DOVER] no score CSV provided; using neutral scores")

    @staticmethod
    def _find_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        lowered = {c.lower(): c for c in df.columns}
        for name in names:
            if name in lowered:
                return lowered[name]
        for c in df.columns:
            lc = c.lower()
            if any(name in lc for name in names):
                return c
        return None

    @staticmethod
    def _value(row, names: Sequence[str], default: float) -> float:
        for col in row.index:
            lc = col.lower()
            if any(name in lc for name in names):
                value = safe_float(row[col], default)
                if value is None:
                    return default
                # DOVER outputs may be 0-1 or 0-100 depending on script/config.
                if value > 10:
                    value = value / 100.0
                elif value > 1:
                    value = value / 10.0
                return float(np.clip(value, 0.0, 1.0))
        return default

    def score(self, video_id: str) -> Dict[str, float]:
        return self.by_id.get(
            str(video_id),
            {"technical": 0.5, "aesthetic": 0.5, "overall": 0.5, "found": False},
        )


class YAMNetExtractor:
    def __init__(self):
        try:
            import tensorflow as tf

            try:
                # Keep TensorFlow/YAMNet on CPU so PyTorch models own the Kaggle GPUs.
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            import tensorflow_hub as hub
            self.hub = hub
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map = self.model.class_map_path().numpy().decode("utf-8")
            self.class_names = pd.read_csv(class_map)["display_name"].tolist()
            self.available = True
            print("[YAMNet] loaded from TensorFlow Hub")
        except Exception as exc:
            print(f"[YAMNet] unavailable ({exc}); using zeros")
            self.available = False
            self.model = None
            self.class_names = []

    def _load_audio(self, video_path: str) -> Optional[np.ndarray]:
        from scipy.io import wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
            "-ac", "1", "-ar", "16000", "-f", "wav", wav_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            sr, wav = wavfile.read(wav_path)
            if wav.dtype != np.float32:
                maxv = np.iinfo(wav.dtype).max if np.issubdtype(wav.dtype, np.integer) else 1.0
                wav = wav.astype(np.float32) / maxv
            return wav.astype(np.float32)
        except Exception:
            return None
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def encode(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        if not self.available:
            return np.zeros(1024, dtype=np.float32), np.zeros(521, dtype=np.float32), []
        wav = self._load_audio(video_path)
        if wav is None or len(wav) == 0:
            return np.zeros(1024, dtype=np.float32), np.zeros(521, dtype=np.float32), []
        try:
            scores, embeddings, _ = self.model(wav)
            scores = scores.numpy()
            embeddings = embeddings.numpy()
            probs = scores.mean(axis=0).astype(np.float32)
            emb = embeddings.mean(axis=0).astype(np.float32)
            top_idx = probs.argsort()[-5:][::-1]
            top_classes = [
                {"label": self.class_names[i] if i < len(self.class_names) else str(i), "prob": float(probs[i])}
                for i in top_idx
            ]
            return emb, probs, top_classes
        except Exception:
            return np.zeros(1024, dtype=np.float32), np.zeros(521, dtype=np.float32), []


class SentenceT5Encoder:
    def __init__(self, model_id="sentence-transformers/sentence-t5-base", device=DEVICE):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Text] loaded {model_id} ({self.dim}d)")

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


class BLIPCaptioner:
    def __init__(
        self,
        model_id="Salesforce/blip-image-captioning-base",
        device=DEVICE,
        enabled=True,
        strict=False,
    ):
        self.enabled = enabled
        self.available = False
        self.strict = strict
        self.device = device
        if not enabled:
            print("[BLIP] disabled")
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
            self.available = True
            print(f"[BLIP] loaded {model_id} on {device}")
        except Exception as exc:
            message = f"[BLIP] unavailable ({exc}); captions will be empty"
            if strict:
                raise RuntimeError(message) from exc
            print(message)

    @torch.no_grad()
    def caption(self, frames: Sequence[Image.Image], num_frames: int = 3) -> Tuple[str, List[str]]:
        if not self.available or not frames:
            return "", []
        try:
            count = max(1, min(int(num_frames), len(frames)))
            indices = np.linspace(0, len(frames) - 1, count, dtype=int)
            selected = [frames[int(i)].convert("RGB") for i in indices]
            inputs = self.processor(images=selected, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            generated = self.model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=32,
            )
            captions = [
                self.processor.decode(ids, skip_special_tokens=True).strip()
                for ids in generated
            ]
            captions = [c for c in captions if c]
            if self.strict and not captions:
                raise RuntimeError("BLIP produced empty captions")
            unique = list(dict.fromkeys(captions))
            if not unique:
                return "", []
            caption = " | ".join(unique)
            return caption, unique
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"BLIP captioning failed: {exc}") from exc
            tqdm.write(f"[BLIP] failed: {exc}")
            return "", []


def build_rationale(
    item: VideoItem,
    caption: str,
    quality: Dict[str, float],
    audio_top: Sequence[Dict],
) -> str:
    parts = []
    if caption:
        parts.append(f"Visual caption: {caption}.")
    metadata = " | ".join([p for p in [item.title, item.description, item.hashtags] if p.strip()])
    if metadata:
        parts.append(f"Metadata context: {metadata}.")
    parts.append(
        "Quality cues: "
        f"aesthetic={quality.get('aesthetic', 0.5):.3f}, "
        f"technical={quality.get('technical', 0.5):.3f}, "
        f"overall={quality.get('overall', 0.5):.3f}."
    )
    if audio_top:
        labels = [str(row.get("label", "")).strip() for row in audio_top[:3] if row.get("label")]
        if labels:
            parts.append(f"Audio cues: {', '.join(labels)}.")
    return " ".join(parts)


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def find_first_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lowered = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for col in df.columns:
        low = str(col).lower()
        if any(name.lower() in low for name in names):
            return col
    return None


def build_video_index(video_dir: str) -> Dict[str, str]:
    """Recursively index videos by filename stem and filename for Kaggle layout variants."""
    index: Dict[str, str] = {}
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if not filename.lower().endswith(VIDEO_EXTENSIONS):
                continue
            path = os.path.join(root, filename)
            stem = os.path.splitext(filename)[0]
            index.setdefault(stem, path)
            index.setdefault(filename, path)
    return index


def build_work(csv_path: str, video_dir: str, max_videos: int, existing_ids: set) -> List[VideoItem]:
    df = pd.read_csv(csv_path)
    id_col = find_first_column(df, ["Id", "video_id", "videoid", "uid"])
    ecr_col = find_first_column(df, ["ECR", "engagement", "label", "target"])
    title_col = find_first_column(df, ["Title", "title"])
    desc_col = find_first_column(df, ["Description", "description", "desc"])
    hashtag_col = find_first_column(df, ["Hashtags", "hashtags", "hashtag"])
    path_col = find_first_column(df, ["video_path", "path", "filename", "file", "video"])

    print(f"CSV columns: {list(df.columns)}")
    print(f"Using columns: id={id_col}, ecr={ecr_col}, title={title_col}, description={desc_col}")

    if id_col is None:
        raise ValueError("Could not find video id column in CSV.")
    if ecr_col is None:
        raise ValueError("Could not find ECR/target column in CSV.")

    video_index = build_video_index(video_dir)
    print(f"Video directory: {video_dir}")
    print(f"Indexed videos: {len(set(video_index.values()))}")
    if video_index:
        print(f"Sample indexed video: {next(iter(video_index.values()))}")

    items = []
    missing_examples = []
    for _, row in df.iterrows():
        video_id = str(row.get(id_col, "")).strip()
        if not video_id or video_id in existing_ids:
            continue
        video_path = video_index.get(video_id) or video_index.get(f"{video_id}.mp4")
        if video_path is None and path_col is not None and pd.notna(row.get(path_col)):
            raw_path = str(row.get(path_col)).strip()
            candidates = [
                raw_path,
                os.path.join(video_dir, raw_path),
                os.path.join(video_dir, os.path.basename(raw_path)),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            if video_path is None:
                video_path = video_index.get(os.path.splitext(os.path.basename(raw_path))[0])
        if video_path is None:
            if len(missing_examples) < 5:
                missing_examples.append(video_id)
            continue
        ecr = safe_float(row.get(ecr_col), None)
        if ecr is None:
            continue
        title = str(row.get(title_col, "")) if title_col and pd.notna(row.get(title_col, "")) else ""
        description = str(row.get(desc_col, "")) if desc_col and pd.notna(row.get(desc_col, "")) else ""
        hashtags = str(row.get(hashtag_col, "")) if hashtag_col and pd.notna(row.get(hashtag_col, "")) else ""
        items.append(VideoItem(video_id, video_path, ecr, title, description, hashtags))
        if len(items) >= max_videos:
            break
    if missing_examples:
        print(f"Missing video examples from CSV ids: {missing_examples}")
    return items


def load_existing(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def save_results(results: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(results, f, ensure_ascii=False)


def extract_final_features(args):
    print("=" * 80)
    print("Final Feature Extraction | SnapUGC-LightKD")
    print(f"Device: {DEVICE} | Caption device: {args.caption_device} | Max videos: {args.max}")
    print("=" * 80)

    existing = load_existing(args.out)
    existing_ids = {str(item.get("video_id")) for item in existing}
    work = build_work(args.csv, args.videos, args.max, existing_ids)
    print(f"Existing: {len(existing)} | To process: {len(work)}")
    if not work:
        if existing:
            save_results(existing, args.out)
            return existing
        raise RuntimeError(
            "No videos matched the CSV. Check --csv, --videos, and printed CSV/video diagnostics."
        )

    reader = VideoReaderHelper()
    clip = CLIPFrameExtractor(args.clip_model)
    motion = None if args.skip_motion else MotionExtractor()
    dover = DoverScoreLookup(args.dover_csv)
    yamnet = None if args.skip_audio else YAMNetExtractor()
    text_encoder = SentenceT5Encoder(args.text_model)
    captioner = BLIPCaptioner(
        args.caption_model,
        device=args.caption_device,
        enabled=not args.skip_caption,
        strict=args.strict_caption,
    )

    results = list(existing)
    errors = 0
    new_count = 0
    consecutive_errors = 0
    t0 = time.time()

    for item in tqdm(work, desc="Extract final features", unit="video"):
        try:
            frames, meta = reader.read_frames(
                item.video_path,
                args.num_frames,
                opening_seconds=args.opening_seconds,
            )
            if not frames:
                raise RuntimeError("empty video")

            clip_frames = clip.encode(frames)
            motion_feats = np.zeros((0, 512), dtype=np.float32)
            if motion is not None:
                clips = reader.read_motion_clips(
                    item.video_path,
                    args.motion_clips,
                    args.motion_frames,
                    opening_seconds=args.opening_seconds,
                )
                motion_feats = motion.encode(clips)

            caption_text, frame_captions = captioner.caption(frames, args.caption_frames)
            metadata_text = " | ".join([p for p in [item.title, item.description, item.hashtags] if p.strip()])

            audio_emb = np.zeros(1024, dtype=np.float32)
            audio_probs = np.zeros(521, dtype=np.float32)
            audio_top = []
            if yamnet is not None:
                audio_emb, audio_probs, audio_top = yamnet.encode(item.video_path)

            quality = dover.score(item.video_id)
            rationale_text = build_rationale(item, caption_text, quality, audio_top)

            metadata_emb = text_encoder.encode(metadata_text)
            caption_emb = text_encoder.encode(caption_text)
            rationale_emb = text_encoder.encode(rationale_text)

            result = {
                "video_id": item.video_id,
                "video_path": item.video_path,
                "duration": meta.get("duration", 0.0),
                "fps": meta.get("fps", 0.0),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "feature_num_frames": meta.get("feature_num_frames", 0),
                "feature_window_seconds": meta.get("feature_window_seconds", 0.0),
                "ecr": item.ecr,
                "title": item.title,
                "description": item.description,
                "hashtags": item.hashtags,
                "clip_frame_embeddings": clip_frames.tolist(),
                "clip_mean_embedding": clip_frames.mean(axis=0).tolist(),
                "motion_clip_embeddings": motion_feats.tolist(),
                "motion_mean_embedding": (motion_feats.mean(axis=0) if len(motion_feats) else np.zeros(512)).tolist(),
                "dover_scores": quality,
                "quality_scores": {
                    "aesthetic": quality["aesthetic"],
                    "technical": quality["technical"],
                    "overall": quality["overall"],
                },
                "yamnet_embedding_mean": audio_emb.tolist(),
                "yamnet_class_probs": audio_probs.tolist(),
                "yamnet_top_classes": audio_top,
                "caption": caption_text,
                "caption_source": "blip" if caption_text else "empty",
                "blip_caption": caption_text,
                "blip_frame_captions": frame_captions,
                "engagement_rationale": rationale_text,
                "metadata_text_embedding": metadata_emb.tolist(),
                "caption_embedding": caption_emb.tolist(),
                "rationale_embedding": rationale_emb.tolist(),
                # Backward-compatible pooled keys for quick legacy checks.
                "visual_emb": clip_frames.mean(axis=0).tolist(),
                "text_emb": metadata_emb.tolist(),
            }
            results.append(result)
            new_count += 1
            consecutive_errors = 0

            if len(results) % args.save_every == 0:
                save_results(results, args.out)
        except Exception as exc:
            errors += 1
            consecutive_errors += 1
            tqdm.write(f"Error {item.video_id}: {exc}")
            if consecutive_errors >= args.max_errors_before_abort:
                raise RuntimeError(
                    f"Aborting after {consecutive_errors} consecutive feature extraction errors. "
                    "Fix the first repeated error before running the full job."
                ) from exc
            if new_count == 0 and errors >= args.max_errors_before_abort:
                raise RuntimeError(
                    f"Aborting after {errors} feature extraction errors and 0 successes. "
                    "Fix the first repeated error before running the full job."
                ) from exc

    save_results(results, args.out)
    elapsed = (time.time() - t0) / 60.0
    print(f"Done. Total: {len(results)} | New errors: {errors} | Time: {elapsed:.1f} min")
    print(f"Output: {args.out}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Final SnapUGC-aligned feature extraction")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-errors-before-abort", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--motion-clips", type=int, default=4)
    parser.add_argument("--motion-frames", type=int, default=16)
    parser.add_argument(
        "--opening-seconds",
        type=float,
        default=None,
        help="Sample visual and motion tokens only from the first N seconds.",
    )
    parser.add_argument("--caption-frames", type=int, default=3)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--text-model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--caption-model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--caption-device", default=DEVICE)
    parser.add_argument("--dover-csv", default=None, help="Precomputed DOVER CSV. Missing means neutral quality scores.")
    parser.add_argument("--strict-caption", action="store_true", help="Abort immediately if BLIP captioning fails.")
    parser.add_argument("--skip-caption", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    args = parser.parse_args()
    extract_final_features(args)


if __name__ == "__main__":
    main()

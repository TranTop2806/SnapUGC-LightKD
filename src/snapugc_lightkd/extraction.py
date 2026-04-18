"""
Final paper-aligned feature extraction for Distil-ShortVU.

Feature groups:
  - CLIP ViT-B/16 frame tokens: visual semantics
  - R(2+1)D-18 Kinetics-400 clip tokens: motion/action
  - DOVER scores from a precomputed CSV: technical/aesthetic/overall video quality
  - YAMNet AudioSet embedding/probabilities: background sound
  - Sentence-T5-base embeddings: metadata, caption, rationale text
  - Qwen2.5-VL-3B-Instruct offline caption/rationale: optional but recommended

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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    def read_frames(self, video_path: str, num_frames: int) -> Tuple[List[Image.Image], Dict]:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return [], {"num_frames": 0, "fps": 0.0, "duration": 0.0, "width": 0, "height": 0}

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in frames]
        fps = float(vr.get_avg_fps() or 0.0)
        h, w = frames[0].shape[:2]
        duration = float(total / fps) if fps > 0 else 0.0
        meta = {
            "num_frames": int(total),
            "fps": fps,
            "duration": duration,
            "width": int(w),
            "height": int(h),
        }
        return pil_frames, meta

    def read_motion_clips(self, video_path: str, num_clips: int, frames_per_clip: int) -> List[np.ndarray]:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return []

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
        feats = self.model.get_image_features(**inputs).detach().cpu().float().numpy()
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
                    "technical": self._value(row, ["technical", "tech"], 0.5),
                    "aesthetic": self._value(row, ["aesthetic", "aes"], 0.5),
                    "overall": self._value(row, ["overall", "quality", "score", "dover"], 0.5),
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
        return self.by_id.get(str(video_id), {"technical": 0.5, "aesthetic": 0.5, "overall": 0.5})


class YAMNetExtractor:
    def __init__(self):
        try:
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


class QwenCaptioner:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device=DEVICE, enabled=True):
        self.enabled = enabled
        self.available = False
        if not enabled:
            print("[Qwen] disabled")
            return
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_id)
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device != "cuda":
                self.model = self.model.to(device)
            self.model.eval()
            self.available = True
            print(f"[Qwen] loaded {model_id}")
        except Exception as exc:
            print(f"[Qwen] unavailable ({exc}); captions/rationales will be empty")

    @torch.no_grad()
    def caption_and_rationale(self, frames: Sequence[Image.Image], title: str, description: str) -> Tuple[str, str, List[str]]:
        if not self.available or not frames:
            return "", "", []
        try:
            from qwen_vl_utils import process_vision_info

            selected = list(frames[:8])
            content = [{"type": "image", "image": img} for img in selected]
            prompt = (
                "You are analyzing a short social-media video from sampled frames and metadata.\n"
                f"Title: {title}\nDescription: {description}\n"
                "Return exactly three sections:\n"
                "CAPTION: concise factual caption.\n"
                "EVENTS: comma-separated important visual events.\n"
                "RATIONALE: short explanation of engagement cues.\n"
                "Do not mention or guess the ground-truth ECR."
            )
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            generated = self.model.generate(**inputs, max_new_tokens=120)
            generated = generated[:, inputs["input_ids"].shape[1]:]
            output = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            return self._parse_output(output)
        except Exception as exc:
            tqdm.write(f"[Qwen] failed: {exc}")
            return "", "", []

    @staticmethod
    def _parse_output(output: str) -> Tuple[str, str, List[str]]:
        caption, rationale, events_text = output, "", ""
        for line in output.splitlines():
            low = line.lower().strip()
            if low.startswith("caption:"):
                caption = line.split(":", 1)[1].strip()
            elif low.startswith("events:"):
                events_text = line.split(":", 1)[1].strip()
            elif low.startswith("rationale:"):
                rationale = line.split(":", 1)[1].strip()
        events = [e.strip() for e in events_text.split(",") if e.strip()]
        return caption, rationale, events


def build_work(csv_path: str, video_dir: str, max_videos: int, existing_ids: set) -> List[VideoItem]:
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        video_id = str(row.get("Id", row.get("id", "")))
        if not video_id or video_id in existing_ids:
            continue
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue
        ecr = safe_float(row.get("ECR"), None)
        if ecr is None:
            continue
        title = str(row.get("Title", "")) if pd.notna(row.get("Title", "")) else ""
        description = str(row.get("Description", "")) if pd.notna(row.get("Description", "")) else ""
        hashtags = str(row.get("Hashtags", "")) if pd.notna(row.get("Hashtags", "")) else ""
        items.append(VideoItem(video_id, video_path, ecr, title, description, hashtags))
        if len(items) >= max_videos:
            break
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
    print("Final Feature Extraction | Distil-ShortVU")
    print(f"Device: {DEVICE} | Max videos: {args.max}")
    print("=" * 80)

    existing = load_existing(args.out)
    existing_ids = {str(item.get("video_id")) for item in existing}
    work = build_work(args.csv, args.videos, args.max, existing_ids)
    print(f"Existing: {len(existing)} | To process: {len(work)}")
    if not work:
        save_results(existing, args.out)
        return existing

    reader = VideoReaderHelper()
    clip = CLIPFrameExtractor(args.clip_model)
    motion = None if args.skip_motion else MotionExtractor()
    dover = DoverScoreLookup(args.dover_csv)
    yamnet = None if args.skip_audio else YAMNetExtractor()
    text_encoder = SentenceT5Encoder(args.text_model)
    qwen = QwenCaptioner(args.qwen_model, enabled=not args.skip_qwen)

    results = list(existing)
    errors = 0
    t0 = time.time()

    for item in tqdm(work, desc="Extract final features", unit="video"):
        try:
            frames, meta = reader.read_frames(item.video_path, args.num_frames)
            if not frames:
                raise RuntimeError("empty video")

            clip_frames = clip.encode(frames)
            motion_feats = np.zeros((0, 512), dtype=np.float32)
            if motion is not None:
                clips = reader.read_motion_clips(item.video_path, args.motion_clips, args.motion_frames)
                motion_feats = motion.encode(clips)

            qwen_caption, qwen_rationale, qwen_events = qwen.caption_and_rationale(
                frames[:args.qwen_frames], item.title, item.description
            )
            metadata_text = " | ".join([p for p in [item.title, item.description, item.hashtags] if p.strip()])
            caption_text = qwen_caption
            rationale_text = qwen_rationale

            metadata_emb = text_encoder.encode(metadata_text)
            caption_emb = text_encoder.encode(caption_text)
            rationale_emb = text_encoder.encode(rationale_text)

            audio_emb = np.zeros(1024, dtype=np.float32)
            audio_probs = np.zeros(521, dtype=np.float32)
            audio_top = []
            if yamnet is not None:
                audio_emb, audio_probs, audio_top = yamnet.encode(item.video_path)

            quality = dover.score(item.video_id)

            result = {
                "video_id": item.video_id,
                "video_path": item.video_path,
                "duration": meta.get("duration", 0.0),
                "fps": meta.get("fps", 0.0),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
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
                "qwen_caption": qwen_caption,
                "qwen_engagement_rationale": qwen_rationale,
                "qwen_visual_events": qwen_events,
                "metadata_text_embedding": metadata_emb.tolist(),
                "caption_embedding": caption_emb.tolist(),
                "rationale_embedding": rationale_emb.tolist(),
                # Backward-compatible pooled keys for quick legacy checks.
                "visual_emb": clip_frames.mean(axis=0).tolist(),
                "text_emb": metadata_emb.tolist(),
            }
            results.append(result)

            if len(results) % args.save_every == 0:
                save_results(results, args.out)
        except Exception as exc:
            errors += 1
            tqdm.write(f"Error {item.video_id}: {exc}")

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
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--motion-clips", type=int, default=4)
    parser.add_argument("--motion-frames", type=int, default=16)
    parser.add_argument("--qwen-frames", type=int, default=8)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--text-model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dover-csv", default=None, help="Precomputed DOVER CSV. Missing means neutral quality scores.")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    args = parser.parse_args()
    extract_final_features(args)


if __name__ == "__main__":
    main()

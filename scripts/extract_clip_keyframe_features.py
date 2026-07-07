#!/usr/bin/env python3
"""Extract CLIP keyframe features from a tar archive of videos (random-access, fast).

Builds a tar index first, then processes videos in parallel-friendly batches.
No full extraction needed — reads each video on demand.

Outputs a .npz with:
  ids            : (N,)     video IDs
  quality_features: (N, T, D) normalized CLIP per-frame embeddings
  quality_mask   : (N, T)   bool, True = valid frame
"""

from __future__ import annotations

import argparse
import io
import tarfile
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────

def uniform_indices(n_total: int, n_sample: int) -> list[int]:
    if n_total <= 0 or n_sample <= 0:
        return []
    if n_total <= n_sample:
        return list(range(n_total))
    step = n_total / n_sample
    return [int(step * i + step / 2) for i in range(n_sample)]


def frames_from_bytes(video_bytes: bytes, n_frames: int) -> list[Image.Image]:
    """Decode video from raw bytes → list of PIL Images."""
    if DECORD_AVAILABLE:
        try:
            vr = decord.VideoReader(io.BytesIO(video_bytes), num_threads=1)
            indices = uniform_indices(len(vr), n_frames)
            if not indices:
                return []
            arr = vr.get_batch(indices).asnumpy()
            return [Image.fromarray(f) for f in arr]
        except Exception:
            pass
    # Fallback: PyAV
    try:
        import av
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]
        total = stream.frames
        indices_set = set(uniform_indices(total, n_frames))
        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i in indices_set:
                frames.append(frame.to_image())
            if len(frames) >= n_frames:
                break
        container.close()
        return frames
    except Exception:
        return []


def frames_from_dir(video_dir: Path, video_id: str, n_frames: int) -> list[Image.Image]:
    """Find video file in directory and decode frames."""
    video_path = None
    for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        p = video_dir / f"{video_id}{ext}"
        if p.exists():
            video_path = p
            break
    if video_path is None:
        return []
    try:
        video_bytes = video_path.read_bytes()
        return frames_from_bytes(video_bytes, n_frames)
    except Exception:
        return []


@torch.no_grad()
def encode_frames(
    images: list[Image.Image],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    if not images:
        return np.zeros((0,), dtype=np.float32)
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        t = torch.stack([preprocess(img) for img in batch]).to(device)
        f = model.encode_image(t)
        f = F.normalize(f, dim=-1)
        all_feats.append(f.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tar", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="ViT-B-32",
                   help="CLIP model: ViT-B-32 (512-d) or ViT-L-14 (768-d)")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def save_npz(results: dict, out_path: str, embed_dim: int, n_frames: int,
             model_name: str, pretrained: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(results.keys())
    feats = np.stack([results[v][0] for v in ids])
    masks = np.stack([results[v][1] for v in ids])
    np.savez_compressed(
        out,
        ids=np.array(ids),
        quality_features=feats,
        quality_mask=masks,
        embed_dim=np.array(embed_dim),
        n_frames=np.array(n_frames),
        model=np.array(model_name),
        pretrained=np.array(pretrained),
    )
    print(f"  Saved {len(ids)} videos → {out} ({out.stat().st_size/1e6:.1f}MB)", flush=True)


def main() -> None:
    args = parse_args()
    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}", flush=True)

    # Target IDs
    df = pd.read_csv(args.labels_csv)
    target_ids = set(df["Id"].astype(str).tolist())
    print(f"Target videos: {len(target_ids)}", flush=True)

    # Resume
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    done_ids: set[str] = set()
    if args.resume and Path(args.out).exists():
        prev = np.load(args.out)
        prev_ids = [str(x) for x in prev["ids"]]
        pf, pm = prev["quality_features"], prev["quality_mask"]
        for i, vid in enumerate(prev_ids):
            results[vid] = (pf[i], pm[i])
        done_ids = set(prev_ids)
        print(f"Resuming from {len(done_ids)} existing", flush=True)

    # Load CLIP model
    print(f"Loading CLIP {args.model} ({args.pretrained}) ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()
    with torch.no_grad():
        _d = torch.zeros(1, 3, 224, 224).to(device)
        embed_dim = int(model.encode_image(_d).shape[-1])
    print(f"Embed dim: {embed_dim}", flush=True)

    # Zero placeholder
    def zero_entry():
        return (
            np.zeros((args.n_frames, embed_dim), dtype=np.float32),
            np.zeros((args.n_frames,), dtype=bool),
        )

    # Determine if input is a directory or a file
    tar_path = Path(args.tar)
    is_dir = tar_path.is_dir()

    if is_dir:
        print(f"Reading videos directly from directory: {tar_path}", flush=True)
        todo = [vid for vid in target_ids if vid not in done_ids]
        print(f"To process: {len(todo)} videos", flush=True)

        n_ok = 0
        n_fail = 0
        t_start = time.time()

        for i, video_id in enumerate(todo):
            try:
                images = frames_from_dir(tar_path, video_id, args.n_frames)
                if not images:
                    raise RuntimeError("no frames")
                feats = encode_frames(images, model, preprocess, device, args.batch_size)
                T = feats.shape[0]
                padded = np.zeros((args.n_frames, embed_dim), dtype=np.float32)
                mask = np.zeros((args.n_frames,), dtype=bool)
                n = min(T, args.n_frames)
                padded[:n] = feats[:n]
                mask[:n] = True
                results[video_id] = (padded, mask)
                n_ok += 1
            except Exception:
                results[video_id] = zero_entry()
                n_fail += 1

            # Progress
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(todo) - i - 1) / rate
                print(
                    f"[{i+1}/{len(todo)}] ok={n_ok} fail={n_fail} "
                    f"rate={rate:.1f}v/s eta={eta/60:.1f}min",
                    flush=True,
                )

            # Checkpoint
            if (i + 1) % args.checkpoint_every == 0:
                save_npz(results, args.out, embed_dim, args.n_frames,
                         args.model, args.pretrained)

        save_npz(results, args.out, embed_dim, args.n_frames, args.model, args.pretrained)
        elapsed = time.time() - t_start
        print(f"\nDone! {n_ok} ok, {n_fail} failed, {elapsed/60:.1f}min total", flush=True)
        return

    # Build tar index
    print(f"Building tar index: {tar_path} ...", flush=True)
    t0 = time.time()
    with tarfile.open(tar_path, "r") as tf:
        all_members = tf.getmembers()

    # Map video_id → tar member (keep only target_ids)
    id_to_member: dict[str, object] = {}
    for m in all_members:
        if not m.isfile():
            continue
        p = Path(m.name)
        if p.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            continue
        stem = p.stem
        if stem.startswith("._"):
            continue
        if stem in target_ids:
            id_to_member[stem] = m
    print(f"Index built in {time.time()-t0:.2f}s: {len(id_to_member)} matching videos", flush=True)

    missing_in_tar = target_ids - set(id_to_member.keys())
    if missing_in_tar:
        print(f"WARNING: {len(missing_in_tar)} target IDs not found in tar", flush=True)

    # Process
    todo = [vid for vid in id_to_member if vid not in done_ids]
    print(f"To process: {len(todo)} videos", flush=True)

    n_ok = 0
    n_fail = 0
    t_start = time.time()

    with tarfile.open(tar_path, "r") as tf:
        for i, video_id in enumerate(todo):
            member = id_to_member[video_id]
            try:
                f = tf.extractfile(member)
                video_bytes = f.read()
                images = frames_from_bytes(video_bytes, args.n_frames)
                if not images:
                    raise RuntimeError("no frames")
                feats = encode_frames(images, model, preprocess, device, args.batch_size)
                T = feats.shape[0]
                padded = np.zeros((args.n_frames, embed_dim), dtype=np.float32)
                mask = np.zeros((args.n_frames,), dtype=bool)
                n = min(T, args.n_frames)
                padded[:n] = feats[:n]
                mask[:n] = True
                results[video_id] = (padded, mask)
                n_ok += 1
            except Exception:
                results[video_id] = zero_entry()
                n_fail += 1

            # Progress
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(todo) - i - 1) / rate
                print(
                    f"[{i+1}/{len(todo)}] ok={n_ok} fail={n_fail} "
                    f"rate={rate:.1f}v/s eta={eta/60:.1f}min",
                    flush=True,
                )

            # Checkpoint
            if (i + 1) % args.checkpoint_every == 0:
                save_npz(results, args.out, embed_dim, args.n_frames,
                         args.model, args.pretrained)

    # Add zeros for any videos not in tar
    for vid in missing_in_tar:
        results[vid] = zero_entry()

    save_npz(results, args.out, embed_dim, args.n_frames, args.model, args.pretrained)
    elapsed = time.time() - t_start
    print(f"\nDone! {n_ok} ok, {n_fail} failed, {elapsed/60:.1f}min total", flush=True)


if __name__ == "__main__":
    main()

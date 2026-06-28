#!/usr/bin/env python3
"""
Extract Lightweight Action Features using MobileNetV3-Small.
Mô phỏng thông tin chuyển động (motion) cực nhẹ cho thiết bị di động.
"""

import argparse
import io
import time
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

try:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    HAS_WEIGHTS_API = True
except ImportError:
    from torchvision.models import mobilenet_v3_small
    HAS_WEIGHTS_API = False

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


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
def main():
    parser = argparse.ArgumentParser(description="Extract Lightweight Action Features.")
    parser.add_argument("--tar", required=True, help="Path to tar archive or directory of videos")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}", flush=True)

    print("Loading MobileNetV3-Small...", flush=True)
    if HAS_WEIGHTS_API:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        preprocess = weights.transforms()
        model = mobilenet_v3_small(weights=weights).features.to(device)
    else:
        # Fallback for older torchvision versions
        preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = mobilenet_v3_small(pretrained=True).features.to(device)
    model.eval()

    df = pd.read_csv(args.labels_csv)
    target_ids = set(df["Id"].astype(str).tolist())
    print(f"Target videos: {len(target_ids)}", flush=True)

    results = {}
    done_ids = set()
    if args.resume and Path(args.out).exists():
        try:
            prev = np.load(args.out)
            prev_ids = [str(x) for x in prev["ids"]]
            pf = prev["lite_action_features"]
            for i, vid in enumerate(prev_ids):
                results[vid] = pf[i]
            done_ids = set(prev_ids)
            print(f"Resumed {len(done_ids)} existing features", flush=True)
        except Exception as e:
            print(f"Error resuming: {e}. Starting fresh.", flush=True)

    todo_ids = sorted(list(target_ids - done_ids))
    print(f"To process: {len(todo_ids)} videos", flush=True)
    if not todo_ids:
        print("Nothing to process!", flush=True)
        if done_ids:
            # Re-save to finalize/sort
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ids = sorted(results.keys())
            feats = np.stack([results[v] for v in ids])
            np.savez_compressed(args.out, ids=np.array(ids), lite_action_features=feats)
            print(f"Finalized and sorted {len(ids)} features.", flush=True)
        return

    n_ok = 0
    n_fail = 0
    t_start = time.time()

    # Determine if tar is a directory or a file
    tar_path = Path(args.tar)
    is_dir = tar_path.is_dir()

    if is_dir:
        print(f"Reading videos directly from directory: {tar_path}", flush=True)
        for i, video_id in enumerate(todo_ids):
            try:
                images = frames_from_dir(tar_path, video_id, args.n_frames)
                if not images:
                    raise RuntimeError("No frames decoded")

                # Preprocess & Extract
                t_imgs = torch.stack([preprocess(img) for img in images]).to(device)
                features = model(t_imgs)  # shape: (T, 576, 7, 7)
                features = features.mean(dim=[2, 3])  # Global Average Pooling -> (T, 576)

                # Compute Temporal Difference (Motion)
                motion = torch.zeros_like(features)
                motion[1:] = features[1:] - features[:-1]

                # Concat Spatial + Motion = (T, 1152)
                spatiotemporal = torch.cat([features, motion], dim=-1).cpu().numpy()

                padded = np.zeros((args.n_frames, 1152), dtype=np.float32)
                n = min(len(images), args.n_frames)
                padded[:n] = spatiotemporal[:n]

                results[video_id] = padded
                n_ok += 1
            except Exception as e:
                results[video_id] = np.zeros((args.n_frames, 1152), dtype=np.float32)
                n_fail += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(todo_ids) - i - 1) / rate
                print(
                    f"[{i+1}/{len(todo_ids)}] ok={n_ok} fail={n_fail} "
                    f"rate={rate:.1f}v/s ETA={eta/60:.1f}m",
                    flush=True,
                )
    else:
        print(f"Reading videos from tar archive: {tar_path}", flush=True)
        # Build tar index first
        print("Building tar index...", flush=True)
        with tarfile.open(tar_path, "r") as tf:
            all_members = tf.getmembers()

        id_to_member = {}
        for m in all_members:
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                continue
            if p.stem.startswith("._"):
                continue
            if p.stem in target_ids:
                id_to_member[p.stem] = m
        print(f"Index built: found {len(id_to_member)} matching videos in tar", flush=True)

        with tarfile.open(tar_path, "r") as tf:
            for i, video_id in enumerate(todo_ids):
                member = id_to_member.get(video_id)
                try:
                    if member is None:
                        raise FileNotFoundError("not in tar")
                    video_bytes = tf.extractfile(member).read()
                    images = frames_from_bytes(video_bytes, args.n_frames)
                    if not images:
                        raise RuntimeError("No frames decoded")

                    # Preprocess & Extract
                    t_imgs = torch.stack([preprocess(img) for img in images]).to(device)
                    features = model(t_imgs)  # shape: (T, 576, 7, 7)
                    features = features.mean(dim=[2, 3])  # Global Average Pooling -> (T, 576)

                    # Compute Temporal Difference (Motion)
                    motion = torch.zeros_like(features)
                    motion[1:] = features[1:] - features[:-1]

                    # Concat Spatial + Motion = (T, 1152)
                    spatiotemporal = torch.cat([features, motion], dim=-1).cpu().numpy()

                    padded = np.zeros((args.n_frames, 1152), dtype=np.float32)
                    n = min(len(images), args.n_frames)
                    padded[:n] = spatiotemporal[:n]

                    results[video_id] = padded
                    n_ok += 1
                except Exception as e:
                    results[video_id] = np.zeros((args.n_frames, 1152), dtype=np.float32)
                    n_fail += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(todo_ids) - i - 1) / rate
                    print(
                        f"[{i+1}/{len(todo_ids)}] ok={n_ok} fail={n_fail} "
                        f"rate={rate:.1f}v/s ETA={eta/60:.1f}m",
                        flush=True,
                    )

    # Save NPZ file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(results.keys())
    feats = np.stack([results[v] for v in ids])
    np.savez_compressed(args.out, ids=np.array(ids), lite_action_features=feats)
    elapsed = time.time() - t_start
    print(f"Done! {n_ok} ok, {n_fail} failed, {elapsed/60:.1f}m total", flush=True)


if __name__ == "__main__":
    main()

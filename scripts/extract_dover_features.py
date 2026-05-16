#!/usr/bin/env python3
"""Extract DOVER technical/aesthetic scores + pooled features for SnapUGC videos.

Uses PyAV instead of decord for cross-platform compatibility (macOS, Kaggle, etc.)
Outputs: dover_features.npz with keys:
  ids, technical_score, aesthetic_score, technical_feature, aesthetic_feature
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DOVER_ROOT = ROOT / "third_party" / "DOVER"
sys.path.insert(0, str(DOVER_ROOT))

from dover.models import DOVER


def load_video_av(video_path, max_frames=None, load_size=None):
    """Load video with PyAV: returns (C, T, H, W) float tensor [0, 255]."""
    import av
    container = av.open(str(video_path))
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if max_frames and i >= max_frames:
            break
        arr = frame.to_ndarray(format='rgb24')  # H, W, C
        frames.append(torch.from_numpy(arr))
    container.close()
    if len(frames) == 0:
        raise RuntimeError(f"No frames: {video_path}")
    video = torch.stack(frames)  # T, H, W, C
    video = video.permute(3, 0, 1, 2).float()  # C, T, H, W
    return video


def get_spatial_fragments(video, fragments_h=7, fragments_w=7,
                          fsize_h=32, fsize_w=32, aligned=32,
                          random=False):
    """Spatial fragment sampling for DOVER technical view."""
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if ratio < 1:
        ovideo = video
        video = F.interpolate(video / 255.0, scale_factor=1 / ratio, mode='bilinear')
        video = (video * 255.0).type_as(ovideo)
    assert dur_t % aligned == 0, f"dur_t={dur_t} not divisible by aligned={aligned}"
    hgrids = torch.LongTensor([min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)])
    wgrids = torch.LongTensor([min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)])
    hlength = res_h // fragments_h
    wlength = res_w // fragments_w
    if hlength > fsize_h:
        rnd_h = torch.randint(hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned))
    else:
        rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    if wlength > fsize_w:
        rnd_w = torch.randint(wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned))
    else:
        rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    target_video = torch.zeros(video.shape[:-2] + (size_h, size_w)).to(video.device)
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[:, t_s:t_e, h_so:h_eo, w_so:w_eo]
    return target_video


def temporal_sampling(total_frames, clip_len, num_clips, frame_interval):
    """Uniform temporal sampling."""
    all_inds = []
    for clip_idx in range(num_clips):
        start = int((total_frames - clip_len * frame_interval) * clip_idx / max(num_clips - 1, 1))
        inds = np.arange(clip_len) * frame_interval + start
        all_inds.append(inds)
    return np.concatenate(all_inds).astype(np.int32)


def preprocess_dover_views(video_tensor, device='cpu'):
    """Prepare technical and aesthetic views from loaded video tensor.
    video_tensor: (C, T, H, W) float [0, 255]
    Returns dict with 'technical' and 'aesthetic' tensors ready for DOVER.
    """
    C, T, H, W = video_tensor.shape
    mean = torch.FloatTensor([123.675, 116.28, 103.53]).to(device)
    std = torch.FloatTensor([58.395, 57.12, 57.375]).to(device)

    # Aesthetic view: resize 224x224, sample 32 frames
    aest_frames = temporal_sampling(T, 32, 1, 2)
    aest_frames = np.clip(aest_frames, 0, T - 1)
    aest_video = video_tensor[:, aest_frames, :, :].to(device)
    aest_video = F.interpolate(aest_video / 255.0, size=(224, 224), mode='bilinear')
    aest_video = ((aest_video * 255.0) - mean.view(-1, 1, 1, 1)) / std.view(-1, 1, 1, 1)
    # Shape: (C, 32, 224, 224) → add batch: (1, C, 32, 224, 224)
    aest_video = aest_video.unsqueeze(0)

    # Technical view: spatial fragments 7x7 × 32x32, 32 frames, 3 clips
    tech_frames = temporal_sampling(T, 32, 3, 2)
    tech_frames = np.clip(tech_frames, 0, T - 1)
    tech_video = video_tensor[:, tech_frames, :, :].to(device)
    # Align to 32
    if tech_video.shape[1] % 32 != 0:
        pad = 32 - (tech_video.shape[1] % 32)
        tech_video = F.pad(tech_video, (0, 0, 0, 0, 0, pad), mode='replicate')
    tech_video = get_spatial_fragments(tech_video, fragments_h=7, fragments_w=7,
                                       fsize_h=32, fsize_w=32, aligned=32)
    tech_video = ((tech_video / 255.0) - mean.view(-1, 1, 1, 1)) / std.view(-1, 1, 1, 1)
    tech_video = tech_video.unsqueeze(0)

    return {'aesthetic': aest_video, 'technical': tech_video}


def extract_dover_features(model, video_path, device='cpu'):
    """Extract scores and pooled features."""
    video = load_video_av(video_path, load_size=320)
    views = preprocess_dover_views(video, device=device)

    with torch.no_grad():
        scores, feats = model(views, inference=True, return_pooled_feats=True)
    # scores: list of [technical_score, aesthetic_score]
    # feats: dict with 'technical' and 'aesthetic' pooled features
    tech_score = scores[0].mean().item()
    aest_score = scores[1].mean().item()
    tech_feat = feats['technical'].mean((-3, -2, -1)).cpu().numpy()
    aest_feat = feats['aesthetic'].mean((-3, -2, -1)).cpu().numpy()
    return {
        'technical_score': tech_score,
        'aesthetic_score': aest_score,
        'technical_feature': tech_feat,
        'aesthetic_feature': aest_feat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', required=True)
    parser.add_argument('--csv', required=True, help='CSV with Id column')
    parser.add_argument('--out', default='dover_features.npz')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--ckpt', default=str(DOVER_ROOT / 'pretrained_weights' / 'DOVER.pth'))
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model_args = {
        'backbone': {
            'technical': {'type': 'swin_tiny_grpb', 'checkpoint': True},
            'aesthetic': {'type': 'conv_tiny'}
        },
        'backbone_preserve_keys': 'technical,aesthetic',
        'divide_head': True,
        'vqa_head': {'in_channels': 768, 'hidden_channels': 64}
    }
    dover = DOVER(**model_args).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    dover.load_state_dict(state, strict=True)
    print(f'Loaded DOVER from {args.ckpt}')

    import pandas as pd
    df = pd.read_csv(args.csv)
    ids = df['Id'].astype(str).tolist()[:args.max_videos]
    print(f'Videos to process: {len(ids)}')

    results = {
        'ids': [],
        'technical_score': [],
        'aesthetic_score': [],
        'technical_feature': [],
        'aesthetic_feature': [],
    }

    for i, vid in enumerate(ids):
        vpath = Path(args.video_dir) / f'{vid}.mp4'
        if not vpath.exists():
            vpath = Path(args.video_dir) / vid
        try:
            feat = extract_dover_features(dover, vpath, device=device)
            results['ids'].append(vid)
            results['technical_score'].append(feat['technical_score'])
            results['aesthetic_score'].append(feat['aesthetic_score'])
            results['technical_feature'].append(feat['technical_feature'])
            results['aesthetic_feature'].append(feat['aesthetic_feature'])
            if device.type == 'mps':
                torch.mps.empty_cache()
            if (i + 1) % 10 == 0:
                print(f'  {i+1}/{len(ids)} done')
        except Exception as e:
            print(f'ERROR {vid}: {e}')
            results['ids'].append(vid)
            results['technical_score'].append(0.0)
            results['aesthetic_score'].append(0.0)
            results['technical_feature'].append(np.zeros(768, dtype=np.float32))
            results['aesthetic_feature'].append(np.zeros(768, dtype=np.float32))
        if (i + 1) % args.save_every == 0:
            np.savez(args.out,
                     ids=np.array(results['ids']),
                     technical_score=np.array(results['technical_score'], dtype=np.float32),
                     aesthetic_score=np.array(results['aesthetic_score'], dtype=np.float32),
                     technical_feature=np.stack(results['technical_feature']).astype(np.float32),
                     aesthetic_feature=np.stack(results['aesthetic_feature']).astype(np.float32))
            print(f'  Checkpoint saved at {i+1}/{len(ids)}')

    np.savez(args.out,
             ids=np.array(results['ids']),
             technical_score=np.array(results['technical_score'], dtype=np.float32),
             aesthetic_score=np.array(results['aesthetic_score'], dtype=np.float32),
             technical_feature=np.stack(results['technical_feature']).astype(np.float32),
             aesthetic_feature=np.stack(results['aesthetic_feature']).astype(np.float32))
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()

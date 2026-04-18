"""Train the KD v2 final paper-aligned temporal pipeline.

Pipeline:
  1. Load features from source/kd_v2/extract_features.py
  2. Train privileged multimodal teacher
  3. Generate teacher soft targets + temporal attention targets
  4. Train lightweight student baseline
  5. Train lightweight student with KD
  6. Report PLCC, SRCC, KRCC, MSE, MAE, and VQualA-style final score
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .models import TeacherModel, StudentModel, count_params


class KDDataset(Dataset):
    """Dataset for the final paper-aligned temporal KD schema."""

    def __init__(self, data_list, max_frames=16, max_motion_clips=4,
                 clip_dim=512, motion_dim=512, audio_dim=1024, text_dim=768):
        self.data = data_list
        self.max_frames = max_frames
        self.max_motion_clips = max_motion_clips
        self.clip_dim = clip_dim
        self.motion_dim = motion_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _quality_value(item, names, default=0.5):
        quality = item.get('dover_scores') or item.get('quality_scores') or {}
        for name in names:
            if name in quality:
                value = float(quality[name])
                return value / 10.0 if value > 1.0 else value
        return default

    @staticmethod
    def _as_vector(value, dim):
        if value is None:
            return torch.zeros(dim, dtype=torch.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.size < dim:
            arr = np.pad(arr, (0, dim - arr.size))
        if arr.size > dim:
            arr = arr[:dim]
        return torch.tensor(arr, dtype=torch.float32)

    @staticmethod
    def _as_sequence(value, max_len, dim):
        if value is None:
            arr = np.zeros((0, dim), dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < dim:
                arr = np.pad(arr, ((0, 0), (0, dim - arr.shape[1])))
            if arr.shape[1] > dim:
                arr = arr[:, :dim]

        length = min(len(arr), max_len)
        out = np.zeros((max_len, dim), dtype=np.float32)
        mask = np.zeros(max_len, dtype=bool)
        if length > 0:
            out[:length] = arr[:length]
            mask[:length] = True
        return torch.tensor(out, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, idx):
        item = self.data[idx]

        clip_value = item.get('clip_frame_embeddings') or item.get('clip_frames') or item.get('visual_emb')
        motion_value = item.get('motion_clip_embeddings') or item.get('motion_mean_embedding')
        audio_value = item.get('yamnet_embedding_mean') or item.get('audio_emb')
        text_value = (
            item.get('metadata_text_embedding')
            or item.get('text_emb')
            or item.get('title_embedding')
        )
        caption_value = item.get('caption_embedding')
        rationale_value = item.get('rationale_embedding')

        clip_frames, clip_mask = self._as_sequence(clip_value, self.max_frames, self.clip_dim)
        motion_clips, motion_mask = self._as_sequence(motion_value, self.max_motion_clips, self.motion_dim)

        aesthetic = self._quality_value(item, ['aesthetic', 'aesthetic_score'], 0.5)
        technical = self._quality_value(item, ['technical', 'technical_score'], 0.5)
        overall = self._quality_value(item, ['overall', 'quality', 'dover'], (aesthetic + technical) / 2)

        result = {
            'clip_frames': clip_frames,
            'clip_mask': clip_mask,
            'motion_clips': motion_clips,
            'motion_mask': motion_mask,
            'audio_emb': self._as_vector(audio_value, self.audio_dim),
            'text_emb': self._as_vector(text_value, self.text_dim),
            'caption_emb': self._as_vector(caption_value, self.text_dim),
            'rationale_emb': self._as_vector(rationale_value, self.text_dim),
            'quality_scores': torch.tensor([technical, aesthetic, overall], dtype=torch.float32),
            'ecr': torch.tensor(item.get('ecr', 0.0) or 0.0, dtype=torch.float32),
            'aesthetic': torch.tensor(aesthetic, dtype=torch.float32),
            'technical': torch.tensor(technical, dtype=torch.float32),
        }

        if 'teacher_ecr' in item:
            result['teacher_ecr'] = torch.tensor(item['teacher_ecr'], dtype=torch.float32)
        if 'teacher_hidden' in item:
            result['teacher_hidden'] = torch.tensor(item['teacher_hidden'], dtype=torch.float32)
        if 'teacher_temporal_attention' in item:
            result['teacher_temporal_attention'] = torch.tensor(
                item['teacher_temporal_attention'], dtype=torch.float32)

        return result


def load_data(json_path, max_samples=None):
    """Load and validate extracted features."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        if json_path.endswith('.jsonl'):
            raw = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)

    data = list(raw.values()) if isinstance(raw, dict) else raw

    valid = []
    for item in data:
        emb = item.get('visual_emb') or item.get('imagebind_emb') or item.get('clip_frame_embeddings')
        if emb is not None and item.get('ecr') is not None:
            valid.append(item)

    if max_samples and len(valid) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(valid), max_samples, replace=False)
        valid = [valid[i] for i in sorted(indices)]

    ecrs = [d['ecr'] for d in valid]
    print(f"  Loaded {len(valid)} samples with ECR")
    print(f"  ECR: mean={np.mean(ecrs):.4f}, std={np.std(ecrs):.4f}, "
          f"min={np.min(ecrs):.4f}, max={np.max(ecrs):.4f}")

    has_text = sum(
        1 for d in valid
        if d.get('text_emb') is not None or d.get('metadata_text_embedding') is not None
    )
    print(f"  With text embeddings: {has_text}/{len(valid)}")

    return valid


def split_data(data, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(data)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return [data[i] for i in train_idx], [data[i] for i in val_idx]


def is_better_metric(metrics, best_value, metric_name):
    """Select checkpoints using the metric that matches the evaluation target."""
    value = float(metrics[metric_name])
    if metric_name in {"mse", "mae"}:
        return value < best_value, value
    return value > best_value, value


def train_teacher_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        kwargs = {
            'clip_frames': batch['clip_frames'].to(device),
            'clip_mask': batch['clip_mask'].to(device),
            'motion_clips': batch['motion_clips'].to(device),
            'motion_mask': batch['motion_mask'].to(device),
            'audio_emb': batch['audio_emb'].to(device),
            'text_emb': batch['text_emb'].to(device),
            'caption_emb': batch['caption_emb'].to(device),
            'rationale_emb': batch['rationale_emb'].to(device),
            'quality_scores': batch['quality_scores'].to(device),
            'ecr_targets': batch['ecr'].to(device),
        }
        optimizer.zero_grad()
        out = model(**kwargs)
        out['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += out['loss'].item()
        n += 1
    return total_loss / max(1, n)


def train_student_epoch(model, loader, optimizer, device, use_kd=False, loss_weights=None):
    model.train()
    total_loss = 0.0
    loss_accum = {}
    n = 0
    for batch in loader:
        kwargs = {
            'clip_frames': batch['clip_frames'].to(device),
            'clip_mask': batch['clip_mask'].to(device),
            'audio_emb': batch['audio_emb'].to(device),
            'text_emb': batch['text_emb'].to(device),
            'ecr_targets': batch['ecr'].to(device),
            'loss_weights': loss_weights,
        }
        if use_kd:
            kwargs['aesthetic_targets'] = batch['aesthetic'].to(device)
            kwargs['technical_targets'] = batch['technical'].to(device)
            if 'teacher_ecr' in batch:
                kwargs['teacher_ecr'] = batch['teacher_ecr'].to(device)
            if 'teacher_hidden' in batch:
                kwargs['teacher_hidden'] = batch['teacher_hidden'].to(device)
            if 'teacher_temporal_attention' in batch:
                kwargs['teacher_temporal_attention'] = batch['teacher_temporal_attention'].to(device)

        optimizer.zero_grad()
        out = model(**kwargs)
        out['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += out['loss'].item()
        for k, v in out['losses'].items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
        n += 1

    metrics = {'loss': total_loss / max(1, n)}
    for k, v in loss_accum.items():
        metrics[k] = v / max(1, n)
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, device, model_type='teacher'):
    model.eval()
    all_pred, all_true = [], []
    for batch in loader:
        if model_type == 'teacher':
            out = model(
                clip_frames=batch['clip_frames'].to(device),
                clip_mask=batch['clip_mask'].to(device),
                motion_clips=batch['motion_clips'].to(device),
                motion_mask=batch['motion_mask'].to(device),
                audio_emb=batch['audio_emb'].to(device),
                text_emb=batch['text_emb'].to(device),
                caption_emb=batch['caption_emb'].to(device),
                rationale_emb=batch['rationale_emb'].to(device),
                quality_scores=batch['quality_scores'].to(device),
            )
        else:
            out = model(
                clip_frames=batch['clip_frames'].to(device),
                clip_mask=batch['clip_mask'].to(device),
                audio_emb=batch['audio_emb'].to(device),
                text_emb=batch['text_emb'].to(device),
            )
        all_pred.extend(out['predicted_ecr'].cpu().numpy())
        all_true.extend(batch['ecr'].numpy())

    pred = np.array(all_pred)
    true = np.array(all_true)
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    plcc = pearsonr(pred, true)[0] if len(pred) > 2 and pred.std() > 0 and true.std() > 0 else 0
    srcc = spearmanr(pred, true).correlation if len(pred) > 2 else 0
    ktau = kendalltau(pred, true).correlation if len(pred) > 2 else 0
    plcc = 0 if np.isnan(plcc) else plcc
    srcc = 0 if np.isnan(srcc) else srcc
    ktau = 0 if np.isnan(ktau) else ktau
    return {
        'mse': mse, 'mae': mae,
        'plcc': plcc, 'srcc': srcc, 'ktau': ktau,
        'final_score': 0.6 * srcc + 0.4 * plcc,
        'pred_mean': pred.mean(), 'pred_std': pred.std(),
    }


@torch.no_grad()
def generate_teacher_targets(teacher, data_list, device, batch_size=64,
                                   max_frames=16, max_motion_clips=4):
    teacher.eval()
    dataset = KDDataset(data_list, max_frames=max_frames, max_motion_clips=max_motion_clips)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_ecr, all_hidden, all_attn = [], [], []
    for batch in loader:
        out = teacher(
            clip_frames=batch['clip_frames'].to(device),
            clip_mask=batch['clip_mask'].to(device),
            motion_clips=batch['motion_clips'].to(device),
            motion_mask=batch['motion_mask'].to(device),
            audio_emb=batch['audio_emb'].to(device),
            text_emb=batch['text_emb'].to(device),
            caption_emb=batch['caption_emb'].to(device),
            rationale_emb=batch['rationale_emb'].to(device),
            quality_scores=batch['quality_scores'].to(device),
        )
        all_ecr.append(out['predicted_ecr'].cpu().numpy())
        all_hidden.append(out['hidden'].cpu().numpy())
        all_attn.append(out['temporal_attention'].cpu().numpy())

    ecrs = np.concatenate(all_ecr)
    hiddens = np.concatenate(all_hidden)
    attns = np.concatenate(all_attn)
    for i, item in enumerate(data_list):
        item['teacher_ecr'] = float(ecrs[i])
        item['teacher_hidden'] = hiddens[i].tolist()
        item['teacher_temporal_attention'] = attns[i].tolist()

    print(f"  Generated final teacher targets for {len(data_list)} samples")
    print(f"  Teacher ECR: mean={ecrs.mean():.4f}, std={ecrs.std():.4f}")
    return data_list
def run_experiment(args):
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print("  FINAL PAPER-ALIGNED TEMPORAL KD EXPERIMENT")
    print(f"  Device: {device} | Data: {args.data}")
    print(f"{'='*70}\n")

    data = load_data(args.data, max_samples=args.max_samples)
    train_data, val_data = split_data(data, val_ratio=0.2)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}\n")

    train_loader = DataLoader(
        KDDataset(train_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch, shuffle=True,
    )
    val_loader = DataLoader(
        KDDataset(val_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch, shuffle=False,
    )

    # ==========================================================
    # PHASE 1: Teacher
    # ==========================================================
    print(f"{'='*70}")
    print("  PHASE 1: Training Final Teacher")
    print(f"{'='*70}")

    teacher = TeacherModel(
        hidden_dim=args.teacher_hidden,
        n_blocks=args.teacher_blocks,
        dropout=args.dropout,
        max_frames=args.max_frames,
        max_motion_clips=args.max_motion_clips,
    ).to(device)
    t_total, t_train = count_params(teacher)
    print(f"  Teacher params: {t_total:,} ({t_train:,} trainable)")

    optimizer = AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.teacher_epochs, eta_min=1e-6)
    best_metric = float('inf') if args.selection_metric in {"mse", "mae"} else -float('inf')
    t_start = time.time()

    for epoch in range(1, args.teacher_epochs + 1):
        train_loss = train_teacher_epoch(teacher, train_loader, optimizer, device)
        scheduler.step()
        if epoch % max(1, args.teacher_epochs // 10) == 0 or epoch == args.teacher_epochs:
            val_metrics = evaluate_model(teacher, val_loader, device, 'teacher')
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.5f} | "
                  f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
                  f"Score={val_metrics['final_score']:.4f} MSE={val_metrics['mse']:.5f}")
            better, metric_value = is_better_metric(
                val_metrics, best_metric, args.selection_metric
            )
            if better:
                best_metric = metric_value
                torch.save(teacher.state_dict(), os.path.join(args.save_dir, 'final_teacher_best.pth'))

    teacher_time = time.time() - t_start
    teacher.load_state_dict(torch.load(os.path.join(args.save_dir, 'final_teacher_best.pth'),
                                       map_location=device, weights_only=True))
    teacher_val = evaluate_model(teacher, val_loader, device, 'teacher')
    print(f"\n  Teacher BEST: PLCC={teacher_val['plcc']:.4f} SRCC={teacher_val['srcc']:.4f} "
          f"Score={teacher_val['final_score']:.4f} MSE={teacher_val['mse']:.5f} "
          f"MAE={teacher_val['mae']:.4f} ({teacher_time:.0f}s)")

    # ==========================================================
    # PHASE 2: Teacher targets
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PHASE 2: Generating Final Teacher Soft Targets")
    print(f"{'='*70}")
    train_data = generate_teacher_targets(
        teacher, train_data, device, args.batch,
        max_frames=args.max_frames, max_motion_clips=args.max_motion_clips,
    )
    val_data = generate_teacher_targets(
        teacher, val_data, device, args.batch,
        max_frames=args.max_frames, max_motion_clips=args.max_motion_clips,
    )
    train_loader = DataLoader(
        KDDataset(train_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch, shuffle=True,
    )
    val_loader = DataLoader(
        KDDataset(val_data, max_frames=args.max_frames, max_motion_clips=args.max_motion_clips),
        batch_size=args.batch, shuffle=False,
    )

    # ==========================================================
    # PHASE 3: Student baseline
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PHASE 3: Training Final Student Baseline")
    print(f"{'='*70}")

    baseline = StudentModel(
        hidden_dim=args.student_hidden,
        teacher_hidden_dim=args.teacher_hidden,
        max_frames=args.max_frames,
        dropout=args.dropout,
    ).to(device)
    s_total, s_train = count_params(baseline)
    print(f"  Student params: {s_total:,} ({s_train:,} trainable)")
    print(f"  Compression ratio: {t_total/s_total:.1f}x smaller than teacher")

    baseline_weights = {
        'ecr_hard': 1.0, 'ecr_soft': 0.0, 'kd_repr': 0.0,
        'temporal_attn': 0.0, 'aesthetic': 0.0, 'technical': 0.0,
    }
    optimizer = AdamW(baseline.parameters(), lr=args.student_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.student_epochs, eta_min=1e-6)
    best_metric = float('inf') if args.selection_metric in {"mse", "mae"} else -float('inf')
    t_start = time.time()

    for epoch in range(1, args.student_epochs + 1):
        train_metrics = train_student_epoch(
            baseline, train_loader, optimizer, device,
            use_kd=False, loss_weights=baseline_weights,
        )
        scheduler.step()
        if epoch % max(1, args.student_epochs // 10) == 0 or epoch == args.student_epochs:
            val_metrics = evaluate_model(baseline, val_loader, device, 'student')
            print(f"  Epoch {epoch:3d}: loss={train_metrics['loss']:.5f} | "
                  f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
                  f"Score={val_metrics['final_score']:.4f} MSE={val_metrics['mse']:.5f}")
            better, metric_value = is_better_metric(
                val_metrics, best_metric, args.selection_metric
            )
            if better:
                best_metric = metric_value
                torch.save(baseline.state_dict(), os.path.join(args.save_dir, 'final_student_baseline_best.pth'))

    baseline_time = time.time() - t_start
    baseline.load_state_dict(torch.load(os.path.join(args.save_dir, 'final_student_baseline_best.pth'),
                                        map_location=device, weights_only=True))
    baseline_val = evaluate_model(baseline, val_loader, device, 'student')
    print(f"\n  Baseline BEST: PLCC={baseline_val['plcc']:.4f} SRCC={baseline_val['srcc']:.4f} "
          f"Score={baseline_val['final_score']:.4f} MSE={baseline_val['mse']:.5f} "
          f"MAE={baseline_val['mae']:.4f} ({baseline_time:.0f}s)")

    # ==========================================================
    # PHASE 4: Student KD
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PHASE 4: Training Final Student + KD")
    print(f"{'='*70}")

    student_kd = StudentModel(
        hidden_dim=args.student_hidden,
        teacher_hidden_dim=args.teacher_hidden,
        max_frames=args.max_frames,
        dropout=args.dropout,
    ).to(device)
    kd_weights = {
        'ecr_hard': 1.0,
        'ecr_soft': args.alpha,
        'kd_repr': args.beta,
        'temporal_attn': args.attn_kd,
        'aesthetic': args.gamma,
        'technical': args.delta,
    }
    print(f"  KD weights: soft={args.alpha}, repr={args.beta}, "
          f"attn={args.attn_kd}, aesthetic={args.gamma}, technical={args.delta}")

    optimizer = AdamW(student_kd.parameters(), lr=args.student_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.student_epochs, eta_min=1e-6)
    best_metric = float('inf') if args.selection_metric in {"mse", "mae"} else -float('inf')
    t_start = time.time()

    for epoch in range(1, args.student_epochs + 1):
        train_metrics = train_student_epoch(
            student_kd, train_loader, optimizer, device,
            use_kd=True, loss_weights=kd_weights,
        )
        scheduler.step()
        if epoch % max(1, args.student_epochs // 10) == 0 or epoch == args.student_epochs:
            val_metrics = evaluate_model(student_kd, val_loader, device, 'student')
            kd_losses = {k: f"{v:.4f}" for k, v in train_metrics.items() if k != 'loss'}
            print(f"  Epoch {epoch:3d}: loss={train_metrics['loss']:.5f} {kd_losses} | "
                  f"PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
                  f"Score={val_metrics['final_score']:.4f} MSE={val_metrics['mse']:.5f}")
            better, metric_value = is_better_metric(
                val_metrics, best_metric, args.selection_metric
            )
            if better:
                best_metric = metric_value
                torch.save(student_kd.state_dict(), os.path.join(args.save_dir, 'final_student_kd_best.pth'))

    kd_time = time.time() - t_start
    student_kd.load_state_dict(torch.load(os.path.join(args.save_dir, 'final_student_kd_best.pth'),
                                          map_location=device, weights_only=True))
    kd_val = evaluate_model(student_kd, val_loader, device, 'student')
    print(f"\n  KD BEST: PLCC={kd_val['plcc']:.4f} SRCC={kd_val['srcc']:.4f} "
          f"Score={kd_val['final_score']:.4f} MSE={kd_val['mse']:.5f} "
          f"MAE={kd_val['mae']:.4f} ({kd_time:.0f}s)")

    print(f"\n{'='*70}")
    print("  FINAL COMPARISON REPORT")
    print(f"{'='*70}")
    print(f"\n  {'Model':<25} {'Params':>10} {'PLCC':>8} {'SRCC':>8} {'KTAU':>8} "
          f"{'Score':>8} {'MSE':>10} {'MAE':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    rows = [
        ('Final Teacher', t_total, teacher_val),
        ('Final Student + KD', s_total, kd_val),
        ('Final Student', s_total, baseline_val),
    ]
    for name, params, m in rows:
        print(f"  {name:<25} {params:>10,} {m['plcc']:>8.4f} {m['srcc']:>8.4f} "
              f"{m['ktau']:>8.4f} {m['final_score']:>8.4f} {m['mse']:>10.6f} {m['mae']:>8.4f}")

    kd_gain_plcc = kd_val['plcc'] - baseline_val['plcc']
    kd_gain_srcc = kd_val['srcc'] - baseline_val['srcc']
    kd_gain_score = kd_val['final_score'] - baseline_val['final_score']
    print(f"\n  Compression: Teacher({t_total:,}) -> Student({s_total:,}) = {t_total/s_total:.1f}x")
    print(f"  KD gain: PLCC={kd_gain_plcc:+.4f}, SRCC={kd_gain_srcc:+.4f}, Score={kd_gain_score:+.4f}")

    report = {
        'pipeline': 'final',
        'data_path': args.data,
        'n_train': len(train_data),
        'n_val': len(val_data),
        'selection_metric': args.selection_metric,
        'teacher': {'params': t_total, **teacher_val},
        'student_kd': {'params': s_total, **kd_val, 'kd_weights': kd_weights},
        'student_baseline': {'params': s_total, **baseline_val},
        'kd_gain_plcc': kd_gain_plcc,
        'kd_gain_srcc': kd_gain_srcc,
        'kd_gain_final_score': kd_gain_score,
    }
    report_path = os.path.join(args.save_dir, 'final_experiment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Report saved: {report_path}")
    print(f"  Models saved: {args.save_dir}/")
    print(f"{'='*70}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="KD v2 final experiment for Distil-ShortVU")
    parser.add_argument('--data', required=True, help='Path to v2 extracted features JSON/JSONL')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or mps')
    parser.add_argument('--save-dir', default='results_kd_v2', help='Output directory')

    parser.add_argument('--teacher-hidden', type=int, default=512)
    parser.add_argument('--teacher-blocks', type=int, default=2)
    parser.add_argument('--teacher-epochs', type=int, default=40)
    parser.add_argument('--teacher-lr', type=float, default=3e-4)

    parser.add_argument('--student-hidden', type=int, default=256)
    parser.add_argument('--student-epochs', type=int, default=60)
    parser.add_argument('--student-lr', type=float, default=5e-4)
    parser.add_argument(
        '--selection-metric',
        choices=['final_score', 'srcc', 'plcc', 'mse', 'mae'],
        default='final_score',
        help='Validation metric used for checkpoint selection.',
    )

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=16)

    parser.add_argument('--alpha', type=float, default=0.5, help='Soft ECR KD loss weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Hidden representation KD weight')
    parser.add_argument('--attn-kd', type=float, default=0.1, help='Temporal attention KD weight')
    parser.add_argument('--gamma', type=float, default=0.2, help='Aesthetic auxiliary loss weight')
    parser.add_argument('--delta', type=float, default=0.2, help='Technical auxiliary loss weight')

    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--max-motion-clips', type=int, default=4)
    parser.add_argument('--quick', action='store_true', help='Quick smoke test')
    args = parser.parse_args()

    if args.quick:
        args.teacher_epochs = 15
        args.student_epochs = 20

    os.makedirs(args.save_dir, exist_ok=True)
    run_experiment(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='results/dover_features_5000.npz')
args = parser.parse_args()

batch_files = sorted(Path('results/dover_batches').glob('batch_*.npz'))
results = {
    'ids': [],
    'technical_score': [],
    'aesthetic_score': [],
    'technical_feature': [],
    'aesthetic_feature': [],
}

for bf in batch_files:
    d = np.load(bf)
    results['ids'].extend(d['ids'].tolist())
    results['technical_score'].extend(d['technical_score'].tolist())
    results['aesthetic_score'].extend(d['aesthetic_score'].tolist())
    results['technical_feature'].append(d['technical_feature'])
    results['aesthetic_feature'].append(d['aesthetic_feature'])

np.savez(args.out,
    ids=np.array(results['ids']),
    technical_score=np.array(results['technical_score'], dtype=np.float32),
    aesthetic_score=np.array(results['aesthetic_score'], dtype=np.float32),
    technical_feature=np.concatenate(results['technical_feature'], axis=0).astype(np.float32),
    aesthetic_feature=np.concatenate(results['aesthetic_feature'], axis=0).astype(np.float32),
)
print(f'Merged {len(batch_files)} batches -> {args.out} ({len(results[\"ids\"])} videos)')

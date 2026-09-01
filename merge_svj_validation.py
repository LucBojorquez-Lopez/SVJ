#!/usr/bin/env python3
"""
merge_svj_validation.py
=======================
Merge the per-task NPZ shards produced by condor/validation.sub into a
single validation_production.npz.

Usage (from project root, after all array tasks finish):
    python merge_svj_validation.py

Arrays with a leading N3 dimension are concatenated; scalar metadata and
string arrays are taken from the first available shard.
"""

import sys
import numpy as np
from pathlib import Path

N_JOBS   = 16
SHARD_PATTERN = 'simulated/svj/validation_production_{}.npz'
OUT_PATH  = Path('simulated/svj/validation_production.npz')

shards = []
for i in range(N_JOBS):
    p = Path(SHARD_PATTERN.format(i))
    if not p.exists():
        print(f'  WARNING: shard {i} not found at {p} — skipping')
        continue
    shards.append((i, np.load(p, allow_pickle=True)))
    print(f'  loaded shard {i}  ({int(shards[-1][1]["N3"])} points)')

if not shards:
    print('ERROR: no shards found. Run condor_submit condor/validation.sub first.')
    sys.exit(1)

first = shards[0][1]

def concat(key):
    return np.concatenate([s[key] for _, s in shards], axis=0)

total_N3 = sum(int(s['N3']) for _, s in shards)

merged = dict(
    js_baseline          = concat('js_baseline'),
    js_interp            = concat('js_interp'),
    js_nearest           = concat('js_nearest'),
    mmd_baseline         = concat('mmd_baseline'),
    mmd_interp           = concat('mmd_interp'),
    mmd_nearest          = concat('mmd_nearest'),
    val_scan_params      = concat('val_scan_params'),
    val_frac_idxs        = concat('val_frac_idxs'),
    nearest_scan_params  = concat('nearest_scan_params'),
    obs_names            = first['obs_names'],
    axis_names           = first['axis_names'],
    N1                   = first['N1'],
    N2                   = first['N2'],
    N3                   = np.int64(total_N3),
    bins                 = first['bins'],
    n_mmd                = first['n_mmd'],
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT_PATH, **merged)
print(f'\nMerged {len(shards)}/{N_JOBS} shards '
      f'({total_N3} total points) → {OUT_PATH}')

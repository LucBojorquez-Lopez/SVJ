#!/usr/bin/env python3
"""
validate_production.py
======================
Full production validation of a finalized scan against PYTHIA truth.

For N3 randomly sampled interior points (uniform in grid-index space) the
script runs three comparisons that place the interpolated model in context:

  Baseline    JS(truth_1, truth_2)          — statistical noise floor
  Interpolation  JS(model,   truth_1)       — quality of the interpolation
  Nearest grid   JS(nearest, truth_1)       — naive nearest-grid-point alternative

where truth_1 and truth_2 are two independent PYTHIA runs at the validation
point, model is N1 samples from the interpolated distribution, and nearest is
a PYTHIA run at the closest evaluated grid point.

The same three comparisons are made for the joint MMD (all observables together)
to capture inter-observable correlations that per-marginal JS misses.

Intended workflow
-----------------
  1. validate_fit.py     → confirm observable transforms are satisfactory
  2. validate_grid.py    → confirm interpolation works on a pre-production scan
  3. validate_production.py → full benchmark on the finalised scan

Usage
-----
python src/run_regression/validate_production.py \\
    simulated/svj/svj_scan.npz \\
    [--N1 50000] [--N2 5000] [--N3 30] \\
    [--n-workers 4] [--n-mmd 2000] [--bins 80] [--seed 0] \\
    --out simulated/svj/validation_production.npz

Output NPZ keys
---------------
js_baseline           (N3, n_obs)   JS(truth_1, truth_2)
js_interp             (N3, n_obs)   JS(model,   truth_1)
js_nearest            (N3, n_obs)   JS(nearest, truth_1)
mmd_baseline          (N3,)         MMD(truth_1, truth_2)
mmd_interp            (N3,)         MMD(model,   truth_1)
mmd_nearest           (N3,)         MMD(nearest, truth_1)
obs_names             (n_obs,)
axis_names            (K,)
val_scan_params       (N3, K)       scan-axis values at each validation point
val_frac_idxs        (N3, K)       fractional grid indices
nearest_scan_params   (N3, K)       scan-axis values of the nearest grid point
N1, N2, N3, bins, n_mmd  int scalars
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from observables import (
    OBSERVABLES, event_valid_mask, validate_scan_selection,
)
from helpers import sample_svj_new
from _val_utils import (
    BINARY,
    load_scan, resolve_point, interp_at,
    sample_interior_points, nearest_grid_point,
    run_pythia, js_per_obs, mmd_rbf, extract_obs_cols,
)


# ── Worker (module-level for ProcessPoolExecutor pickling) ─────────────────────

def _worker(args):
    """
    Run one production validation point: 3 PYTHIA calls + sampling + JS + MMD.

    seed_offset scheme (nWorkers=1 → PYTHIA seed = 1 + seed_offset):
      truth_1 at val point   : seed_offset = 3 * task_id
      truth_2 at val point   : seed_offset = 3 * task_id + 1
      nearest at grid point  : seed_offset = 3 * task_id + 2
    """
    (task_id,
     full_params,           # fully resolved physics params at validation point
     nearest_full_params,   # fully resolved physics params at nearest grid point
     R_upper, flat_obs,
     param_offsets_arr, obs_names,
     N1, N2, bins, n_mmd) = args

    so1 = 3 * task_id
    so2 = 3 * task_id + 1
    so3 = 3 * task_id + 2

    # ── Run PYTHIA ──────────────────────────────────────────────────────────────
    try:
        data1, cmap1 = run_pythia(full_params,         N2, so1, f'{task_id}_t1')
        data2, cmap2 = run_pythia(full_params,         N2, so2, f'{task_id}_t2')
        datan, cmapn = run_pythia(nearest_full_params, N2, so3, f'{task_id}_ng')
    except RuntimeError as e:
        return task_id, None, str(e)

    # ── Filter and extract observable columns ────────────────────────────────────
    try:
        mask1, _ = event_valid_mask(data1,  obs_names, cmap1)
        mask2, _ = event_valid_mask(data2,  obs_names, cmap2)
        maskn, _ = event_valid_mask(datan,  obs_names, cmapn)
        T1 = extract_obs_cols(data1[mask1], cmap1, obs_names)
        T2 = extract_obs_cols(data2[mask2], cmap2, obs_names)
        TN = extract_obs_cols(datan[maskn], cmapn, obs_names)

        if len(T1) < 20 or len(T2) < 20 or len(TN) < 20:
            return task_id, None, 'Too few valid events after cuts'

        T1 = T1[:N2]
        T2 = T2[:N2]
        TN = TN[:N2]

    except Exception as e:
        return task_id, None, f'Event processing: {e}'

    # ── Sample from interpolated model ─────────────────────────────────────────
    try:
        model = sample_svj_new(
            R_upper, flat_obs, param_offsets_arr, obs_names,
            n_samples=N1)
    except Exception as e:
        return task_id, None, f'Sampling failed: {e}'

    # ── JS divergences ─────────────────────────────────────────────────────────
    js_base    = js_per_obs(T1,    T2, obs_names, bins=bins)
    js_inter   = js_per_obs(model, T1, obs_names, bins=bins)
    js_near    = js_per_obs(TN,    T1, obs_names, bins=bins)

    # ── MMD (joint, all observables) ───────────────────────────────────────────
    mmd_base  = mmd_rbf(T1,    T2,    n_sub=n_mmd)
    mmd_inter = mmd_rbf(model, T1,    n_sub=n_mmd)
    mmd_near  = mmd_rbf(TN,    T1,    n_sub=n_mmd)

    return task_id, {
        'js_baseline': js_base,
        'js_interp':   js_inter,
        'js_nearest':  js_near,
        'mmd_baseline': mmd_base,
        'mmd_interp':   mmd_inter,
        'mmd_nearest':  mmd_near,
    }, None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Full production validation with nearest-grid benchmark and MMD.')
    ap.add_argument('scan_npz', help='Path to svj_scan.npz')
    ap.add_argument('--N1', type=int, default=50_000,
                    help='Model samples per validation point (default: 50000)')
    ap.add_argument('--N2', type=int, default=5_000,
                    help='PYTHIA events per run per point (default: 5000)')
    ap.add_argument('--N3', type=int, default=30,
                    help='Number of interior validation points (default: 30)')
    ap.add_argument('--n-workers', type=int, default=4,
                    help='Parallel workers; each runs 3 PYTHIA calls (default: 4)')
    ap.add_argument('--n-mmd', type=int, default=2000,
                    help='Subsample size for MMD computation (default: 2000)')
    ap.add_argument('--bins', type=int, default=80,
                    help='Histogram bins for JS distance (default: 80)')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for point sampling (default: 0)')
    ap.add_argument('--out', required=True, help='Output NPZ path')
    args = ap.parse_args()

    if not os.path.exists(BINARY):
        print(f'ERROR: binary not found at {BINARY}. '
              "Run 'make svj_regression' first.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # ── Load scan ──────────────────────────────────────────────────────────────
    print(f'Loading scan: {args.scan_npz}')
    scan = load_scan(args.scan_npz)
    axis_names = scan['axis_names']
    obs_names  = scan['obs_names']
    K     = len(axis_names)
    n_obs = len(obs_names)

    validate_scan_selection(obs_names)
    grid_shape = ' × '.join(str(len(v)) for v in scan['axis_vals'])
    print(f"  Axes ({K}): {axis_names}   grid: {grid_shape}")
    print(f"  Observables ({n_obs}): {obs_names}")
    print(f"  MMD subsample: {args.n_mmd}")

    # ── Sample N3 interior validation points ───────────────────────────────────
    frac_idxs, scan_points = sample_interior_points(scan, args.N3, rng)
    print(f'\nSampled {args.N3} interior validation points.')

    # ── Pre-compute interpolated params and find nearest grid points ───────────
    tasks = []
    nearest_scan_pts = []   # for saving to NPZ
    for i, (fi, sp) in enumerate(zip(frac_idxs, scan_points)):
        full_params = resolve_point(sp, scan['fixed_params'], scan['derived_exprs'])

        _, near_sp = nearest_grid_point(scan, fi)
        nearest_full_params = resolve_point(
            near_sp, scan['fixed_params'], scan['derived_exprs'])
        nearest_scan_pts.append(near_sp)

        try:
            R_upper, flat_obs = interp_at(scan, sp)
        except Exception as e:
            print(f'  WARNING: interpolation failed for point {i} '
                  f'({sp}): {e} — skipping.')
            nearest_scan_pts.pop()   # keep list aligned with tasks
            continue

        if not np.all(np.isfinite(flat_obs)):
            print(f'  WARNING: NaN in interpolated params for point {i} '
                  '(scan may have missing grid points) — skipping.')
            nearest_scan_pts.pop()
            continue

        tasks.append((
            i,
            full_params,
            nearest_full_params,
            R_upper, flat_obs,
            scan['param_offsets'], obs_names,
            args.N1, args.N2, args.bins, args.n_mmd,
        ))

    if not tasks:
        print('ERROR: no valid interior points could be prepared.')
        sys.exit(1)

    # ── Run in parallel ────────────────────────────────────────────────────────
    n_total = len(tasks)
    js_baseline = np.full((args.N3, n_obs), np.nan)
    js_interp   = np.full((args.N3, n_obs), np.nan)
    js_nearest  = np.full((args.N3, n_obs), np.nan)
    mmd_baseline = np.full(args.N3, np.nan)
    mmd_interp   = np.full(args.N3, np.nan)
    mmd_nearest  = np.full(args.N3, np.nan)

    n_done = n_fail = 0
    w = len(str(n_total))

    print(f'\nRunning {n_total} points × 3 PYTHIA calls '
          f'(--n-workers {args.n_workers}) ...\n')

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            task_id, result, err = fut.result()
            n_done += 1
            if result is None:
                n_fail += 1
                print(f'  [{n_done:>{w}}/{n_total}] '
                      f'point {task_id:>3}: FAILED — {err}', flush=True)
            else:
                js_baseline[task_id]  = result['js_baseline']
                js_interp[task_id]    = result['js_interp']
                js_nearest[task_id]   = result['js_nearest']
                mmd_baseline[task_id] = result['mmd_baseline']
                mmd_interp[task_id]   = result['mmd_interp']
                mmd_nearest[task_id]  = result['mmd_nearest']
                print(
                    f'  [{n_done:>{w}}/{n_total}] point {task_id:>3}: ok  '
                    f'js_base={np.nanmean(result["js_baseline"]):.4f}  '
                    f'js_interp={np.nanmean(result["js_interp"]):.4f}  '
                    f'js_near={np.nanmean(result["js_nearest"]):.4f}  '
                    f'mmd_interp={result["mmd_interp"]:.4f}',
                    flush=True)

    print(f'\nDone. {n_done - n_fail}/{n_total} succeeded, {n_fail} failed.')

    # ── Save ───────────────────────────────────────────────────────────────────
    val_scan  = np.array([[sp[n] for n in axis_names] for sp in scan_points])
    near_scan = np.array([[sp[n] for n in axis_names] for sp in nearest_scan_pts])
    # nearest_scan_pts may be shorter if some points were skipped; pad with NaN
    if len(near_scan) < args.N3:
        pad = np.full((args.N3 - len(near_scan), K), np.nan)
        near_scan = np.vstack([near_scan, pad])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        js_baseline          = js_baseline,
        js_interp            = js_interp,
        js_nearest           = js_nearest,
        mmd_baseline         = mmd_baseline,
        mmd_interp           = mmd_interp,
        mmd_nearest          = mmd_nearest,
        obs_names            = np.array(obs_names,  dtype=object),
        axis_names           = np.array(axis_names, dtype=object),
        val_scan_params      = val_scan,
        val_frac_idxs        = frac_idxs,
        nearest_scan_params  = near_scan,
        N1                   = np.int64(args.N1),
        N2                   = np.int64(args.N2),
        N3                   = np.int64(args.N3),
        bins                 = np.int64(args.bins),
        n_mmd                = np.int64(args.n_mmd),
    )
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
validate_grid.py
================
Pre-production check that the scan interpolation reproduces PYTHIA truth
at randomly sampled interior points of the grid.

Run this after validate_fit.py has confirmed that your observable transforms
and distributions are satisfactory, and after a scan has completed.

For N3 randomly sampled interior points (uniform in grid-index space):
  1. Runs PYTHIA twice at each point (N2 events each) → truth_1, truth_2.
  2. Interpolates the model at the point; draws N1 samples.
  3. Computes Jensen-Shannon distance per observable for:
       js_baseline : JS(truth_1, truth_2)  — statistical noise floor
       js_interp   : JS(model,   truth_1)  — interpolation quality

This tells you whether the interpolation is doing better than the statistical
noise floor, and by how much.  No nearest-grid comparison; no MMD.
See validate_production.py for the full benchmarked validation.

Usage
-----
python src/run_regression/validate_grid.py \\
    simulated/svj/svj_scan.npz \\
    [--N1 50000] [--N2 5000] [--N3 20] \\
    [--n-workers 4] [--bins 80] [--seed 0] \\
    --out simulated/svj/validation_grid.npz

Output NPZ keys
---------------
js_baseline      (N3, n_obs)  JS(truth_1, truth_2) per point
js_interp        (N3, n_obs)  JS(model,   truth_1) per point
obs_names        (n_obs,)
axis_names       (K,)
val_scan_params  (N3, K)      scan-axis values at each validation point
val_frac_idxs   (N3, K)      fractional grid indices
N1, N2, N3, bins int scalars
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
    sample_interior_points,
    run_pythia, js_per_obs, extract_obs_cols,
)


# ── Worker (module-level for ProcessPoolExecutor pickling) ─────────────────────

def _worker(args):
    """
    Run one validation point: 2 PYTHIA calls + interpolated sampling + JS.

    seed_offset scheme (nWorkers=1 → seed = 1 + seed_offset):
      truth_1: seed_offset = 2 * task_id
      truth_2: seed_offset = 2 * task_id + 1
    """
    (task_id,
     full_params,
     R_upper, nu, flat_obs,
     param_offsets_arr, obs_names,
     N1, N2, bins) = args

    so1 = 2 * task_id
    so2 = 2 * task_id + 1

    try:
        data1, cmap1 = run_pythia(full_params, N2, so1, f'{task_id}_t1')
        data2, cmap2 = run_pythia(full_params, N2, so2, f'{task_id}_t2')
    except RuntimeError as e:
        return task_id, None, str(e)

    try:
        mask1, _ = event_valid_mask(data1, obs_names, cmap1)
        mask2, _ = event_valid_mask(data2, obs_names, cmap2)
        T1 = extract_obs_cols(data1[mask1], cmap1, obs_names)
        T2 = extract_obs_cols(data2[mask2], cmap2, obs_names)

        if len(T1) < 20 or len(T2) < 20:
            return task_id, None, 'Too few valid events after cuts'

        # Cap at N2 rows in case the binary produced more
        T1 = T1[:N2]
        T2 = T2[:N2]

    except Exception as e:
        return task_id, None, f'Event processing: {e}'

    try:
        model = sample_svj_new(
            R_upper, nu, flat_obs, param_offsets_arr, obs_names,
            n_samples=N1)
    except Exception as e:
        return task_id, None, f'Sampling failed: {e}'

    js_base  = js_per_obs(T1,    T2, obs_names, bins=bins)
    js_inter = js_per_obs(model, T1, obs_names, bins=bins)

    return task_id, {'js_baseline': js_base, 'js_interp': js_inter}, None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Pre-production interpolation check against PYTHIA truth.')
    ap.add_argument('scan_npz', help='Path to svj_scan.npz')
    ap.add_argument('--N1', type=int, default=50_000,
                    help='Model samples per validation point (default: 50000)')
    ap.add_argument('--N2', type=int, default=5_000,
                    help='PYTHIA events per run per point (default: 5000)')
    ap.add_argument('--N3', type=int, default=20,
                    help='Number of interior validation points (default: 20)')
    ap.add_argument('--n-workers', type=int, default=4,
                    help='Parallel workers; each runs 2 PYTHIA calls (default: 4)')
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

    # ── Sample N3 interior points ──────────────────────────────────────────────
    frac_idxs, scan_points = sample_interior_points(scan, args.N3, rng)
    print(f'\nSampled {args.N3} interior validation points.')

    # ── Pre-compute interpolated params in the main process ────────────────────
    tasks = []
    for i, (fi, sp) in enumerate(zip(frac_idxs, scan_points)):
        full_params = resolve_point(sp, scan['fixed_params'], scan['derived_exprs'])
        try:
            R_upper, nu, flat_obs = interp_at(scan, sp)
        except Exception as e:
            print(f'  WARNING: interpolation failed for point {i} '
                  f'({sp}): {e} — skipping.')
            continue

        if not np.all(np.isfinite(flat_obs)):
            print(f'  WARNING: NaN in interpolated params for point {i} '
                  '(scan may have missing grid points) — skipping.')
            continue

        tasks.append((
            i,
            full_params,
            R_upper, nu, flat_obs,
            scan['param_offsets'], obs_names,
            args.N1, args.N2, args.bins,
        ))

    if not tasks:
        print('ERROR: no valid interior points could be prepared.')
        sys.exit(1)

    # ── Run in parallel ────────────────────────────────────────────────────────
    n_total = len(tasks)
    js_baseline = np.full((args.N3, n_obs), np.nan)
    js_interp   = np.full((args.N3, n_obs), np.nan)
    n_done = n_fail = 0

    print(f'\nRunning {n_total} points × 2 PYTHIA calls '
          f'(--n-workers {args.n_workers}) ...\n')

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            task_id, result, err = fut.result()
            n_done += 1
            if result is None:
                n_fail += 1
                print(f'  [{n_done:>{len(str(n_total))}}/{n_total}] '
                      f'point {task_id:>3}: FAILED — {err}', flush=True)
            else:
                js_baseline[task_id] = result['js_baseline']
                js_interp[task_id]   = result['js_interp']
                jb_m = float(np.nanmean(result['js_baseline']))
                ji_m = float(np.nanmean(result['js_interp']))
                print(f'  [{n_done:>{len(str(n_total))}}/{n_total}] '
                      f'point {task_id:>3}: ok  '
                      f'js_base={jb_m:.4f}  js_interp={ji_m:.4f}',
                      flush=True)

    print(f'\nDone. {n_done - n_fail}/{n_total} succeeded, {n_fail} failed.')

    # ── Save ───────────────────────────────────────────────────────────────────
    val_scan = np.array([[sp[n] for n in axis_names] for sp in scan_points])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        js_baseline     = js_baseline,
        js_interp       = js_interp,
        obs_names       = np.array(obs_names,   dtype=object),
        axis_names      = np.array(axis_names,  dtype=object),
        val_scan_params = val_scan,
        val_frac_idxs   = frac_idxs,
        N1              = np.int64(args.N1),
        N2              = np.int64(args.N2),
        N3              = np.int64(args.N3),
        bins            = np.int64(args.bins),
    )
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

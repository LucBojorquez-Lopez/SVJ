#!/usr/bin/env python3
"""
fit_raw.py
==========
Re-fit the SVJ scan interpolation on previously saved raw event data without
re-running the event generator.

Requires a raw NPZ file produced by scan_svj.py --save-raw:
  raw_flat       (total_events, n_obs)
  raw_grid_flat  (total_events, 4)  — [i,j,k,l] for each event

Also requires the scan NPZ (for grid axes and metadata):
  svj_scan.npz  (any format — used only for grid axis vectors)

Usage
-----
    python fit_raw.py raw_events.npz [options]

    --scan-npz PATH      scan NPZ to read grid axes from (default: sibling of raw NPZ)
    --obs LIST           comma-separated observable names (default: from scan NPZ obs_names)
    --out-npz PATH       output NPZ (default: raw NPZ dir / svj_scan_refit.npz)
    --mvt-iters N        max EM iterations for MVT fit (default: 200)
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))

from observables import (
    OBSERVABLES, DEFAULT_SCAN,
    n_fitted_params, param_offsets as obs_param_offsets,
    fit_observable_col, event_valid_mask, validate_scan_selection,
)
from scan_svj import fit_mvt_em, _save, _save_metadata


# ── Per-point re-fitting worker ────────────────────────────────────────────────

def _refit_worker(args):
    """Re-fit one grid point from pre-filtered raw data."""
    i, j, k, l, X_raw, obs_selection, mvt_iters = args
    fail = (i, j, k, l, None, None, None)

    if len(X_raw) < 20:
        return fail

    try:
        offsets  = obs_param_offsets(obs_selection)
        n_params = int(offsets[-1])
        flat_p   = np.empty(n_params)
        tr_cols  = []

        for idx, obs_name in enumerate(obs_selection):
            spec     = OBSERVABLES[obs_name]
            pipeline = spec['pipeline']
            dist     = spec['distribution']
            col_idx  = idx   # X_raw columns are already in obs_selection order
            x_col    = X_raw[:, col_idx].copy()
            y_std, params = fit_observable_col(x_col, pipeline, dist)
            p_start = int(offsets[idx])
            p_end   = int(offsets[idx + 1])
            flat_p[p_start:p_end] = params
            tr_cols.append(y_std)

        tr_data = np.column_stack(tr_cols)
        R_upper, nu = fit_mvt_em(tr_data, max_iter=mvt_iters)
        return (i, j, k, l, flat_p, R_upper, nu)
    except Exception:
        return fail


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Re-fit SVJ scan interpolation from saved raw events.')
    parser.add_argument('raw_npz', help='Path to *_raw.npz produced by --save-raw')
    parser.add_argument('--scan-npz', default=None,
                        help='Scan NPZ to read grid axes from')
    parser.add_argument('--obs', default=None,
                        help='Comma-separated observable names (default: from scan NPZ)')
    parser.add_argument('--out-npz', default=None,
                        help='Output NPZ path')
    parser.add_argument('--mvt-iters', type=int, default=200,
                        help='Max EM iterations for MVT fit')
    parser.add_argument('--n-workers', type=int, default=8,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    raw_path = Path(args.raw_npz)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        sys.exit(1)

    # Locate scan NPZ
    if args.scan_npz:
        scan_path = Path(args.scan_npz)
    else:
        scan_path = raw_path.parent / 'svj_scan.npz'
    if not scan_path.exists():
        print(f"Error: scan NPZ not found at {scan_path}. Use --scan-npz.")
        sys.exit(1)

    raw   = np.load(raw_path, allow_pickle=True)
    scan  = np.load(scan_path, allow_pickle=True)

    raw_flat      = np.array(raw['raw_flat'])       # (total_events, n_obs)
    raw_grid_flat = np.array(raw['raw_grid_flat'])  # (total_events, 4)

    mZ_vals     = scan['mZ_vals']
    mRho_vals   = scan['mRho_vals']
    rinv_vals   = scan['rinv_vals']
    alphaD_vals = scan['alphaD_vals']
    scan_params = np.array(scan['scan_params'])

    # Observable selection
    if args.obs:
        obs_selection = [s.strip() for s in args.obs.split(',')]
    elif 'obs_names' in scan:
        obs_selection = list(scan['obs_names'])
    else:
        obs_selection = DEFAULT_SCAN
    validate_scan_selection(obs_selection)

    n_obs    = len(obs_selection)
    n_corr   = n_obs * (n_obs - 1) // 2
    obs_offsets  = obs_param_offsets(obs_selection)
    total_params = int(obs_offsets[-1]) + n_corr + 1

    mZ_n     = len(mZ_vals)
    mRho_n   = len(mRho_vals)
    rinv_n   = len(rinv_vals)
    alphaD_n = len(alphaD_vals)
    param_flat = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, total_params), np.nan)

    # Build per-grid-point event arrays
    grid_events: dict[tuple, list] = {}
    for evt_idx in range(len(raw_flat)):
        ijk = tuple(raw_grid_flat[evt_idx].astype(int))
        if ijk not in grid_events:
            grid_events[ijk] = []
        grid_events[ijk].append(raw_flat[evt_idx])

    tasks = []
    for (i, j, k, l), evts in grid_events.items():
        X_raw = np.vstack(evts)
        tasks.append((i, j, k, l, X_raw, obs_selection, args.mvt_iters))

    n_todo = len(tasks)
    print(f"Re-fitting {n_todo} grid points from {raw_path.name}  "
          f"({n_obs} observables, {args.n_workers} workers)")

    out_path = (Path(args.out_npz) if args.out_npz
                else raw_path.parent / 'svj_scan_refit.npz')

    corr_start = int(obs_offsets[-1])

    t0 = time.time()
    done = failed = 0
    width = len(str(n_todo))

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(_refit_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                result = fut.result()
            except Exception as e:
                print(f"WARNING: worker exception: {e}", flush=True)
                failed += 1
                done   += 1
                continue

            i, j, k, l, flat_p, R_upper, nu = result
            done += 1
            ok = flat_p is not None
            if ok:
                param_flat[i, j, k, l, :corr_start]   = flat_p
                param_flat[i, j, k, l, corr_start:-1] = R_upper
                param_flat[i, j, k, l, -1]            = nu
            else:
                failed += 1

            elapsed = time.time() - t0
            avg     = elapsed / done
            h, rem  = divmod(int(avg * (n_todo - done)), 3600)
            mm, s   = divmod(rem, 60)
            print(f"[{done:{width}}/{n_todo}]  ({i},{j},{k},{l})"
                  f"  {'ok  ' if ok else 'FAIL'}"
                  f"  ETA {h:02d}:{mm:02d}:{s:02d}", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min; {failed}/{n_todo} failed.")

    SCAN_PARAM_NAMES = np.array(
        ['mZ', 'mRho', 'mPi', 'LambdaQCD', 'rinv', 'alphaD'], dtype=object)
    _save(out_path, mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
          param_flat, obs_offsets, obs_selection,
          scan_params, SCAN_PARAM_NAMES, n_obs)
    print(f"Saved → {out_path}")

    _save_metadata(out_path.parent, obs_selection, obs_offsets, n_corr, {})


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
fit_raw.py
==========
Re-fit the SVJ scan interpolation on previously saved raw event data without
re-running the event generator.

Requires a raw NPZ file produced by scan_svj.py --save-raw:
  raw_flat       (total_events, n_obs)
  raw_grid_flat  (total_events, K)  — K grid-axis indices per event

Also requires the scan NPZ (to read grid axis metadata):
  svj_scan.npz  — used for axis_names, {name}_vals, scan_params, scan_param_names

Usage
-----
    python fit_raw.py raw_events.npz [options]

    --scan-npz PATH      scan NPZ to read grid axes from (default: sibling of raw NPZ)
    --obs LIST           comma-separated observable names (default: from scan NPZ obs_names)
    --out-npz PATH       output NPZ (default: raw NPZ dir / svj_scan_refit.npz)
    --mvt-iters N        max EM iterations for MVT fit (default: 200)
    --n-workers N        parallel worker processes (default: 8)
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))

from observables import (
    OBSERVABLES, DEFAULT_SCAN,
    param_offsets as obs_param_offsets,
    fit_observable_col, validate_scan_selection,
)
from scan_svj import fit_mvt_em


# ── Per-point re-fitting worker ────────────────────────────────────────────────

def _refit_worker(args):
    """
    Re-fit one grid point from pre-filtered raw data.

    args = (grid_indices, X_raw, obs_selection, mvt_iters)

    Returns (grid_indices, flat_p, R_upper, nu) on success,
            (grid_indices, None, None, None)     on failure.
    """
    grid_indices, X_raw, obs_selection, mvt_iters = args
    fail = (grid_indices, None, None, None)

    if len(X_raw) < 20:
        return fail

    try:
        offsets  = obs_param_offsets(obs_selection)
        n_params = int(offsets[-1])
        flat_p   = np.empty(n_params)
        tr_cols  = []

        for idx, obs_name in enumerate(obs_selection):
            spec  = OBSERVABLES[obs_name]
            x_col = X_raw[:, idx].copy()   # X_raw columns are in obs_selection order
            y_std, params = fit_observable_col(x_col, spec['pipeline'],
                                               spec['distribution'])
            flat_p[int(offsets[idx]):int(offsets[idx + 1])] = params
            tr_cols.append(y_std)

        R_upper, nu = fit_mvt_em(np.column_stack(tr_cols), max_iter=mvt_iters)
        return (grid_indices, flat_p, R_upper, nu)
    except Exception:
        return fail


# ── NPZ writer (mirrors scan_svj._save without requiring ScanConfig) ──────────

def _save_refit(out_file, axis_names, axis_vals_dict,
                param_flat, obs_offsets, obs_names,
                scan_params, scan_param_names, n_obs):
    n_corr     = n_obs * (n_obs - 1) // 2
    corr_start = int(obs_offsets[-1])
    nu_idx     = corr_start + n_corr
    kwargs = dict(
        axis_names       = np.array(axis_names, dtype=object),
        param_flat       = param_flat,
        param_offsets    = obs_offsets,
        corr_start       = np.array(corr_start, dtype=int),
        nu_idx           = np.array(nu_idx,     dtype=int),
        obs_names        = np.array(obs_names,  dtype=object),
        scan_params      = scan_params,
        scan_param_names = np.array(scan_param_names, dtype=object),
    )
    for name, vals in axis_vals_dict.items():
        kwargs[f'{name}_vals'] = vals
    np.savez(out_file, **kwargs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Re-fit SVJ scan interpolation from saved raw events.')
    parser.add_argument('raw_npz', help='Path to *_raw.npz produced by --save-raw')
    parser.add_argument('--scan-npz', default=None,
                        help='Scan NPZ to read grid axes from (default: sibling)')
    parser.add_argument('--obs', default=None,
                        help='Comma-separated observable names (default: from scan NPZ)')
    parser.add_argument('--out-npz', default=None,
                        help='Output NPZ path (default: sibling svj_scan_refit.npz)')
    parser.add_argument('--mvt-iters', type=int, default=200,
                        help='Max EM iterations for MVT fit')
    parser.add_argument('--n-workers', type=int, default=8,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    raw_path = Path(args.raw_npz)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        sys.exit(1)

    scan_path = Path(args.scan_npz) if args.scan_npz else raw_path.parent / 'svj_scan.npz'
    if not scan_path.exists():
        print(f"Error: scan NPZ not found at {scan_path}. Use --scan-npz.")
        sys.exit(1)

    raw  = np.load(raw_path,  allow_pickle=True)
    scan = np.load(scan_path, allow_pickle=True)

    raw_flat      = np.array(raw['raw_flat'])       # (total_events, n_raw_obs)
    raw_grid_flat = np.array(raw['raw_grid_flat'])  # (total_events, K)

    # ── Read dynamic axis info from scan NPZ ─────────────────────────────────
    axis_names     = list(scan['axis_names'])
    axis_vals_dict = {name: np.array(scan[f'{name}_vals']) for name in axis_names}
    axis_sizes     = [len(axis_vals_dict[n]) for n in axis_names]
    K              = len(axis_names)

    scan_params      = np.array(scan['scan_params'])
    scan_param_names = list(scan['scan_param_names'])

    if raw_grid_flat.shape[1] != K:
        print(f"Error: raw_grid_flat has {raw_grid_flat.shape[1]} index columns "
              f"but scan NPZ has {K} axes ({axis_names}). NPZ mismatch.")
        sys.exit(1)

    # ── Observable selection ──────────────────────────────────────────────────
    if args.obs:
        obs_selection = [s.strip() for s in args.obs.split(',')]
    elif 'obs_names' in scan:
        obs_selection = list(scan['obs_names'])
    else:
        obs_selection = DEFAULT_SCAN
    validate_scan_selection(obs_selection)

    n_obs        = len(obs_selection)
    n_corr       = n_obs * (n_obs - 1) // 2
    obs_offsets  = obs_param_offsets(obs_selection)
    total_params = int(obs_offsets[-1]) + n_corr + 1

    # ── Allocate result arrays ────────────────────────────────────────────────
    grid_shape = tuple(axis_sizes)
    param_flat = np.full(grid_shape + (total_params,), np.nan)

    # ── Build per-grid-point event arrays ─────────────────────────────────────
    grid_events: dict = {}
    for evt_idx in range(len(raw_flat)):
        gidx = tuple(raw_grid_flat[evt_idx].astype(int))
        if gidx not in grid_events:
            grid_events[gidx] = []
        grid_events[gidx].append(raw_flat[evt_idx])

    tasks = [(gidx, np.vstack(evts), obs_selection, args.mvt_iters)
             for gidx, evts in grid_events.items()]

    n_todo = len(tasks)
    print(f"Re-fitting {n_todo} grid points from {raw_path.name}")
    print(f"  Axes ({K}): {', '.join(f'{n}({s})' for n, s in zip(axis_names, axis_sizes))}")
    print(f"  Observables ({n_obs}): {obs_selection}")
    print(f"  Workers: {args.n_workers}")

    out_path   = Path(args.out_npz) if args.out_npz else raw_path.parent / 'svj_scan_refit.npz'
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

            gidx, flat_p, R_upper, nu = result
            done += 1
            ok = flat_p is not None
            if ok:
                param_flat[gidx + (slice(None, corr_start),)] = flat_p
                param_flat[gidx + (slice(corr_start, -1),)]   = R_upper
                param_flat[gidx + (-1,)]                       = nu
            else:
                failed += 1

            elapsed = time.time() - t0
            avg     = elapsed / done
            h, rem  = divmod(int(avg * (n_todo - done)), 3600)
            mm, s   = divmod(rem, 60)
            print(f"[{done:{width}}/{n_todo}]  {gidx}"
                  f"  {'ok  ' if ok else 'FAIL'}"
                  f"  ETA {h:02d}:{mm:02d}:{s:02d}", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min; {failed}/{n_todo} failed.")

    _save_refit(out_path, axis_names, axis_vals_dict,
                param_flat, obs_offsets, obs_selection,
                scan_params, scan_param_names, n_obs)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()

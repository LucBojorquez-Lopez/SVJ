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
    --n-workers N        parallel worker processes (default: 8)
"""

import sys
import json
import argparse
import numpy as np

import raw_store
from pathlib import Path
from concurrent.futures import (ProcessPoolExecutor, wait,
                                FIRST_COMPLETED)
import time

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))

from observables import (
    OBSERVABLES, DEFAULT_SCAN,
    param_offsets as obs_param_offsets,
    fit_observable_col, validate_scan_selection,
)
from scan_svj import fit_mvn_corr


# ── Per-point re-fitting worker ────────────────────────────────────────────────

def _refit_worker(args):
    """
    Re-fit one grid point from pre-filtered raw data.

    args = (grid_indices, X_raw, obs_selection)

    Returns (grid_indices, flat_p, R_upper) on success,
            (grid_indices, None, None)       on failure.
    """
    grid_indices, X_raw, obs_selection = args
    fail = (grid_indices, None, None)

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
            # point_mass matters: several observables have a physical atom at a
            # boundary (maxMuPt/maxElePt = 0 when no lepton is produced, fInv = 0
            # for a fully visible jet, dPhiMET* at +-pi).  scan_svj passes this;
            # omitting it here made every re-fit of a selection containing one of
            # them die inside boxcox with "Data must be positive", which the
            # blanket except turned into a silent FAIL for the whole point.
            y_std, params = fit_observable_col(x_col, spec['pipeline'],
                                               spec['distribution'],
                                               point_mass=spec.get('point_mass'))
            flat_p[int(offsets[idx]):int(offsets[idx + 1])] = params
            tr_cols.append(y_std)

        R_upper = fit_mvn_corr(np.column_stack(tr_cols))
        return (grid_indices, flat_p, R_upper)
    except Exception:
        return fail


# ── NPZ writer (mirrors scan_svj._save without requiring ScanConfig) ──────────

def _save_refit(out_file, axis_names, axis_vals_dict,
                param_flat, obs_offsets, obs_names,
                scan_params, scan_param_names, n_obs):
    corr_start = int(obs_offsets[-1])
    kwargs = dict(
        axis_names       = np.array(axis_names, dtype=object),
        param_flat       = param_flat,
        param_offsets    = obs_offsets,
        corr_start       = np.array(corr_start, dtype=int),
        obs_names        = np.array(obs_names,  dtype=object),
        scan_params      = scan_params,
        scan_param_names = np.array(scan_param_names, dtype=object),
    )
    for name, vals in axis_vals_dict.items():
        kwargs[f'{name}_vals'] = vals
    np.savez(out_file, **kwargs)


def _save_meta(out_path, scan_path, obs_selection, obs_offsets, n_corr,
               axis_vals_dict):
    """
    Write the svj_scan_meta.json beside the refit NPZ.

    Without it the refit is not loadable by the GUI, which reads fixed_params
    and derived_exprs from the meta and otherwise silently falls back to the
    hardcoded defaults in svj_explorer (mq=4, jetR=1) -- wrong for any scan
    whose cfg derives them, i.e. every production scan.

    Those two fields live only in the cfg, not in the NPZ, so they are carried
    over from the input scan's meta.  The fields that DO change with a refit --
    the observable selection and its parameter layout -- are overwritten.
    """
    src = Path(scan_path).parent / 'svj_scan_meta.json'
    meta = {}
    if src.exists():
        try:
            meta = json.loads(src.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {src} is not valid JSON; writing meta without "
                  f"fixed/derived params")
    else:
        print(f"WARNING: no {src.name} beside the scan NPZ -- the refit meta "
              f"will lack fixed_params/derived_exprs, and the GUI will fall "
              f"back to its built-in defaults")

    meta['obs_selection'] = list(obs_selection)
    meta['param_offsets'] = [int(x) for x in obs_offsets]
    meta['n_corr'] = int(n_corr)
    meta['scan_axes'] = {k: [float(v) for v in vals]
                         for k, vals in axis_vals_dict.items()}
    meta.setdefault('fixed_params', {})
    meta.setdefault('derived_exprs', {})
    meta['refit_from'] = str(scan_path)

    dest = Path(out_path).parent / 'svj_scan_meta.json'
    dest.write_text(json.dumps(meta, indent=2))
    print(f"Saved → {dest}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Re-fit SVJ scan interpolation from saved raw events.')
    parser.add_argument('raw_npz', nargs='+',
                        help='Raw store stem(s) or *_raw.npz from --save-raw. '
                             'A production scan is sharded across one stem per '
                             'Condor slice, so a glob is the normal usage: '
                             "'.../svj_scan_*_raw.idx.npz'. RawReader stitches "
                             'them and checks they agree on obs_names.')
    parser.add_argument('--scan-npz', default=None,
                        help='Scan NPZ to read grid axes from (default: sibling)')
    parser.add_argument('--obs', default=None,
                        help='Comma-separated observable names (default: from scan NPZ)')
    parser.add_argument('--out-npz', default=None,
                        help='Output NPZ path (default: sibling svj_scan_refit.npz)')
    parser.add_argument('--n-workers', type=int, default=8,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    raw_paths = [Path(x) for x in args.raw_npz]
    missing = [str(x) for x in raw_paths if not x.exists()]
    if missing:
        print(f"Error: not found: {missing[:5]}"
              + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))
        sys.exit(1)
    raw_path = raw_paths[0]
    if len(raw_paths) > 1:
        print(f"  raw store: {len(raw_paths)} shards under {raw_path.parent}")

    scan_path = Path(args.scan_npz) if args.scan_npz else raw_path.parent / 'svj_scan.npz'
    if not scan_path.exists():
        print(f"Error: scan NPZ not found at {scan_path}. Use --scan-npz.")
        sys.exit(1)

    scan = np.load(scan_path, allow_pickle=True)

    # Two on-disk formats.  The streaming store (raw_store.py) keeps points as
    # contiguous slices of a float32 memmap, so a re-fit reads only the points
    # it needs; the legacy *_raw.npz held one flat array plus a per-event copy
    # of the grid indices, and is still accepted for scans made before that.
    reader = raw_obs_names = None
    if raw_store.exists(raw_path):
        reader        = raw_store.RawReader(raw_paths)
        raw_obs_names = reader.obs_names
    else:
        if len(raw_paths) > 1:
            print("Error: multiple inputs are only supported for the streaming "
                  "raw store; the legacy *_raw.npz format holds one flat array.")
            sys.exit(1)
        raw = np.load(raw_path, allow_pickle=True)
        raw_flat      = np.array(raw['raw_flat'])
        raw_grid_flat = np.array(raw['raw_grid_flat'])
        # The legacy format never recorded which observables its columns were,
        # so the only thing to go on is the scan NPZ they were written with.
        raw_obs_names = ([str(n) for n in scan['obs_names']]
                         if 'obs_names' in scan else None)

    # ── Read dynamic axis info from scan NPZ ─────────────────────────────────
    axis_names     = list(scan['axis_names'])
    axis_vals_dict = {name: np.array(scan[f'{name}_vals']) for name in axis_names}
    axis_sizes     = [len(axis_vals_dict[n]) for n in axis_names]
    K              = len(axis_names)

    scan_params      = np.array(scan['scan_params'])
    scan_param_names = list(scan['scan_param_names'])

    if reader is None and raw_grid_flat.shape[1] != K:
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
    total_params = int(obs_offsets[-1]) + n_corr

    # ── Allocate result arrays ────────────────────────────────────────────────
    grid_shape = tuple(axis_sizes)
    param_flat = np.full(grid_shape + (total_params,), np.nan)

    # ── Column mapping ────────────────────────────────────────────────────────
    # _refit_worker indexes X_raw by position in obs_selection, so the columns
    # it is handed must already be in that order.  The store may hold many more
    # observables than are being re-fitted -- that is the point of saving them
    # all -- so slice the requested ones out here, by name.  Without this a
    # subset re-fit silently reads the first len(obs_selection) saved columns,
    # which are almost never the ones asked for.
    if raw_obs_names is not None:
        missing = [n for n in obs_selection if n not in raw_obs_names]
        if missing:
            print(f"Error: raw store has no column for: {missing}")
            print(f"       stored observables ({len(raw_obs_names)}): {raw_obs_names}")
            sys.exit(1)
        cols = [raw_obs_names.index(n) for n in obs_selection]
    else:
        # Legacy store with no recorded names: fall back to the old positional
        # assumption, which is only correct for a full re-fit in saved order.
        if len(obs_selection) > 0:
            print("  NOTE: raw store records no observable names; assuming its "
                  "columns are exactly obs_selection, in order.")
        cols = list(range(len(obs_selection)))

    # ── Per-grid-point event arrays ───────────────────────────────────────────
    #
    # LAZY on purpose.  This used to be a list comprehension over
    # reader.iter_points(), which materialised every point before fitting
    # started: for the production grid that is 16384 x 20000 x 17 x 8 bytes =
    # 44 GB (float64 doubles the stored float32), and it took down a 57 GB login
    # node before printing a single line.  raw_store exists to stream -- see its
    # module docstring on `request_memory` -- so honour that here and keep only
    # a bounded window of points in flight (see IN_FLIGHT_PER_WORKER below).
    if reader is not None:
        n_todo = reader.n_points
        task_iter = ((gidx, np.asarray(evts[:, cols], dtype=np.float64),
                      obs_selection)
                     for gidx, evts in reader.iter_points())
    else:
        # Legacy single-array format: already fully in memory by construction,
        # so there is nothing to stream and the grouping stays eager.
        grid_events: dict = {}
        for evt_idx in range(len(raw_flat)):
            gidx = tuple(raw_grid_flat[evt_idx].astype(int))
            grid_events.setdefault(gidx, []).append(raw_flat[evt_idx])
        _tasks = [(gidx, np.vstack(evts)[:, cols], obs_selection)
                  for gidx, evts in grid_events.items()]
        n_todo = len(_tasks)
        task_iter = iter(_tasks)

    src_desc = (f'{len(raw_paths)} shards' if len(raw_paths) > 1
                else raw_path.name)
    print(f"Re-fitting {n_todo} grid points from {src_desc}")
    print(f"  Axes ({K}): {', '.join(f'{n}({s})' for n, s in zip(axis_names, axis_sizes))}")
    print(f"  Observables ({n_obs}): {obs_selection}")
    print(f"  Total params/point: {total_params}  (obs: {obs_offsets[-1]}, corr: {n_corr})")
    print(f"  Workers: {args.n_workers}")

    out_path   = Path(args.out_npz) if args.out_npz else raw_path.parent / 'svj_scan_refit.npz'
    corr_start = int(obs_offsets[-1])

    t0 = time.time()
    done = failed = 0
    width = len(str(n_todo))

    # Enough in flight to keep every worker busy through the EOS read latency,
    # small enough that peak memory is a few hundred MB rather than tens of GB.
    window = max(2 * args.n_workers, 8)

    def _submit_next(ex, futs):
        try:
            futs.add(ex.submit(_refit_worker, next(task_iter)))
            return True
        except StopIteration:
            return False

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = set()
        for _ in range(window):
            if not _submit_next(ex, futs):
                break
        while futs:
            finished, futs = wait(futs, return_when=FIRST_COMPLETED)
            for fut in finished:
                _submit_next(ex, futs)
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"WARNING: worker exception: {e}", flush=True)
                    failed += 1
                    done   += 1
                    continue

                gidx, flat_p, R_upper = result
                done += 1
                ok = flat_p is not None
                if ok:
                    param_flat[gidx + (slice(None, corr_start),)] = flat_p
                    param_flat[gidx + (slice(corr_start, None),)] = R_upper
                else:
                    failed += 1

                elapsed = time.time() - t0
                avg     = elapsed / done
                h, rem  = divmod(int(avg * (n_todo - done)), 3600)
                mm, sec = divmod(rem, 60)
                if done % 25 == 0 or done == n_todo or not ok:
                    print(f"[{done:{width}}/{n_todo}]  {gidx}"
                          f"  {'ok  ' if ok else 'FAIL'}"
                          f"  ETA {h:02d}:{mm:02d}:{sec:02d}", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min; {failed}/{n_todo} failed.")

    _save_refit(out_path, axis_names, axis_vals_dict,
                param_flat, obs_offsets, obs_selection,
                scan_params, scan_param_names, n_obs)
    print(f"Saved → {out_path}")
    _save_meta(out_path, scan_path, obs_selection, obs_offsets, n_corr,
               axis_vals_dict)


if __name__ == '__main__':
    main()

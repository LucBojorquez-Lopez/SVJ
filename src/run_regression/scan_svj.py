#!/usr/bin/env python3
"""
scan_svj.py
===========
Scan an arbitrary subset of SVJ physics parameters, fitting a per-observable
transform pipeline + Gaussian (MVN) copula at each grid point.

Which parameters are scanned, fixed, or derived is fully controlled by the
[scan] / [fixed] / [derived] sections of scan_regression.cfg — no code
changes required to add or remove scan axes.

At each grid point the pipeline is:
  1. Run ./svj_regression (save_tsv=1).
  2. Range-check each event against the pipeline requirements for all
     selected observables; discard events failing any check.
  3. For each observable: apply invertible transforms, fit the chosen
     distribution, map to standard-normal (probit of CDF).
  4. Fit a Gaussian copula correlation matrix R to the standard-normal
     columns (sample Pearson correlation).
  5. Store: per-observable transform+dist params in a flat vector and the
     upper triangle of R.

Observable selection comes from src/observables.py DEFAULT_SCAN.  Override
via --obs on the command line (comma-separated list of observable names).

NPZ format (svj_scan.npz):
  axis_names       (K,)                names of the K scan axes (string array)
  {name}_vals      (N_k,)              grid values for each axis k
  param_flat       (N_0,...,N_{K-1}, total_params)
  param_offsets    (n_obs + 1,)        index into param_flat per observable
  corr_start       scalar int          start of R upper-triangle in param_flat
  obs_names        (n_obs,)
  scan_params      (N_0,...,N_{K-1}, n_scan_phys)   all resolved physics params per point
  scan_param_names (n_scan_phys,)

Derived-param expressions (from [derived] section) are also stored in
simulated/svj/svj_scan_meta.json so that downstream tools (e.g. the
explorer's validation runner) can reconstruct the full parameter set.

Resumption:
  If the output NPZ exists, already-finished points are skipped.

Checkpointing:
  Results are saved every checkpoint_every completions.

Merge mode (for SLURM array jobs):
  python scan_svj.py --merge --n-jobs 4
"""

import subprocess
import sys
import os
import tempfile
import time
import json
import itertools
import configparser
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List
import argparse
import resource

_HERE  = Path(__file__).resolve().parent
_SRC   = _HERE.parent
BINARY = str(_SRC / 'generate_events' / 'svj_regression')

sys.path.insert(0, str(_SRC))
from observables import (
    OBSERVABLES, DEFAULT_SCAN,
    n_fitted_params, param_offsets as obs_param_offsets,
    event_valid_mask, fit_observable_col, validate_scan_selection,
    load_tsv,
)


# ── Scan configuration ─────────────────────────────────────────────────────────

@dataclass
class ScanConfig:
    scan_axes:     Dict[str, np.ndarray]   # ordered name → values array
    fixed_params:  Dict[str, float]
    derived_exprs: Dict[str, str]           # name → arithmetic expression string
    sim_params:    Dict[str, object]        # nEvent, nWorkers, n_outer_workers, …


def read_scan_cfg(path: str) -> ScanConfig:
    """
    Parse an INI-style scan config with [scan], [fixed], [derived], [simulation].

    [scan] rows:   name = min, max, n[, spacing]
                   spacing = linear (default) or log
    [derived] rows: name = arithmetic expression using param names, +, -, *, /
    """
    cp = configparser.ConfigParser(
        inline_comment_prefixes=('#',),
        default_section=None,
    )
    cp.optionxform = str   # preserve key case (mRho stays mRho, not mrho)
    cp.read(path)

    scan_axes: Dict[str, np.ndarray] = {}
    if cp.has_section('scan'):
        for name, val in cp['scan'].items():
            parts = [p.strip() for p in val.split(',')]
            if len(parts) < 3:
                raise ValueError(
                    f"[scan] {name} = {val!r}: expected 'min, max, n[, spacing]'")
            lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
            spacing = parts[3].lower() if len(parts) > 3 else 'linear'
            if spacing == 'log':
                vals = np.logspace(np.log10(lo), np.log10(hi), n)
            else:
                vals = np.linspace(lo, hi, n)
            scan_axes[name] = vals

    fixed_params: Dict[str, float] = {}
    if cp.has_section('fixed'):
        for name, val in cp['fixed'].items():
            fixed_params[name] = float(val)

    derived_exprs: Dict[str, str] = {}
    if cp.has_section('derived'):
        for name, expr in cp['derived'].items():
            derived_exprs[name] = expr.strip()

    sim_params: Dict[str, object] = {}
    if cp.has_section('simulation'):
        for name, val in cp['simulation'].items():
            val = val.strip()
            try:
                sim_params[name] = int(val)
            except ValueError:
                try:
                    sim_params[name] = float(val)
                except ValueError:
                    sim_params[name] = val

    return ScanConfig(scan_axes, fixed_params, derived_exprs, sim_params)


def resolve_point(scan_point: dict, fixed: dict, derived_exprs: dict) -> dict:
    """
    Build a complete physics parameter dict for one grid point.

    scan_point    : dict of scan-axis name → value (one point on the grid)
    fixed         : dict of fixed-parameter name → value
    derived_exprs : dict of derived-parameter name → expression string

    Expressions may reference scan params, fixed params, or other derived params
    in any order — a multi-pass loop resolves dependencies automatically.
    Only arithmetic (+, -, *, /) and parentheses are permitted (no builtins).
    """
    params: dict = {}
    params.update(fixed)
    params.update(scan_point)

    unresolved = dict(derived_exprs)
    for _ in range(len(unresolved) + 1):   # at most N passes for N expressions
        still_pending = {}
        for name, expr in unresolved.items():
            try:
                params[name] = float(eval(expr, {'__builtins__': {}}, dict(params)))
            except NameError:
                still_pending[name] = expr   # dependency not yet available; retry
            except Exception as e:
                raise ValueError(
                    f"Could not evaluate derived expression '{name} = {expr}': {e}")
        if not still_pending:
            break
        unresolved = still_pending
    else:
        raise ValueError(
            f"Derived expressions could not be resolved (circular dependency or "
            f"undefined name): {list(unresolved.keys())}")

    return params


# ── Gaussian copula fitter ─────────────────────────────────────────────────────

def fit_mvn_corr(X):
    """
    Fit a Gaussian copula correlation matrix to standard-normal columns.

    Parameters
    ----------
    X : np.ndarray, shape (N, K)
        Already standard-normal marginals (probit-transformed CDF outputs).

    Returns
    -------
    R_upper : np.ndarray, shape (K*(K-1)//2,)
        Upper triangle of the sample Pearson correlation matrix.
    """
    K = X.shape[1]
    if K == 1:
        return np.array([])
    R = np.corrcoef(X.T)
    np.fill_diagonal(R, 1.0)
    return R[np.triu_indices(K, k=1)]


# ── Temp config writer ─────────────────────────────────────────────────────────

def write_point_cfg(path, point_params, nEvent, nWorkers, tsv_file):
    with open(path, 'w') as f:
        f.write("# auto-generated per-point config\n")
        for k, v in point_params.items():
            f.write(f"{k} = {v}\n")
        f.write(f"nEvent     = {nEvent}\n")
        f.write(f"nWorkers   = {nWorkers}\n")
        f.write(f"save_tsv   = 1\n")
        f.write(f"tsv_file   = {tsv_file}\n")


# ── Per-observable transform + fit  ───────────────────────────────────────────

def _apply_transforms(X_raw, obs_selection, col_map):
    """
    Apply per-observable pipeline to pre-filtered event data.

    Parameters
    ----------
    X_raw : np.ndarray, shape (N, n_cols)
    obs_selection : list of str
    col_map : dict[str, int]

    Returns
    -------
    tr_data   : np.ndarray, shape (N, n_obs)
    flat_params : np.ndarray, shape (total_obs_params,)
    offsets   : np.ndarray, shape (n_obs+1,)
    """
    offsets  = obs_param_offsets(obs_selection)
    n_params = int(offsets[-1])
    flat_p   = np.empty(n_params)
    tr_cols  = []

    for i, obs_name in enumerate(obs_selection):
        obs_spec = OBSERVABLES[obs_name]
        col      = col_map[obs_spec['col']]
        x_col    = X_raw[:, col].copy()
        y_std, params = fit_observable_col(x_col, obs_spec['pipeline'],
                                           obs_spec['distribution'])
        p_start = int(offsets[i])
        p_end   = int(offsets[i + 1])
        flat_p[p_start:p_end] = params
        tr_cols.append(y_std)

    return np.column_stack(tr_cols), flat_p, offsets


# ── Worker — must be module-level for multiprocessing pickling ─────────────────

_OBS_SELECTION = None
_SAVE_RAW      = False


def _worker(args):
    """
    Run one grid point.

    args = (task_id, grid_indices, point_params, nEvent, nWorkers_inner,
            obs_selection, save_raw)

    Returns
    -------
    (grid_indices, flat_params, R_upper, raw_X, n_discarded, cpp_peak_kb)  on success
    (grid_indices, None, None, None, None, cpp_peak_kb_or_0)              on failure
    """
    (task_id, grid_indices, point_params,
     nEvent, nWorkers_inner, obs_selection, save_raw) = args

    # tempfile.gettempdir() honours $TMPDIR.  Batch systems point that at
    # job-private scratch (HTCondor does so on lxplus), whereas worker-node
    # /tmp is often small and shared between jobs — and each of these TSVs
    # holds nEvent events.  Falls back to /tmp when $TMPDIR is unset.
    _scratch = tempfile.gettempdir()
    temp_cfg = os.path.join(_scratch, f'svj_scan_{task_id}.cfg')
    temp_tsv = os.path.join(_scratch, f'svj_scan_{task_id}.tsv')
    fail     = (grid_indices, None, None, None, None, 0)

    write_point_cfg(temp_cfg, point_params, nEvent, nWorkers_inner, temp_tsv)

    try:
        proc = subprocess.run([BINARY, temp_cfg], capture_output=True, text=True)
        # Capture C++ peak RSS immediately after binary exits (RUSAGE_CHILDREN
        # accumulates across all subprocess.run calls in this worker process).
        cpp_peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        fail = (grid_indices, None, None, None, None, cpp_peak_kb)

        if proc.returncode != 0:
            return fail

        if not os.path.exists(temp_tsv):
            return fail

        try:
            data, col_map = load_tsv(temp_tsv)
        finally:
            try:
                os.remove(temp_tsv)
            except FileNotFoundError:
                pass

        if data.ndim != 2:
            return fail

        mask, n_disc = event_valid_mask(data, obs_selection, col_map)
        X_valid = data[mask]

        if len(X_valid) < 20:
            return fail

        try:
            tr_data, flat_params, offsets = _apply_transforms(
                X_valid, obs_selection, col_map)
        except Exception:
            return fail

        try:
            R_upper = fit_mvn_corr(tr_data)
        except Exception:
            return fail

        raw_X = (X_valid[:, [col_map[OBSERVABLES[n]['col']] for n in obs_selection]]
                 if save_raw else None)

        return (grid_indices, flat_params, R_upper, raw_X, n_disc, cpp_peak_kb)

    except Exception:
        return fail
    finally:
        try:
            os.remove(temp_cfg)
        except FileNotFoundError:
            pass


# ── NPZ helpers ────────────────────────────────────────────────────────────────

def _save(out_file, scan_cfg: ScanConfig,
          param_flat, obs_offsets, obs_names,
          scan_params_arr, scan_param_names, n_obs):
    axis_names = list(scan_cfg.scan_axes.keys())
    corr_start = int(obs_offsets[-1])

    kwargs = dict(
        axis_names       = np.array(axis_names, dtype=object),
        param_flat       = param_flat,
        param_offsets    = obs_offsets,
        corr_start       = np.array(corr_start, dtype=int),
        obs_names        = np.array(obs_names,  dtype=object),
        scan_params      = scan_params_arr,
        scan_param_names = np.array(scan_param_names, dtype=object),
    )
    for name, vals in scan_cfg.scan_axes.items():
        kwargs[f'{name}_vals'] = vals
    np.savez(out_file, **kwargs)


def _save_metadata(out_dir, scan_cfg: ScanConfig,
                   obs_selection, obs_offsets, n_corr):
    meta = {
        'obs_selection': list(obs_selection),
        'param_offsets': [int(x) for x in obs_offsets],
        'n_corr':        n_corr,
        'scan_axes':     {name: list(map(float, vals))
                          for name, vals in scan_cfg.scan_axes.items()},
        'fixed_params':  scan_cfg.fixed_params,
        'derived_exprs': scan_cfg.derived_exprs,
    }
    with open(out_dir / 'svj_scan_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def _save_raw_npz(out_file, raw_flat, raw_grid_flat):
    np.savez(out_file,
             raw_flat      = raw_flat,
             raw_grid_flat = raw_grid_flat)


# ── Merge helper ───────────────────────────────────────────────────────────────

def _merge(out_dir, n_jobs, keep_shards=False):
    files = [out_dir / f'svj_scan_{i}.npz' for i in range(n_jobs)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"Error: missing files: {missing}")
        sys.exit(1)

    d0          = np.load(files[0], allow_pickle=True)
    axis_names  = list(d0['axis_names'])
    param_flat  = np.array(d0['param_flat'])
    scan_params = np.array(d0['scan_params'])

    for f in files[1:]:
        d    = np.load(f, allow_pickle=True)
        pf   = np.array(d['param_flat'])
        sp   = np.array(d['scan_params'])
        mask = np.all(np.isfinite(pf), axis=-1)
        param_flat[mask]  = pf[mask]
        scan_params[mask] = sp[mask]

    out_file = out_dir / 'svj_scan.npz'
    kwargs   = {
        'axis_names':       d0['axis_names'],
        'param_flat':       param_flat,
        'param_offsets':    d0['param_offsets'],
        'corr_start':       d0['corr_start'],
        'obs_names':        d0['obs_names'],
        'scan_params':      scan_params,
        'scan_param_names': d0['scan_param_names'],
    }
    for name in axis_names:
        kwargs[f'{name}_vals'] = d0[f'{name}_vals']
    np.savez(out_file, **kwargs)
    print(f"Saved → {out_file}")

    if keep_shards:
        print(f"Shard files kept (--keep-shards).")
    else:
        for f in files:
            f.unlink(missing_ok=True)
        print(f"Removed {len(files)} shard file(s).")

    n_done  = int(np.sum(np.all(np.isfinite(param_flat), axis=-1)))
    n_total = param_flat[..., 0].size
    print(f"Merged {n_jobs} files → {out_file}")
    print(f"  {n_done} / {n_total} points complete")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', nargs='?',
                        default=str(_HERE / 'scan_regression.cfg'))
    parser.add_argument('--obs', type=str, default=None,
                        help='Comma-separated observable names (default: DEFAULT_SCAN)')
    parser.add_argument('--job-index', type=int, default=0,
                        help='Index of this SLURM array job slice (0-based)')
    parser.add_argument('--n-jobs',    type=int, default=1,
                        help='Total number of parallel job slices')
    parser.add_argument('--merge', action='store_true',
                        help='Merge per-job NPZs into svj_scan.npz and exit')
    parser.add_argument('--keep-shards', action='store_true',
                        help='Keep per-job shard NPZs after merging (default: delete on success)')
    parser.add_argument('--save-raw', action='store_true',
                        help='Save pre-transform event data alongside the NPZ')
    args = parser.parse_args()

    job_index = args.job_index
    n_jobs    = args.n_jobs
    save_raw  = args.save_raw

    out_dir = Path('simulated') / 'svj'

    if args.merge:
        out_dir.mkdir(parents=True, exist_ok=True)
        _merge(out_dir, n_jobs, keep_shards=args.keep_shards)
        return

    scan_cfg_path = args.cfg
    if not os.path.exists(scan_cfg_path):
        print(f"Error: config file '{scan_cfg_path}' not found.")
        sys.exit(1)
    if not os.path.exists(BINARY):
        print(f"Error: binary '{BINARY}' not found. Run 'make svj_regression' first.")
        sys.exit(1)

    scan_cfg = read_scan_cfg(scan_cfg_path)

    if not scan_cfg.scan_axes:
        print("Error: [scan] section is empty — nothing to scan.")
        sys.exit(1)

    # ── Output paths ────────────────────────────────────────────────────────────
    out_dir_str = scan_cfg.sim_params.get('output_dir', 'simulated')
    out_dir     = Path(str(out_dir_str)) / 'svj'
    fname       = f'svj_scan_{job_index}.npz' if n_jobs > 1 else 'svj_scan.npz'
    out_file    = out_dir / fname
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Simulation settings ──────────────────────────────────────────────────────
    sp_   = scan_cfg.sim_params
    nEvent       = int(sp_.get('nEvent',           50000))
    nWorkers_inn = int(sp_.get('nWorkers',             1))
    n_outer      = int(sp_.get('n_outer_workers',     32))
    chk_every    = int(sp_.get('checkpoint_every',   200))

    # ── Observable selection ─────────────────────────────────────────────────────
    if args.obs is not None:
        obs_selection = [s.strip() for s in args.obs.split(',')]
    else:
        obs_selection = DEFAULT_SCAN
    validate_scan_selection(obs_selection)
    n_obs       = len(obs_selection)
    n_corr      = n_obs * (n_obs - 1) // 2
    obs_offsets = obs_param_offsets(obs_selection)
    total_params = int(obs_offsets[-1]) + n_corr

    # ── Grid ─────────────────────────────────────────────────────────────────────
    axis_names = list(scan_cfg.scan_axes.keys())
    axis_vals  = [scan_cfg.scan_axes[n] for n in axis_names]
    axis_sizes = [len(v) for v in axis_vals]
    total      = 1
    for s in axis_sizes:
        total *= s

    # All resolved physics params stored per point (scan axes + derived)
    scan_param_names = axis_names + list(scan_cfg.derived_exprs.keys())
    n_scan_phys      = len(scan_param_names)

    # ── Allocate result arrays ───────────────────────────────────────────────────
    grid_shape  = tuple(axis_sizes)
    param_flat  = np.full(grid_shape + (total_params,),  np.nan)
    scan_p_arr  = np.full(grid_shape + (n_scan_phys,),   np.nan)

    # ── Resumption ───────────────────────────────────────────────────────────────
    n_preloaded = 0
    if out_file.exists():
        try:
            existing = np.load(out_file, allow_pickle=True)
            pf_loaded = existing['param_flat']
            if pf_loaded.shape == param_flat.shape:
                param_flat[...] = pf_loaded
                scan_p_arr[...]  = existing['scan_params']
                n_preloaded = int(np.sum(np.all(np.isfinite(param_flat), axis=-1)))
                print(f"Resuming: {n_preloaded} points done, loaded from {out_file}")
            else:
                print(f"Warning: existing NPZ shape {pf_loaded.shape} != "
                      f"current {param_flat.shape}. Starting fresh.")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}); starting fresh.")

    # ── Pre-fill scan_p_arr for all points ──────────────────────────────────────
    all_grid_indices = list(itertools.product(*[range(s) for s in axis_sizes]))
    for gidx in all_grid_indices:
        scan_point = {axis_names[k]: axis_vals[k][gidx[k]]
                      for k in range(len(axis_names))}
        pp = resolve_point(scan_point, scan_cfg.fixed_params, scan_cfg.derived_exprs)
        scan_p_arr[gidx] = [pp[n] for n in scan_param_names]

    # ── Build task list for this job's slice ─────────────────────────────────────
    tasks = []
    for flat_idx, gidx in enumerate(all_grid_indices):
        if flat_idx % n_jobs != job_index:
            continue
        if np.all(np.isfinite(param_flat[gidx])):
            continue
        scan_point  = {axis_names[k]: axis_vals[k][gidx[k]]
                       for k in range(len(axis_names))}
        point_params = resolve_point(scan_point, scan_cfg.fixed_params,
                                     scan_cfg.derived_exprs)
        # flat_idx is unique across all grid points — use it directly for /tmp filenames
        task_id = flat_idx
        tasks.append((task_id, gidx, point_params,
                      nEvent, nWorkers_inn, obs_selection, save_raw))

    n_todo  = len(tasks)
    job_str = f"job {job_index}/{n_jobs-1}  " if n_jobs > 1 else ""
    grid_str = '×'.join(str(s) for s in axis_sizes)
    print(f"SVJ scan  {job_str}({n_outer} outer × {nWorkers_inn} C++ threads)")
    print(f"  Axes ({len(axis_names)}): {', '.join(f'{n}({s})' for n, s in zip(axis_names, axis_sizes))}")
    print(f"  Grid:       {grid_str} = {total} total  ({total // n_jobs} this job)")
    print(f"  Observables ({n_obs}): {obs_selection}")
    print(f"  Total params/point: {total_params}  "
          f"(obs: {obs_offsets[-1]}, corr: {n_corr})")
    print(f"  To do:      {n_todo}  (already done: {n_preloaded})")
    print(f"  Events/pt:  {nEvent}")
    if save_raw:
        print("  --save-raw: raw events will be written alongside NPZ")
    print(f"  Output:     {out_file}\n")

    if n_todo == 0:
        print("Nothing left to do.")
        _save_metadata(out_dir, scan_cfg, obs_selection, obs_offsets, n_corr)
        return

    corr_start = int(obs_offsets[-1])

    t0     = time.time()
    done   = 0
    failed = 0
    width  = len(str(n_todo))

    raw_flat_list      = [] if save_raw else None
    raw_grid_flat_list = [] if save_raw else None
    cpp_rss_kb_list: list = []

    with ProcessPoolExecutor(max_workers=n_outer) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}

        for fut in as_completed(futs):
            try:
                result = fut.result()
                # Unpack inside the guard: a malformed result tuple must be
                # counted as one failed point, never kill a multi-hour scan.
                gidx, flat_p, R_upper, raw_X, n_disc, cpp_rss_kb = result
            except Exception as e:
                print(f"  WARNING: worker exception: {e}", flush=True)
                failed += 1
                done   += 1
                continue

            done += 1
            ok    = flat_p is not None

            if ok:
                param_flat[gidx + (slice(None, corr_start),)]  = flat_p
                param_flat[gidx + (slice(corr_start, None),)]  = R_upper

                if save_raw and raw_X is not None:
                    n_ev = len(raw_X)
                    raw_flat_list.append(raw_X)
                    raw_grid_flat_list.append(
                        np.tile(list(gidx), (n_ev, 1)))
                if cpp_rss_kb:
                    cpp_rss_kb_list.append(cpp_rss_kb)
            else:
                failed += 1
                if cpp_rss_kb:
                    cpp_rss_kb_list.append(cpp_rss_kb)

            elapsed = time.time() - t0
            avg     = elapsed / done
            h, rem  = divmod(int(avg * (n_todo - done)), 3600)
            mm, s   = divmod(rem, 60)
            if ok and n_disc is not None:
                total_disc = int(n_disc.sum())
                per_obs = ", ".join(
                    f"{name}:{int(n)}"
                    for name, n in zip(obs_selection, n_disc)
                    if n > 0
                )
                disc_str = f"  disc={total_disc}" + (f" [{per_obs}]" if per_obs else "")
            else:
                disc_str = ""
            print(f"[{done:{width}}/{n_todo}]  {gidx}"
                  f"  {'ok  ' if ok else 'FAIL'}"
                  f"  ETA {h:02d}:{mm:02d}:{s:02d}"
                  f"{disc_str}", flush=True)

            if done % chk_every == 0:
                _save(out_file, scan_cfg, param_flat, obs_offsets,
                      obs_selection, scan_p_arr, scan_param_names, n_obs)
                print(f"  [checkpoint @ {done}]", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nDone. {done} points in {elapsed_total / 60:.1f} min; {failed} failed.")

    _save(out_file, scan_cfg, param_flat, obs_offsets,
          obs_selection, scan_p_arr, scan_param_names, n_obs)
    print(f"Saved → {out_file}")
    print(f"  param_flat: {param_flat.shape}")
    n_nan = int(np.sum(np.any(np.isnan(param_flat), axis=-1)))
    if n_nan:
        print(f"  WARNING: {n_nan} grid points have NaN.")

    _save_metadata(out_dir, scan_cfg, obs_selection, obs_offsets, n_corr)

    if save_raw and raw_flat_list:
        raw_file = out_dir / (out_file.stem + '_raw.npz')
        _save_raw_npz(raw_file,
                      np.vstack(raw_flat_list),
                      np.vstack(raw_grid_flat_list))
        print(f"Raw events → {raw_file}")

    # ── Resource usage report ────────────────────────────────────────────────────
    _self_kb   = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _python_mb = _self_kb / 1024
    if cpp_rss_kb_list:
        _cpp_mb     = max(cpp_rss_kb_list) / 1024
        _est_mb     = _cpp_mb * n_outer + _python_mb
        _suggest_gb = max(1, int(_est_mb / 1024 * 1.5) + 1)
    else:
        _cpp_mb     = 0.0
        _est_mb     = _python_mb
        _suggest_gb = max(1, int(_est_mb / 1024 * 1.5) + 1)

    print(f"\n=== Resource usage (scan) ===")
    print(f"  Python main process peak RSS : {_python_mb:7.0f} MB")
    if cpp_rss_kb_list:
        print(f"  C++ binary peak RSS (max)    : {_cpp_mb:7.0f} MB"
              f"  ({nWorkers_inn} thread(s)/point, {len(cpp_rss_kb_list)} measurements)")
        print(f"  n_outer_workers (concurrent) : {n_outer}")
        print(f"  Estimated concurrent peak    : {_est_mb:7.0f} MB"
              f"  ({n_outer} × {_cpp_mb:.0f} + {_python_mb:.0f})")
    print(f"  → suggested --mem            : {_suggest_gb}G"
          f"  (concurrent estimate × 1.5)")
    print(f"=============================")


if __name__ == '__main__':
    main()

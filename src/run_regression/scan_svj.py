#!/usr/bin/env python3
"""
scan_svj.py
===========
Scan (mZ', mRho, rinv, alphaD) parameter space, fitting a per-observable
transform pipeline + Multivariate-t copula at each grid point.

"Scan" here means simulate → fit → store interpolation parameters.
The resulting NPZ is used by helpers.py to interpolate the fitted distribution
to arbitrary parameter points — hence "SVJ interpolation" rather than
"gennorm regression".

At each grid point the pipeline is:
  1. Run ./svj_regression (save_tsv=1, full_obs=1).
  2. Range-check each event against the pipeline requirements for all
     selected observables; discard events failing any check.
  3. For each observable: apply invertible transforms, fit the chosen
     distribution, map to standard-normal (probit of CDF).
  4. Fit a zero-mean Multivariate-t (MVT) correlation matrix R and
     degrees-of-freedom ν to the standard-normal columns via EM.
  5. Store: per-observable transform+dist params in a flat vector, upper
     triangle of R, and ν.

Observable selection comes from src/observables.py DEFAULT_SCAN.  Override
via --obs on the command line (comma-separated list of observable names).

NPZ format (svj_scan.npz):
  param_flat     (N_mZ, N_mRho, N_rinv, N_alphaD, total_params)
  param_offsets  (n_obs + 1,)           index into param_flat for each obs
  corr_start     scalar int             param_flat[..., corr_start:corr_start+n_corr] = R triu
  nu_idx         scalar int             param_flat[..., nu_idx] = ν
  obs_names      (n_obs,)
  scan_params    (N_mZ, N_mRho, N_rinv, N_alphaD, 6)
  mZ_vals, mRho_vals, rinv_vals, alphaD_vals

Optional --save-raw flag writes svj_scan_raw.npz alongside:
  raw_flat       (total_valid_events, n_obs)   pre-transform observable values
  raw_grid_flat  (total_valid_events, 4)        grid indices [i,j,k,l]

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
import time
import json
import itertools
import numpy as np
import scipy.special as sp
import scipy.stats as st
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

_HERE  = Path(__file__).resolve().parent
_SRC   = _HERE.parent
BINARY = str(_SRC / 'generate_events' / 'svj_regression')

# Import observable registry
sys.path.insert(0, str(_SRC))
from observables import (
    OBSERVABLES, DEFAULT_SCAN,
    n_fitted_params, param_offsets as obs_param_offsets,
    event_valid_mask, fit_observable_col, validate_scan_selection,
)

MRHO_MPION_RATIO  = 8.0 / 15.5
MRHO_LAMBDA_RATIO = 5.0 / 15.5


# ── MVT EM fitter ──────────────────────────────────────────────────────────────

def _mvt_loglik(X, R_inv, log_det_R, nu):
    N, K = X.shape
    deltas = np.sum((X @ R_inv) * X, axis=1)
    ll = (N * (sp.gammaln((nu + K) / 2) - sp.gammaln(nu / 2)
               - K / 2 * np.log(np.pi * nu) - 0.5 * log_det_R)
          - (nu + K) / 2 * np.sum(np.log(1.0 + deltas / nu)))
    return ll


def fit_mvt_em(X, max_iter=200, tol=1e-6):
    """
    Fit zero-mean MVT(0, R, ν) via EM + golden-section search for ν.

    Parameters
    ----------
    X : np.ndarray, shape (N, K)
        Already standard-normal marginals (probit-transformed CDF outputs).

    Returns
    -------
    R_upper : np.ndarray, shape (K*(K-1)//2,)
        Upper-triangle of the K×K correlation matrix.
    nu : float
        Fitted degrees of freedom.
    """
    N, K = X.shape
    R = np.corrcoef(X.T).copy()
    np.fill_diagonal(R, 1.0)
    nu = 5.0

    for _ in range(max_iter):
        # E-step
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            break
        deltas = np.sum((X @ R_inv) * X, axis=1)   # (N,)
        w = (nu + K) / (nu + deltas)                 # E-step weights

        # M-step for R (weighted outer products, then normalize to corr matrix)
        Sigma = (X.T * w) @ X / N
        D = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
        R_new = Sigma / np.outer(D, D)
        np.fill_diagonal(R_new, 1.0)

        # Golden-section for ν using new R_new
        try:
            R_new_inv = np.linalg.inv(R_new)
            log_det_R  = np.log(np.maximum(np.linalg.det(R_new), 1e-300))
        except np.linalg.LinAlgError:
            R_new = R
            break

        deltas_new = np.sum((X @ R_new_inv) * X, axis=1)

        def neg_ll(log_nu):
            nu_ = np.exp(log_nu)
            ll  = (N * (sp.gammaln((nu_ + K) / 2) - sp.gammaln(nu_ / 2)
                        - K / 2 * np.log(np.pi * nu_) - 0.5 * log_det_R)
                   - (nu_ + K) / 2 * np.sum(np.log(1.0 + deltas_new / nu_)))
            return -ll

        a, b = np.log(1.01), np.log(500.0)
        gr = (np.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        for _ in range(120):
            if abs(b - a) < 1e-9:
                break
            if neg_ll(c) < neg_ll(d):
                b = d
            else:
                a = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr
        nu_new = np.exp((a + b) / 2)

        R_diff = float(np.max(np.abs(R_new - R)))
        nu_diff = abs(nu_new - nu) / (nu + 1e-8)
        R = R_new
        nu = nu_new

        if R_diff < tol and nu_diff < tol:
            break

    idx = np.triu_indices(K, k=1)
    return R[idx], float(nu)


# ── Config reader ──────────────────────────────────────────────────────────────

def read_cfg(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, val = line.partition('=')
            cfg[key.strip()] = val.split('#')[0].strip()
    return cfg

def cfg_float(cfg, key, default): return float(cfg.get(key, default))
def cfg_int  (cfg, key, default): return int  (cfg.get(key, default))
def cfg_str  (cfg, key, default): return       cfg.get(key, str(default))


# ── Temp config writer ─────────────────────────────────────────────────────────

def write_point_cfg(path, params):
    with open(path, 'w') as f:
        f.write("# auto-generated per-point config\n")
        for k, v in params.items():
            f.write(f"{k} = {v}\n")


# ── Per-observable transform + fit  ───────────────────────────────────────────

def _apply_transforms(X_raw, obs_selection):
    """
    Apply per-observable pipeline to pre-filtered event data.

    Parameters
    ----------
    X_raw : np.ndarray, shape (N, 22)
        Raw TSV data, already filtered to valid rows.
    obs_selection : list of str

    Returns
    -------
    tr_data   : np.ndarray, shape (N, n_obs)  — standard-normal marginals
    flat_params : np.ndarray, shape (total_obs_params,)
    offsets   : np.ndarray, shape (n_obs+1,)
    """
    offsets  = obs_param_offsets(obs_selection)
    n_params = int(offsets[-1])
    flat_p   = np.empty(n_params)
    tr_cols  = []

    for i, obs_name in enumerate(obs_selection):
        obs_spec = OBSERVABLES[obs_name]
        col      = obs_spec['col']
        pipeline = obs_spec['pipeline']
        dist     = obs_spec['distribution']

        x_col    = X_raw[:, col].copy()

        # Thrust flip is in pipeline as affine_flip; negative values from that
        # are handled by boxcox.  No extra pre-processing needed here.
        y_std, params = fit_observable_col(x_col, pipeline, dist)
        p_start = int(offsets[i])
        p_end   = int(offsets[i + 1])
        flat_p[p_start:p_end] = params
        tr_cols.append(y_std)

    tr_data = np.column_stack(tr_cols)   # (N, n_obs)
    return tr_data, flat_p, offsets


# ── Worker — must be module-level for multiprocessing pickling ─────────────────

# Module-level globals set by main() before spawning workers
_OBS_SELECTION = None
_SAVE_RAW      = False


def _worker(args):
    """
    Run one grid point.

    Returns
    -------
    (i, j, k, l, flat_params, R_upper, nu, raw_X, n_discarded_by_obs)  on success
    (i, j, k, l, None, None, None, None, None)                          on failure
    """
    (task_id, i, j, k, l,
     mZ, mRho, mPi, LambdaQCD, rinv, alphaD,
     mq, Brl, jetR, nEvent, nWorkers_inner,
     obs_selection, save_raw) = args

    temp_cfg = f'/tmp/svj_scan_{task_id}.cfg'
    temp_tsv = f'/tmp/svj_scan_{task_id}.tsv'
    fail = (i, j, k, l, None, None, None, None, None)

    write_point_cfg(temp_cfg, {
        'mZ':         mZ,         'mq':       mq,
        'mPi':        mPi,        'mRho':     mRho,
        'rinv':       rinv,       'rinv2':    rinv,
        'Brl':        Brl,        'alphaD':   alphaD,
        'nEvent':     nEvent,     'jetR':     jetR,
        'LambdaDQCD': LambdaQCD,  'nWorkers': nWorkers_inner,
        'save_tsv':   1,          'full_obs': 1,
        'tsv_file':   temp_tsv,
    })

    try:
        proc = subprocess.run([BINARY, temp_cfg], capture_output=True, text=True)
        if proc.returncode != 0:
            return fail

        if not os.path.exists(temp_tsv):
            return fail

        try:
            data = np.loadtxt(temp_tsv, comments='#')
        finally:
            try:
                os.remove(temp_tsv)
            except FileNotFoundError:
                pass

        if data.ndim != 2 or data.shape[1] != 22:
            return fail

        # Range check: discard events failing any observable's pipeline requirements
        mask, n_disc = event_valid_mask(data, obs_selection)
        X_valid = data[mask]

        if len(X_valid) < 20:
            return fail

        try:
            tr_data, flat_params, offsets = _apply_transforms(X_valid, obs_selection)
        except Exception:
            return fail

        try:
            R_upper, nu = fit_mvt_em(tr_data)
        except Exception:
            return fail

        raw_X = X_valid[:, [OBSERVABLES[n]['col'] for n in obs_selection]] if save_raw else None

        return (i, j, k, l, flat_params, R_upper, nu,
                raw_X, n_disc)

    except Exception:
        return fail
    finally:
        try:
            os.remove(temp_cfg)
        except FileNotFoundError:
            pass


# ── NPZ helpers ────────────────────────────────────────────────────────────────

def _build_param_flat_shape(mZ_n, mRho_n, rinv_n, alphaD_n,
                            obs_selection, obs_offsets):
    n_obs    = len(obs_selection)
    n_corr   = n_obs * (n_obs - 1) // 2
    n_params = int(obs_offsets[-1]) + n_corr + 1   # obs_params + corr_upper + nu
    return (mZ_n, mRho_n, rinv_n, alphaD_n, n_params)


def _save(out_file, mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
          param_flat, obs_offsets, obs_names,
          scan_params, scan_param_names, n_obs):
    n_corr     = n_obs * (n_obs - 1) // 2
    corr_start = int(obs_offsets[-1])
    nu_idx     = corr_start + n_corr
    np.savez(
        out_file,
        mZ_vals          = mZ_vals,
        mRho_vals        = mRho_vals,
        rinv_vals        = rinv_vals,
        alphaD_vals      = alphaD_vals,
        param_flat       = param_flat,
        param_offsets    = obs_offsets,
        corr_start       = np.array(corr_start, dtype=int),
        nu_idx           = np.array(nu_idx,     dtype=int),
        obs_names        = np.array(obs_names,  dtype=object),
        scan_params      = scan_params,
        scan_param_names = scan_param_names,
    )


def _save_metadata(out_dir, obs_selection, obs_offsets, n_corr, scan_cfg):
    meta = {
        'obs_selection': list(obs_selection),
        'param_offsets': [int(x) for x in obs_offsets],
        'n_corr':        n_corr,
        'nu_at_index':   int(obs_offsets[-1]) + n_corr,
        'scan_cfg':      scan_cfg,
    }
    with open(out_dir / 'svj_scan_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


def _save_raw_npz(out_file, raw_flat, raw_grid_flat):
    np.savez(out_file,
             raw_flat      = raw_flat,
             raw_grid_flat = raw_grid_flat)


# ── Merge helper ───────────────────────────────────────────────────────────────

def _merge(out_dir, n_jobs):
    files = [out_dir / f'svj_scan_{i}.npz' for i in range(n_jobs)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"Error: missing files: {missing}")
        sys.exit(1)

    d0         = np.load(files[0], allow_pickle=True)
    param_flat = np.array(d0['param_flat'])
    scan_params = np.array(d0['scan_params'])

    for f in files[1:]:
        d   = np.load(f, allow_pickle=True)
        pf  = np.array(d['param_flat'])
        sp  = np.array(d['scan_params'])
        mask = np.all(np.isfinite(pf), axis=-1)
        param_flat[mask]  = pf[mask]
        sp_mask = np.all(np.isfinite(sp), axis=-1)
        scan_params[sp_mask] = sp[sp_mask]

    out_file = out_dir / 'svj_scan.npz'
    np.savez(
        out_file,
        mZ_vals          = d0['mZ_vals'],
        mRho_vals        = d0['mRho_vals'],
        rinv_vals        = d0['rinv_vals'],
        alphaD_vals      = d0['alphaD_vals'],
        param_flat       = param_flat,
        param_offsets    = d0['param_offsets'],
        corr_start       = d0['corr_start'],
        nu_idx           = d0['nu_idx'],
        obs_names        = d0['obs_names'],
        scan_params      = scan_params,
        scan_param_names = d0['scan_param_names'],
    )
    n_done  = int(np.sum(np.all(np.isfinite(param_flat), axis=-1)))
    n_total = param_flat[..., 0].size
    print(f"Merged {n_jobs} files → {out_file}")
    print(f"  {n_done} / {n_total} points complete")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', nargs='?', default=str(_HERE / 'scan_regression.cfg'))
    parser.add_argument('--obs', type=str, default=None,
                        help='Comma-separated observable names (default: DEFAULT_SCAN)')
    parser.add_argument('--job-index', type=int, default=0,
                        help='Index of this SLURM array job slice (0-based)')
    parser.add_argument('--n-jobs',    type=int, default=1,
                        help='Total number of parallel job slices')
    parser.add_argument('--merge', action='store_true',
                        help='Merge per-job NPZs into svj_scan.npz and exit')
    parser.add_argument('--save-raw', action='store_true',
                        help='Save pre-transform event data alongside the NPZ')
    args = parser.parse_args()

    job_index = args.job_index
    n_jobs    = args.n_jobs
    save_raw  = args.save_raw

    out_dir = Path('simulated') / 'svj'

    if args.merge:
        out_dir.mkdir(parents=True, exist_ok=True)
        _merge(out_dir, n_jobs)
        return

    scan_cfg_path = args.cfg

    if not os.path.exists(scan_cfg_path):
        print(f"Error: config file '{scan_cfg_path}' not found.")
        sys.exit(1)
    if not os.path.exists(BINARY):
        print(f"Error: binary '{BINARY}' not found. Run 'make svj_regression' first.")
        sys.exit(1)

    cfg = read_cfg(scan_cfg_path)

    # Grid axes
    mZ_min     = cfg_float(cfg, 'mZ_min',      500.0)
    mZ_max     = cfg_float(cfg, 'mZ_max',     4000.0)
    mZ_n       = cfg_int  (cfg, 'mZ_n',             8)
    mRho_min   = cfg_float(cfg, 'mRho_min',    10.0)
    mRho_max   = cfg_float(cfg, 'mRho_max',    30.0)
    mRho_n     = cfg_int  (cfg, 'mRho_n',           8)
    rinv_min   = cfg_float(cfg, 'rinv_min',    0.05)
    rinv_max   = cfg_float(cfg, 'rinv_max',    0.70)
    rinv_n     = cfg_int  (cfg, 'rinv_n',           8)
    alphaD_min = cfg_float(cfg, 'alphaD_min',  0.10)
    alphaD_max = cfg_float(cfg, 'alphaD_max',  0.80)
    alphaD_n   = cfg_int  (cfg, 'alphaD_n',         8)

    mq       = cfg_float(cfg, 'mq',              4.0)
    Brl      = cfg_float(cfg, 'Brl',             0.3)
    jetR     = cfg_float(cfg, 'jetR',            1.0)
    nEvent   = cfg_int  (cfg, 'nEvent',         2000)
    nWorkers = cfg_int  (cfg, 'nWorkers',          2)
    n_outer  = cfg_int  (cfg, 'n_outer_workers',  32)
    chk_every= cfg_int  (cfg, 'checkpoint_every', 200)

    out_dir  = Path(cfg_str(cfg, 'output_dir', 'simulated')) / 'svj'
    fname    = f'svj_scan_{job_index}.npz' if n_jobs > 1 else 'svj_scan.npz'
    out_file = out_dir / fname
    out_dir.mkdir(parents=True, exist_ok=True)

    # Observable selection
    if args.obs is not None:
        obs_selection = [s.strip() for s in args.obs.split(',')]
    else:
        obs_selection = DEFAULT_SCAN
    validate_scan_selection(obs_selection)
    n_obs    = len(obs_selection)
    n_corr   = n_obs * (n_obs - 1) // 2
    obs_offsets = obs_param_offsets(obs_selection)
    total_params = int(obs_offsets[-1]) + n_corr + 1   # obs params + corr + nu

    mZ_vals     = np.linspace(mZ_min,     mZ_max,     mZ_n)
    mRho_vals   = np.linspace(mRho_min,   mRho_max,   mRho_n)
    rinv_vals   = np.linspace(rinv_min,   rinv_max,   rinv_n)
    alphaD_vals = np.linspace(alphaD_min, alphaD_max, alphaD_n)
    total       = mZ_n * mRho_n * rinv_n * alphaD_n
    SCAN_PARAM_NAMES = np.array(
        ['mZ', 'mRho', 'mPi', 'LambdaQCD', 'rinv', 'alphaD'], dtype=object)

    # Allocate result arrays
    param_flat  = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, total_params), np.nan)
    scan_params = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, 6),             np.nan)

    # Resumption: load existing results and skip already-finished points
    n_preloaded = 0
    if out_file.exists():
        try:
            existing = np.load(out_file, allow_pickle=True)
            pf_loaded = existing['param_flat']
            # Shape must match: total_params may differ if obs_selection changed
            if pf_loaded.shape[-1] == total_params:
                param_flat[...]  = pf_loaded
                scan_params[...] = existing['scan_params']
                n_preloaded = int(np.sum(np.all(np.isfinite(param_flat), axis=-1)))
                print(f"Resuming: {n_preloaded} points done, loaded from {out_file}")
            else:
                print(f"Warning: existing NPZ has {pf_loaded.shape[-1]} params but "
                      f"current selection needs {total_params}. Starting fresh.")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}); starting fresh.")

    # Fill scan_params for all points
    all_points = list(itertools.product(
        enumerate(mZ_vals), enumerate(mRho_vals),
        enumerate(rinv_vals), enumerate(alphaD_vals)))

    for (i, mZ), (j, mRho), (k, rinv), (l, alphaD) in all_points:
        scan_params[i, j, k, l] = [mZ, mRho,
                                    mRho * MRHO_MPION_RATIO,
                                    mRho * MRHO_LAMBDA_RATIO,
                                    rinv, alphaD]

    # Build task list for this job's slice
    tasks = []
    for flat_idx, ((i, mZ), (j, mRho), (k, rinv), (l, alphaD)) in enumerate(all_points):
        if flat_idx % n_jobs != job_index:
            continue
        if np.all(np.isfinite(param_flat[i, j, k, l])):
            continue
        mPi       = scan_params[i, j, k, l, 2]
        LambdaQCD = scan_params[i, j, k, l, 3]
        task_id   = (i * mRho_n * rinv_n * alphaD_n
                     + j * rinv_n * alphaD_n
                     + k * alphaD_n + l)
        tasks.append((task_id, i, j, k, l,
                      mZ, mRho, mPi, LambdaQCD, rinv, alphaD,
                      mq, Brl, jetR, nEvent, nWorkers,
                      obs_selection, save_raw))

    n_todo  = len(tasks)
    job_str = f"job {job_index}/{n_jobs-1}  " if n_jobs > 1 else ""
    print(f"SVJ scan  {job_str}({n_outer} outer × {nWorkers} C++ threads)")
    print(f"  Grid:       {mZ_n}×{mRho_n}×{rinv_n}×{alphaD_n} = {total} total  "
          f"({total // n_jobs} this job)")
    print(f"  Observables ({n_obs}): {obs_selection}")
    print(f"  Total params/point: {total_params}  "
          f"(obs: {obs_offsets[-1]}, corr: {n_corr}, nu: 1)")
    print(f"  To do:      {n_todo}  (already done: {n_preloaded})")
    print(f"  Events/pt:  {nEvent}")
    if save_raw:
        print(f"  --save-raw: raw events will be written alongside NPZ")
    print(f"  Output:     {out_file}\n")

    if n_todo == 0:
        print("Nothing left to do.")
        _save_metadata(out_dir, obs_selection, obs_offsets, n_corr,
                       {k: v for k, v in cfg.items()})
        return

    corr_start = int(obs_offsets[-1])

    t0     = time.time()
    done   = 0
    failed = 0
    width  = len(str(n_todo))

    raw_flat_list      = [] if save_raw else None
    raw_grid_flat_list = [] if save_raw else None

    with ProcessPoolExecutor(max_workers=n_outer) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}

        for fut in as_completed(futs):
            try:
                result = fut.result()
            except Exception as e:
                print(f"  WARNING: worker exception: {e}", flush=True)
                failed += 1
                done   += 1
                continue

            i, j, k, l, flat_p, R_upper, nu, raw_X, n_disc = result
            done += 1
            ok = flat_p is not None

            if ok:
                param_flat[i, j, k, l, :corr_start]     = flat_p
                param_flat[i, j, k, l, corr_start:-1]   = R_upper
                param_flat[i, j, k, l, -1]              = nu

                if save_raw and raw_X is not None:
                    n_ev = len(raw_X)
                    raw_flat_list.append(raw_X)
                    raw_grid_flat_list.append(
                        np.tile([i, j, k, l], (n_ev, 1)))
            else:
                failed += 1

            elapsed = time.time() - t0
            avg     = elapsed / done
            h, rem  = divmod(int(avg * (n_todo - done)), 3600)
            mm, s   = divmod(rem, 60)
            disc_str = (f"  disc={int(n_disc.sum())}" if ok and n_disc is not None
                        else "")
            print(f"[{done:{width}}/{n_todo}]  ({i},{j},{k},{l})"
                  f"  {'ok  ' if ok else 'FAIL'}"
                  f"  ETA {h:02d}:{mm:02d}:{s:02d}"
                  f"{disc_str}", flush=True)

            if done % chk_every == 0:
                _save(out_file, mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
                      param_flat, obs_offsets, obs_selection,
                      scan_params, SCAN_PARAM_NAMES, n_obs)
                print(f"  [checkpoint @ {done}]", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nDone. {done} points in {elapsed_total / 60:.1f} min; {failed} failed.")

    _save(out_file, mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
          param_flat, obs_offsets, obs_selection,
          scan_params, SCAN_PARAM_NAMES, n_obs)
    print(f"Saved → {out_file}")
    print(f"  param_flat: {param_flat.shape}")
    n_nan = int(np.sum(np.any(np.isnan(param_flat), axis=-1)))
    if n_nan:
        print(f"  WARNING: {n_nan} grid points have NaN.")

    _save_metadata(out_dir, obs_selection, obs_offsets, n_corr,
                   {k: v for k, v in cfg.items()})

    if save_raw and raw_flat_list:
        raw_file = out_dir / (out_file.stem + '_raw.npz')
        _save_raw_npz(raw_file,
                      np.vstack(raw_flat_list),
                      np.vstack(raw_grid_flat_list))
        print(f"Raw events → {raw_file}")


if __name__ == '__main__':
    main()

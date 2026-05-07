#!/usr/bin/env python3
"""
scan_regression_gennorm.py
==========================
Scan (mZ', mRho, rinv, alphaD) parameter space, fitting a Multivariate Normal
on 12 transformed observables at each grid point.

At each grid point the pipeline is:
  1. Run ./svj_regression (save_tsv=1, full_obs=1); raw TSV is written to a
     worker-unique /tmp path and deleted immediately after reading.
  2. Apply observable selection + row filter (see below).
  3. Fit Box-Cox + GenNorm per marginal → transform_params (12×4).
  4. Map data to standard-normal marginals → tr_data (M×12).
  5. Estimate MVN correlation matrix with mean=0, var=1 fixed:
       R = tr_data.T @ tr_data / M
     Extract upper-triangle → corr_params (66,).

Observable selection (mask cols of raw 16-col TSV):
  idx  raw col   name
   0     0       leadVisPt
   1     1       leadWidth
   2     2       MET
   3     4       maxMuPt
   4     5       1 - jetThrust         (flipped)
   5     7       hemiMass1
   6     8       hemiMass2
   7    11       e2c
   8    12       e3c
   9    13       tau1
  10    14       tau2
  11    15       tau3

Parallelisation:
  n_outer_workers grid points run simultaneously (Python ProcessPoolExecutor).
  Each spawns ./svj_regression with nWorkers internal C++ threads.
  Total hardware threads used ≈ n_outer_workers × nWorkers.

Resumption:
  If simulated/gennorm/gennorm_scan.npz already exists, already-finished grid
  points are loaded and skipped. Run the script again after a crash to continue.

Checkpointing:
  Results are saved to the output file every checkpoint_every completions.

Output: simulated/gennorm/gennorm_scan.npz
  corr_params      (N_mZ, N_mRho, N_rinv, N_alphaD, 66)
  transform_params (N_mZ, N_mRho, N_rinv, N_alphaD, 12, 4)  [lam, β, loc, scale]
  obs_names        (12,)
  scan_params      (N_mZ, N_mRho, N_rinv, N_alphaD, 6)  [mZ,mRho,mPi,Λ,rinv,αD]
  mZ_vals, mRho_vals, rinv_vals, alphaD_vals
"""

import subprocess
import sys
import os
import time
import itertools
import numpy as np
import scipy.stats as st
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

BINARY = './svj_regression'

MRHO_MPION_RATIO  = 8.0 / 15.5
MRHO_LAMBDA_RATIO = 5.0 / 15.5

OBS_MASK  = np.array([0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15])
N_OBS     = len(OBS_MASK)                  # 12
CORR_IDX  = np.triu_indices(N_OBS, k=1)   # upper-triangle index pair
N_CORR    = len(CORR_IDX[0])              # 66 = C(12,2)

OBS_NAMES = np.array([
    'leadVisPt', 'leadWidth', 'MET',
    'maxMuPt', 'inv_jetThrust', 'hemiMass1', 'hemiMass2',
    'e2c', 'e3c', 'tau1', 'tau2', 'tau3',
], dtype=object)


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


# ── Preprocessing + transform (no helpers.py import) ──────────────────────────

def load_and_preprocess(tsv_path):
    """Read raw 16-col TSV, apply selection and row filter. Returns X or None."""
    data = np.loadtxt(tsv_path, comments='#')
    if data.ndim != 2 or data.shape[1] != 16:
        return None

    X = data[:, OBS_MASK].copy()       # (N, 12)
    X[:, 4] = 1.0 - X[:, 4]           # flip jetThrust

    valid = np.all((X > 0) & np.isfinite(X), axis=1)
    X = X[valid]
    return X if len(X) >= 20 else None


def transform_data(X):
    """Box-Cox + GenNorm CDF → standard normal per column."""
    n_obs   = X.shape[1]
    params  = np.empty((n_obs, 4))
    tr_data = np.empty(X.shape)
    for i in range(n_obs):
        x_bc, lam = st.boxcox(X[:, i])
        # Seed MLE with empirical moments: cuts convergence time ~2-3×.
        # Beta=2 (Gaussian) is a good neutral starting point after Box-Cox.
        beta, loc, scale = st.gennorm.fit(
            x_bc, 2.0, loc=x_bc.mean(), scale=x_bc.std()
        )
        params[i] = lam, beta, loc, scale
        cdf_vals = np.clip(
            st.gennorm.cdf(x_bc, beta, loc=loc, scale=scale),
            1e-10, 1.0 - 1e-10,
        )
        tr_data[:, i] = st.norm.ppf(cdf_vals)
    return tr_data, params


def estimate_corr(tr_data):
    """Upper-triangle of MVN correlation matrix (mean=0, var=1 fixed)."""
    R = (tr_data.T @ tr_data) / len(tr_data)   # (12, 12)
    return R[CORR_IDX]                           # (55,)


# ── Worker — must be module-level for multiprocessing pickling ─────────────────

def _worker(args):
    """
    Run one grid point. Returns (i, j, k, l, corr_55, tf_11x4) or
    (i, j, k, l, None, None) on failure. Cleans up all temp files.
    """
    (task_id, i, j, k, l,
     mZ, mRho, mPi, LambdaQCD, rinv, alphaD,
     mq, Brl, jetR, nEvent, nWorkers_inner) = args

    temp_cfg = f'/tmp/svj_gennorm_{task_id}.cfg'
    temp_tsv = f'/tmp/svj_gennorm_{task_id}.tsv'

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
            return i, j, k, l, None, None

        if not os.path.exists(temp_tsv):
            return i, j, k, l, None, None

        try:
            X = load_and_preprocess(temp_tsv)
        finally:
            try:
                os.remove(temp_tsv)
            except FileNotFoundError:
                pass

        if X is None:
            return i, j, k, l, None, None

        tr_data, tf_params = transform_data(X)
        return i, j, k, l, estimate_corr(tr_data), tf_params

    except Exception:
        return i, j, k, l, None, None
    finally:
        try:
            os.remove(temp_cfg)
        except FileNotFoundError:
            pass


# ── NPZ save helper ────────────────────────────────────────────────────────────

def _save(out_file, mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
          corr_params, transform_params, scan_params, scan_param_names):
    np.savez(
        out_file,
        mZ_vals          = mZ_vals,
        mRho_vals        = mRho_vals,
        rinv_vals        = rinv_vals,
        alphaD_vals      = alphaD_vals,
        corr_params      = corr_params,
        transform_params = transform_params,
        obs_names        = OBS_NAMES,
        scan_params      = scan_params,
        scan_param_names = scan_param_names,
    )


# ── Merge helper ───────────────────────────────────────────────────────────────

def _merge(out_dir, n_jobs):
    """
    Combine gennorm_scan_0.npz … gennorm_scan_{n_jobs-1}.npz
    into gennorm_scan.npz.  Run after all array jobs finish:
        python scan_regression_gennorm.py --merge --n-jobs 4
    """
    files = [out_dir / f'gennorm_scan_{i}.npz' for i in range(n_jobs)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print(f"Error: missing files: {missing}")
        sys.exit(1)

    d0               = np.load(files[0], allow_pickle=True)
    corr_params      = np.array(d0['corr_params'])
    transform_params = np.array(d0['transform_params'])
    scan_params      = np.array(d0['scan_params'])

    for f in files[1:]:
        d    = np.load(f, allow_pickle=True)
        cp   = np.array(d['corr_params'])
        tp   = np.array(d['transform_params'])
        sp   = np.array(d['scan_params'])
        mask = np.all(np.isfinite(cp), axis=-1)
        corr_params[mask]      = cp[mask]
        transform_params[mask] = tp[mask]
        sp_mask = np.all(np.isfinite(sp), axis=-1)
        scan_params[sp_mask]   = sp[sp_mask]

    out_file = out_dir / 'gennorm_scan.npz'
    np.savez(
        out_file,
        mZ_vals          = d0['mZ_vals'],
        mRho_vals        = d0['mRho_vals'],
        rinv_vals        = d0['rinv_vals'],
        alphaD_vals      = d0['alphaD_vals'],
        corr_params      = corr_params,
        transform_params = transform_params,
        obs_names        = d0['obs_names'],
        scan_params      = scan_params,
        scan_param_names = d0['scan_param_names'],
    )
    n_done  = int(np.sum(np.all(np.isfinite(corr_params), axis=-1)))
    n_total = corr_params[..., 0].size
    print(f"Merged {n_jobs} files → {out_file}")
    print(f"  {n_done} / {n_total} points complete")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', nargs='?', default='scan_regression.cfg')
    parser.add_argument('--job-index', type=int, default=0,
                        help='Index of this job slice (0-based)')
    parser.add_argument('--n-jobs',    type=int, default=1,
                        help='Total number of parallel job slices')
    parser.add_argument('--merge', action='store_true',
                        help='Merge per-job NPZs into gennorm_scan.npz and exit')
    args = parser.parse_args()

    job_index = args.job_index
    n_jobs    = args.n_jobs

    out_dir = Path('simulated') / 'gennorm'   # needed early for --merge

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

    out_dir  = Path(cfg_str(cfg, 'output_dir', 'simulated')) / 'gennorm'
    fname    = f'gennorm_scan_{job_index}.npz' if n_jobs > 1 else 'gennorm_scan.npz'
    out_file = out_dir / fname
    out_dir.mkdir(parents=True, exist_ok=True)

    mZ_vals     = np.linspace(mZ_min,     mZ_max,     mZ_n)
    mRho_vals   = np.linspace(mRho_min,   mRho_max,   mRho_n)
    rinv_vals   = np.linspace(rinv_min,   rinv_max,   rinv_n)
    alphaD_vals = np.linspace(alphaD_min, alphaD_max, alphaD_n)

    total = mZ_n * mRho_n * rinv_n * alphaD_n
    SCAN_PARAM_NAMES = np.array(
        ['mZ', 'mRho', 'mPi', 'LambdaQCD', 'rinv', 'alphaD'], dtype=object)

    # Allocate result arrays
    corr_params      = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, N_CORR),   np.nan)
    transform_params = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, N_OBS, 4), np.nan)
    scan_params      = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, 6),         np.nan)

    # Resumption: load existing results and skip already-finished points
    n_preloaded = 0
    if out_file.exists():
        try:
            existing = np.load(out_file, allow_pickle=True)
            corr_params[...]      = existing['corr_params']
            transform_params[...] = existing['transform_params']
            scan_params[...]      = existing['scan_params']
            n_preloaded = int(np.sum(np.all(np.isfinite(corr_params), axis=-1)))
            print(f"Resuming: {n_preloaded} points already done, loaded from {out_file}")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}); starting fresh.")

    # Enumerate all grid points in a fixed order, then assign to jobs round-robin
    # so each job always owns the same deterministic 1/n_jobs slice of the grid.
    all_points = list(itertools.product(
        enumerate(mZ_vals), enumerate(mRho_vals),
        enumerate(rinv_vals), enumerate(alphaD_vals)))

    # scan_params is cheap; fill for all points regardless of job slice
    for (i, mZ), (j, mRho), (k, rinv), (l, alphaD) in all_points:
        mPi       = mRho * MRHO_MPION_RATIO
        LambdaQCD = mRho * MRHO_LAMBDA_RATIO
        scan_params[i, j, k, l] = [mZ, mRho, mPi, LambdaQCD, rinv, alphaD]

    # Build task list for this job's slice, skipping already-done points
    tasks = []
    for flat_idx, ((i, mZ), (j, mRho), (k, rinv), (l, alphaD)) in enumerate(all_points):
        if flat_idx % n_jobs != job_index:
            continue
        if np.all(np.isfinite(corr_params[i, j, k, l])):
            continue  # already done (resumption)
        mPi       = scan_params[i, j, k, l, 2]
        LambdaQCD = scan_params[i, j, k, l, 3]
        task_id   = i * (mRho_n * rinv_n * alphaD_n) \
                  + j * (rinv_n * alphaD_n) \
                  + k * alphaD_n + l
        tasks.append((task_id, i, j, k, l,
                      mZ, mRho, mPi, LambdaQCD, rinv, alphaD,
                      mq, Brl, jetR, nEvent, nWorkers))

    n_todo = len(tasks)
    job_str = f"job {job_index}/{n_jobs-1}  " if n_jobs > 1 else ""
    print(f"SVJ gennorm scan  {job_str}({n_outer} outer × {nWorkers} C++ threads)")
    print(f"  Grid:    {mZ_n}×{mRho_n}×{rinv_n}×{alphaD_n} = {total} total  "
          f"({total // n_jobs} this job)")
    print(f"  To do:   {n_todo}  (already done: {n_preloaded})")
    print(f"  Events/point: {nEvent}")
    print(f"  Output:  {out_file}\n")

    if n_todo == 0:
        print("Nothing left to do.")
        return

    t0     = time.time()
    done   = 0
    failed = 0
    width  = len(str(n_todo))

    save_args = (out_file, mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
                 corr_params, transform_params, scan_params, SCAN_PARAM_NAMES)

    with ProcessPoolExecutor(max_workers=n_outer) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}

        for fut in as_completed(futs):
            try:
                i, j, k, l, corr, tf = fut.result()
            except Exception as e:
                print(f"  WARNING: worker exception: {e}", flush=True)
                failed += 1
                done   += 1
                continue

            done += 1
            ok = corr is not None
            if ok:
                corr_params[i, j, k, l]      = corr
                transform_params[i, j, k, l] = tf
            else:
                failed += 1

            elapsed = time.time() - t0
            avg     = elapsed / done
            h, rem  = divmod(int(avg * (n_todo - done)), 3600)
            mm, s   = divmod(rem, 60)
            print(f"[{done:{width}}/{n_todo}]  ({i},{j},{k},{l})"
                  f"  {'ok  ' if ok else 'FAIL'}"
                  f"  ETA {h:02d}:{mm:02d}:{s:02d}", flush=True)

            if done % chk_every == 0:
                _save(*save_args)
                print(f"  [checkpoint @ {done}]", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nDone. {done} points in {elapsed_total / 60:.1f} min; {failed} failed.")

    _save(*save_args)
    print(f"Saved → {out_file}")
    print(f"  corr_params:      {corr_params.shape}")
    print(f"  transform_params: {transform_params.shape}")
    n_nan = int(np.sum(np.any(np.isnan(corr_params), axis=-1)))
    if n_nan:
        print(f"  WARNING: {n_nan} grid points have NaN.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
scan_regression.py
==================
Scan (mZ', mRho, rinv, alphaD) parameter space.

For each grid point:
  • Derive  mPi      = mRho * (8.0 / 15.5)   [fixed mrho/mpion ratio]
            LambdaQCD = mRho * (5.0 / 15.5)   [fixed mrho/Lambda ratio]
  • Write a temporary svj_regression config and call ./svj_regression.
  • Parse the "RESULT: ..." line from stdout (10 floats).
  • Store in a (N_mZ, N_mRho, N_rinv, N_alphaD, 10) array.

Saved to: <output_dir>/<output_file>  (default: simulated/regression_scan.npz)

  Array layout of the 10 parameters per point:
    index  name         description
    0      mu_pT        mean of leading-jet visible pT (GeV)
    1      mu_logW      mean of log(leading-jet width)
    2      mu_logMET    mean of log(event MET proxy)
    3      S00          Sigma[pT, pT]
    4      S01          Sigma[pT, logW]
    5      S02          Sigma[pT, logMET]
    6      S11          Sigma[logW, logW]
    7      S12          Sigma[logW, logMET]
    8      S22          Sigma[logMET, logMET]
    9      nu           degrees of freedom (multivariate-t)

  Access example:
    d = np.load('simulated/regression_scan.npz', allow_pickle=True)
    # axes
    mZ_vals, mRho_vals, rinv_vals, alphaD_vals = (
        d['mZ_vals'], d['mRho_vals'], d['rinv_vals'], d['alphaD_vals'])
    params = d['params']          # shape (N_mZ, N_mRho, N_rinv, N_alphaD, 10)
    names  = d['param_names']     # array of 10 strings

    # reconstruct mu and Sigma at grid point (i,j,k,l):
    p = params[i, j, k, l]
    mu    = p[0:3]
    Sigma = np.array([[p[3], p[4], p[5]],
                      [p[4], p[6], p[7]],
                      [p[5], p[7], p[8]]])
    nu    = p[9]
"""

import subprocess
import sys
import os
import time
import itertools
import tempfile
import numpy as np
from pathlib import Path

BINARY   = './svj_regression'
TEMP_CFG = '/tmp/svj_scan_point.cfg'

PARAM_NAMES = np.array([
    'mu_pT', 'mu_logW', 'mu_logMET',
    'S00', 'S01', 'S02',
    'S11', 'S12',
    'S22',
    'nu',
], dtype=object)

MRHO_MPION_RATIO  = 8.0  / 15.5   # mPi      = mRho * this
MRHO_LAMBDA_RATIO = 5.0  / 15.5   # LambdaQCD = mRho * this


# ── Config reader (same key=value style as svj_regression.cfg) ────────────────

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
            key = key.strip()
            val = val.split('#')[0].strip()
            if val:
                cfg[key] = val
    return cfg

def cfg_float(cfg, key, default):
    return float(cfg.get(key, default))

def cfg_int(cfg, key, default):
    return int(cfg.get(key, default))

def cfg_str(cfg, key, default):
    return cfg.get(key, str(default))


# ── Temp config writer ────────────────────────────────────────────────────────

def write_point_cfg(path, params: dict):
    with open(path, 'w') as f:
        f.write("# auto-generated per-point config — do not edit\n")
        for k, v in params.items():
            f.write(f"{k} = {v}\n")


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_point(cfg_path):
    """
    Call ./svj_regression with cfg_path, return the 10 fitted floats,
    or an array of NaNs if the run fails.
    """
    proc = subprocess.run(
        [BINARY, cfg_path],
        capture_output=True, text=True
    )
    # Find the RESULT line (robust to any PYTHIA banner text)
    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT:'):
            result_line = line
            break

    if result_line is None:
        print(f"  WARNING: no RESULT line found. stderr:\n{proc.stderr[:400]}")
        return np.full(10, np.nan)

    tokens = result_line.split()[1:]   # drop 'RESULT:'
    if len(tokens) != 10:
        print(f"  WARNING: expected 10 tokens, got {len(tokens)}: {tokens}")
        return np.full(10, np.nan)

    vals = np.array([float(t) for t in tokens])
    if np.any(np.isnan(vals)):
        print("  WARNING: NaN in result.")
    return vals


# ── Main scan ─────────────────────────────────────────────────────────────────

def main():
    scan_cfg_path = 'scan_regression.cfg'
    if len(sys.argv) > 1:
        scan_cfg_path = sys.argv[1]

    if not os.path.exists(scan_cfg_path):
        print(f"Error: config file '{scan_cfg_path}' not found.")
        sys.exit(1)

    if not os.path.exists(BINARY):
        print(f"Error: binary '{BINARY}' not found. Run 'make svj_regression' first.")
        sys.exit(1)

    cfg = read_cfg(scan_cfg_path)

    # ── Grid axes ──
    mZ_min  = cfg_float(cfg, 'mZ_min',  500.0)
    mZ_max  = cfg_float(cfg, 'mZ_max', 4000.0)
    mZ_n    = cfg_int  (cfg, 'mZ_n',      10)

    mRho_min  = cfg_float(cfg, 'mRho_min',  10.0)
    mRho_max  = cfg_float(cfg, 'mRho_max',  30.0)
    mRho_n    = cfg_int  (cfg, 'mRho_n',      10)

    rinv_min  = cfg_float(cfg, 'rinv_min',  0.05)
    rinv_max  = cfg_float(cfg, 'rinv_max',  0.75)
    rinv_n    = cfg_int  (cfg, 'rinv_n',      10)

    alphaD_min  = cfg_float(cfg, 'alphaD_min',  0.05)
    alphaD_max  = cfg_float(cfg, 'alphaD_max',  0.50)
    alphaD_n    = cfg_int  (cfg, 'alphaD_n',      10)

    # ── Fixed params ──
    mq    = cfg_float(cfg, 'mq',    4.0)
    rinv2 = cfg_float(cfg, 'rinv2', 0.3)
    Brl   = cfg_float(cfg, 'Brl',   0.3)
    jetR  = cfg_float(cfg, 'jetR',  1.0)

    # ── Sim settings ──
    nEvent   = cfg_int(cfg, 'nEvent',   1000)
    nWorkers = cfg_int(cfg, 'nWorkers',   10)

    # ── Output ──
    out_dir  = Path(cfg_str(cfg, 'output_dir',  'simulated'))
    out_file = out_dir / cfg_str(cfg, 'output_file', 'regression_scan.npz')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build grids
    mZ_vals     = np.linspace(mZ_min,     mZ_max,     mZ_n)
    mRho_vals   = np.linspace(mRho_min,   mRho_max,   mRho_n)
    rinv_vals   = np.linspace(rinv_min,   rinv_max,   rinv_n)
    alphaD_vals = np.linspace(alphaD_min, alphaD_max, alphaD_n)

    total = mZ_n * mRho_n * rinv_n * alphaD_n
    params      = np.full((mZ_n, mRho_n, rinv_n, alphaD_n, 10), np.nan)
    # scan_params[i,j,k,l] = [mZ, mRho, mPi, LambdaQCD, rinv, alphaD]
    scan_params = np.full((mZ_n, mRho_n, rinv_n, alphaD_n,  6), np.nan)
    SCAN_PARAM_NAMES = np.array(
        ["mZ", "mRho", "mPi", "LambdaQCD", "rinv", "alphaD"], dtype=object)

    print(f"SVJ regression parameter scan")
    print(f"  mZ':    {mZ_n} points in [{mZ_min}, {mZ_max}] GeV")
    print(f"  mRho:   {mRho_n} points in [{mRho_min}, {mRho_max}] GeV")
    print(f"  rinv:   {rinv_n} points in [{rinv_min}, {rinv_max}]")
    print(f"  alphaD: {alphaD_n} points in [{alphaD_min}, {alphaD_max}]")
    print(f"  Total:  {total} points  x  {nEvent} events  x  {nWorkers} threads")
    print(f"  Output: {out_file}\n")

    t0 = time.time()
    done = 0

    for (i, mZ), (j, mRho), (k, rinv), (l, alphaD) in itertools.product(
            enumerate(mZ_vals),
            enumerate(mRho_vals),
            enumerate(rinv_vals),
            enumerate(alphaD_vals)):

        done += 1
        mPi      = mRho * MRHO_MPION_RATIO
        LambdaQCD = mRho * MRHO_LAMBDA_RATIO

        # ETA estimate
        elapsed = time.time() - t0
        eta_str = ''
        if done > 1:
            avg = elapsed / (done - 1)
            remaining = avg * (total - done + 1)
            h, rem = divmod(int(remaining), 3600)
            m, s   = divmod(rem, 60)
            eta_str = f'  ETA {h:02d}:{m:02d}:{s:02d}'

        print(f"[{done:>{len(str(total))}}/{total}]"
              f"  mZ={mZ:.0f}  mRho={mRho:.2f}  rinv={rinv:.3f}  alphaD={alphaD:.3f}"
              f"  (mPi={mPi:.2f}  Lambda={LambdaQCD:.2f})"
              f"{eta_str}",
              flush=True)

        write_point_cfg(TEMP_CFG, {
            'mZ':         mZ,
            'mq':         mq,
            'mPi':        mPi,
            'mRho':       mRho,
            'rinv':       rinv,
            'rinv2':      rinv,
            'Brl':        Brl,
            'alphaD':     alphaD,
            'nEvent':     nEvent,
            'jetR':       jetR,
            'LambdaDQCD': LambdaQCD,
            'nWorkers':   nWorkers,
            'save_tsv':   0,
        })

        scan_params[i, j, k, l] = [mZ, mRho, mPi, LambdaQCD, rinv, alphaD]
        params[i, j, k, l]      = run_point(TEMP_CFG)

    elapsed_total = time.time() - t0
    print(f"\nDone. Total time: {elapsed_total/60:.1f} min")

    np.savez(
        out_file,
        mZ_vals          = mZ_vals,
        mRho_vals        = mRho_vals,
        rinv_vals        = rinv_vals,
        alphaD_vals      = alphaD_vals,
        params           = params,        # (N_mZ, N_mRho, N_rinv, N_alphaD, 10)
        param_names      = PARAM_NAMES,
        scan_params      = scan_params,   # (N_mZ, N_mRho, N_rinv, N_alphaD,  6)
        scan_param_names = SCAN_PARAM_NAMES,
    )
    print(f"Saved to {out_file}")
    print(f"  params shape: {params.shape}")
    n_nan = np.sum(np.any(np.isnan(params), axis=-1))
    if n_nan:
        print(f"  WARNING: {n_nan} grid points have NaN (failed runs).")

    # Clean up temp file
    try:
        os.remove(TEMP_CFG)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()

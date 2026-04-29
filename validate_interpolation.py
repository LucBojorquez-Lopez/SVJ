#!/usr/bin/env python3
"""
validate_interpolation.py
=========================
Validate the SVJ regression interpolation against live PYTHIA simulations.

For N random points sampled uniformly within the grid bounds:
  1. Interpolate the 10 fit parameters via helpers.interpolate_params()
  2. Run ./svj_regression (no TSV saved) to get the live-fitted parameters
  3. Compute KLD(live_fit, interpolated) — live fit is the "true" distribution
  4. Compute the Euclidean distance to the nearest grid point in normalised [0,1]^4 space
  5. Append (mZ, mRho, rinv, alphaD, KLD, dist) to simulated/validation_results.npy,
     saving atomically after every completed point.

Usage:
    python validate_interpolation.py <N_points> <N_events> [cfg_path] [n_parallel]

    N_points   : number of random validation points
    N_events   : number of PYTHIA events per simulated point
    cfg_path   : path to config file (default: svj_regression.cfg)
    n_parallel : simultaneous points (default: cpu_count // nWorkers, min 1)

Fixed physics params (mq, rinv2, Brl, jetR, nWorkers) are read from cfg_path.
mPi and LambdaDQCD are derived per point:
    mPi        = mRho * (8.0 / 15.5)
    LambdaDQCD = mRho * (5.0 / 15.5)

Output columns (saved as float64 (n, 6) array):
    0: mZ      1: mRho      2: rinv      3: alphaD
    4: KLD     5: dist_to_nearest_grid_point  (normalised Euclidean)
"""

import sys
import os
import threading
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

BINARY   = './svj_regression'
OUT_FILE = Path('simulated/validation_results.npy')

MRHO_MPION_RATIO  = 8.0 / 15.5
MRHO_LAMBDA_RATIO = 5.0 / 15.5

_save_lock  = threading.Lock()
_print_lock = threading.Lock()


# ── Config helpers ────────────────────────────────────────────────────────────

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


# ── Per-point helpers ─────────────────────────────────────────────────────────

def write_point_cfg(path, params):
    with open(path, 'w') as f:
        f.write("# auto-generated validation config — do not edit\n")
        for k, v in params.items():
            f.write(f"{k} = {v}\n")


def run_svj(cfg_path):
    """Call ./svj_regression, return the 10 fitted floats or None on failure."""
    proc = subprocess.run([BINARY, cfg_path], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT:'):
            tokens = line.split()[1:]
            if len(tokens) != 10:
                print(f"  WARNING: expected 10 tokens, got {len(tokens)}")
                return None
            vals = np.array([float(t) for t in tokens])
            if np.any(np.isnan(vals)):
                print("  WARNING: NaN in RESULT.")
                return None
            return vals
    print(f"  WARNING: no RESULT line. stderr:\n{proc.stderr[:400]}")
    return None


def nearest_grid_dist(mZ, mRho, rinv, alphaD):
    """
    Euclidean distance to the nearest grid point in normalised [0,1]^4 space.
    Each axis is normalised independently by its [min, max] range.
    """
    axes = [
        (mZ,     helpers._mZ_vals),
        (mRho,   helpers._mRho_vals),
        (rinv,   helpers._rinv_vals),
        (alphaD, helpers._alphaD_vals),
    ]
    d2 = 0.0
    for val, grid in axes:
        lo, hi    = grid[0], grid[-1]
        val_norm  = (val - lo) / (hi - lo)
        grid_norm = (grid - lo) / (hi - lo)
        nearest   = grid_norm[np.argmin(np.abs(grid - val))]
        d2       += (val_norm - nearest) ** 2
    return np.sqrt(d2)


def save_results(results, path):
    """Atomically overwrite path with the current (n, 6) results list."""
    tmp = str(path.parent / (path.stem + '.tmp.npy'))
    np.save(tmp, np.array(results, dtype=np.float64))
    os.replace(tmp, path)


# ── Worker ────────────────────────────────────────────────────────────────────

def validate_point(point_idx, mZ, mRho, rinv, alphaD, fixed, n_events, n_workers):
    """
    Run one validation point in a thread.
    Returns (6-tuple, None) on success, or (None, reason_str) on failure.
    Uses pid+tid for the temp cfg filename to avoid collisions across threads.
    """
    cfg_path   = f'/tmp/svj_validate_{os.getpid()}_{threading.get_ident()}.cfg'
    mPi        = mRho * MRHO_MPION_RATIO
    LambdaDQCD = mRho * MRHO_LAMBDA_RATIO

    try:
        interp_params = helpers.interpolate_params(mZ, mRho, rinv, alphaD)
    except ValueError as e:
        return None, f"interpolation error: {e}"

    write_point_cfg(cfg_path, {
        'mZ':         mZ,
        'mq':         fixed['mq'],
        'mPi':        mPi,
        'mRho':       mRho,
        'rinv':       rinv,
        'rinv2':      fixed['rinv2'],
        'Brl':        fixed['Brl'],
        'alphaD':     alphaD,
        'nEvent':     n_events,
        'jetR':       fixed['jetR'],
        'LambdaDQCD': LambdaDQCD,
        'nWorkers':   n_workers,
        'save_tsv':   0,
    })

    live_params = run_svj(cfg_path)
    try:
        os.remove(cfg_path)
    except FileNotFoundError:
        pass

    if live_params is None:
        return None, "simulation failed"

    kld_val = helpers.kld_params(live_params, interp_params)
    dist    = nearest_grid_dist(mZ, mRho, rinv, alphaD)

    return (mZ, mRho, rinv, alphaD, kld_val, dist), None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <N_points> <N_events> [cfg_path] [n_parallel]")
        sys.exit(1)

    n_points       = int(sys.argv[1])
    n_events       = int(sys.argv[2])
    cfg_path       = sys.argv[3] if len(sys.argv) > 3 else 'svj_regression.cfg'
    n_parallel_arg = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if not os.path.exists(cfg_path):
        print(f"Error: config '{cfg_path}' not found.")
        sys.exit(1)
    if not os.path.exists(BINARY):
        print(f"Error: binary '{BINARY}' not found.")
        sys.exit(1)

    cfg      = read_cfg(cfg_path)
    mq       = cfg_float(cfg, 'mq',       4.0)
    rinv2    = cfg_float(cfg, 'rinv2',    0.3)
    Brl      = cfg_float(cfg, 'Brl',      0.3)
    jetR     = cfg_float(cfg, 'jetR',     1.0)
    nWorkers = cfg_int  (cfg, 'nWorkers', 10)

    fixed = {'mq': mq, 'rinv2': rinv2, 'Brl': Brl, 'jetR': jetR}

    # n_parallel: how many svj_regression calls run simultaneously.
    # Default: pack as many as fit given nWorkers each, up to cpu_count.
    cpu_count  = os.cpu_count() or 1
    n_parallel = n_parallel_arg if n_parallel_arg is not None \
                 else max(1, cpu_count // nWorkers)

    # Grid bounds from the loaded interpolator axes
    mZ_min,     mZ_max     = helpers._mZ_vals[0],     helpers._mZ_vals[-1]
    mRho_min,   mRho_max   = helpers._mRho_vals[0],   helpers._mRho_vals[-1]
    rinv_min,   rinv_max   = helpers._rinv_vals[0],   helpers._rinv_vals[-1]
    alphaD_min, alphaD_max = helpers._alphaD_vals[0], helpers._alphaD_vals[-1]

    print("SVJ interpolation validation")
    print(f"  N points     : {n_points}")
    print(f"  N events     : {n_events} per point")
    print(f"  n_parallel   : {n_parallel}  (simultaneous svj_regression calls)")
    print(f"  nWorkers     : {nWorkers}  per call  (total ~{n_parallel * nWorkers} threads)")
    print(f"  CPUs         : {cpu_count}")
    print(f"  Grid         : mZ [{mZ_min}, {mZ_max}]  mRho [{mRho_min}, {mRho_max}]"
          f"  rinv [{rinv_min}, {rinv_max}]  alphaD [{alphaD_min}, {alphaD_max}]")
    print(f"  Output       : {OUT_FILE}\n")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        existing = np.load(OUT_FILE)
        results  = list(existing)
        print(f"  Resuming: {len(results)} point(s) already saved.\n")
    else:
        results = []
        save_results(results, OUT_FILE)

    rng = np.random.default_rng()
    random_points = np.column_stack([
        rng.uniform(mZ_min,     mZ_max,     n_points),
        rng.uniform(mRho_min,   mRho_max,   n_points),
        rng.uniform(rinv_min,   rinv_max,   n_points),
        rng.uniform(alphaD_min, alphaD_max, n_points),
    ])

    n_done    = 0
    n_skipped = 0

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {
            executor.submit(
                validate_point,
                i, mZ, mRho, rinv, alphaD,
                fixed, n_events, nWorkers
            ): (i, mZ, mRho, rinv, alphaD)
            for i, (mZ, mRho, rinv, alphaD) in enumerate(random_points)
        }

        for future in as_completed(futures):
            i, mZ, mRho, rinv, alphaD = futures[future]
            result, reason = future.result()

            with _print_lock:
                n_done += 1
                if result is None:
                    n_skipped += 1
                    print(f"[{n_done}/{n_points}]  mZ={mZ:.1f}  mRho={mRho:.3f}"
                          f"  rinv={rinv:.4f}  alphaD={alphaD:.4f}"
                          f"  SKIP: {reason}", flush=True)
                else:
                    print(f"[{n_done}/{n_points}]  mZ={mZ:.1f}  mRho={mRho:.3f}"
                          f"  rinv={rinv:.4f}  alphaD={alphaD:.4f}"
                          f"  KLD={result[4]:.6f}  dist={result[5]:.4f}", flush=True)

            if result is not None:
                with _save_lock:
                    results.append(list(result))
                    save_results(results, OUT_FILE)

    print(f"\nDone. {len(results)} point(s) saved to {OUT_FILE}"
          f"  ({n_skipped} skipped).")


if __name__ == '__main__':
    main()

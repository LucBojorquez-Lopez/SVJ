"""
_val_utils.py
=============
Shared utilities for the three SVJ validation scripts.
Not intended to be called directly.
"""

import os
import sys
import tempfile
import json
import subprocess
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import jensenshannon, cdist

_HERE  = Path(__file__).resolve().parent   # src/run_regression/
_SRC   = _HERE.parent                       # src/
BINARY = str(_SRC / 'generate_events' / 'svj_regression')

# Ensure src/ is importable in worker subprocesses (inherited via fork on Linux).
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── Parameter resolution ───────────────────────────────────────────────────────

def resolve_point(scan_point, fixed_params, derived_exprs):
    """
    Merge scan + fixed params, then evaluate derived expressions.
    Identical logic to scan_svj.resolve_point; reproduced here to keep
    validation scripts self-contained.
    """
    params = {}
    params.update(fixed_params)
    params.update(scan_point)

    unresolved = dict(derived_exprs)
    for _ in range(len(unresolved) + 1):
        still_pending = {}
        for name, expr in unresolved.items():
            try:
                params[name] = float(eval(expr, {'__builtins__': {}}, dict(params)))
            except NameError:
                still_pending[name] = expr
            except Exception as e:
                raise ValueError(f"Cannot evaluate '{name} = {expr}': {e}")
        if not still_pending:
            break
        unresolved = still_pending
    else:
        raise ValueError(
            f"Derived expressions could not be resolved "
            f"(circular or undefined): {list(unresolved.keys())}")
    return params


# ── Scan loader ────────────────────────────────────────────────────────────────

def load_scan(npz_path):
    """
    Load a scan NPZ and its co-located svj_scan_meta.json.

    Returns a dict with:
        axis_names    list[str]
        axis_vals     list of np.ndarray
        param_flat    np.ndarray
        obs_names     list[str]
        param_offsets np.ndarray (n_obs+1,)
        corr_start    int
        fixed_params  dict[str, float]
        derived_exprs dict[str, str]
        interp        RegularGridInterpolator over param_flat
    """
    npz_path = Path(npz_path)
    npz = np.load(npz_path, allow_pickle=True)

    axis_names = [str(n) for n in npz['axis_names']]
    axis_vals  = [np.array(npz[f'{n}_vals']) for n in axis_names]
    param_flat = np.array(npz['param_flat'])
    obs_names  = [str(n) for n in npz['obs_names']]
    offsets    = np.array(npz['param_offsets'], dtype=int)
    corr_start = int(npz['corr_start'])

    meta_path = npz_path.parent / 'svj_scan_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta JSON not found at {meta_path}. "
            "Re-run scan_svj.py to regenerate it.")
    with open(meta_path) as f:
        meta = json.load(f)

    interp = RegularGridInterpolator(
        tuple(axis_vals), param_flat,
        method='linear', bounds_error=True)

    return {
        'axis_names':    axis_names,
        'axis_vals':     axis_vals,
        'param_flat':    param_flat,
        'obs_names':     obs_names,
        'param_offsets': offsets,
        'corr_start':    corr_start,
        'fixed_params':  meta['fixed_params'],
        'derived_exprs': meta['derived_exprs'],
        'interp':        interp,
    }


# ── Interpolation at a single point ────────────────────────────────────────────

def interp_at(scan, scan_point):
    """
    Interpolate param_flat at scan_point (dict axis_name → value).

    Returns
    -------
    R_upper       np.ndarray (n_corr,)
    flat_obs      np.ndarray (total_obs_params,)
    """
    axis_names = scan['axis_names']
    pt = np.array([[scan_point[n] for n in axis_names]])
    p  = scan['interp'](pt)[0]
    cs = scan['corr_start']
    return p[cs:], p[:cs]


# ── Interior point sampler ─────────────────────────────────────────────────────

def sample_interior_points(scan, N, rng):
    """
    Sample N random interior points, uniform in grid-index space.

    For each axis with N_k grid values, fractional indices are drawn from
    [0.5, N_k − 1.5] so the point always lies strictly between two grid
    nodes.  For N_k = 2 this collapses to the single midpoint 0.5.

    Returns
    -------
    frac_idxs   (N, K) float array of fractional grid indices
    scan_points list of N dicts {axis_name: physical_value}
    """
    axis_names = scan['axis_names']
    axis_vals  = scan['axis_vals']
    K = len(axis_names)

    lo = np.full(K, 0.5)
    hi = np.array([len(v) - 1.5 for v in axis_vals])

    bad = hi < lo
    if bad.any():
        raise ValueError(
            f"Axes {[axis_names[k] for k in np.where(bad)[0]]} have fewer "
            "than 2 grid points — cannot sample an interior point.")

    frac = rng.uniform(lo, hi, size=(N, K))

    scan_points = []
    for fi in frac:
        sp = {
            axis_names[k]: float(
                np.interp(fi[k], np.arange(len(axis_vals[k])), axis_vals[k]))
            for k in range(K)
        }
        scan_points.append(sp)

    return frac, scan_points


# ── Nearest grid point ─────────────────────────────────────────────────────────

def nearest_grid_point(scan, frac_idx):
    """
    Return the index tuple and scan_point dict of the grid vertex nearest
    to frac_idx (fractional grid indices, K-float array).
    """
    axis_names = scan['axis_names']
    axis_vals  = scan['axis_vals']
    K = len(axis_names)
    gidx = tuple(
        int(np.clip(round(float(frac_idx[k])), 0, len(axis_vals[k]) - 1))
        for k in range(K))
    scan_pt = {axis_names[k]: float(axis_vals[k][gidx[k]]) for k in range(K)}
    return gidx, scan_pt


# ── PYTHIA runner ──────────────────────────────────────────────────────────────

def run_pythia(full_params, n_events, seed_offset, task_tag):
    """
    Run the SVJ binary at full_params with n_events events.

    seed_offset is written to the cfg so repeated calls with the same
    physics params but different seed_offsets produce statistically
    independent event samples (seed = 1 + seed_offset for nWorkers=1).

    task_tag is a unique string used to name the scratch files (avoids
    collisions between concurrent workers).

    Returns
    -------
    (data, col_map) as from observables.load_tsv

    Raises RuntimeError on binary failure or missing output.
    Cleans up scratch files in all cases.
    """
    from observables import load_tsv

    # See the note in scan_svj._worker: gettempdir() honours $TMPDIR so batch
    # jobs write to job-private scratch rather than a shared worker-node /tmp.
    _uid     = os.environ.get('SLURM_ARRAY_TASK_ID') or str(os.getpid())
    _scratch = tempfile.gettempdir()
    cfg_path = os.path.join(_scratch, f'val_{_uid}_{task_tag}.cfg')
    tsv_path = os.path.join(_scratch, f'val_{_uid}_{task_tag}.tsv')

    try:
        with open(cfg_path, 'w') as fh:
            fh.write("# auto-generated validation config\n")
            for k, v in full_params.items():
                fh.write(f"{k} = {v}\n")
            fh.write(f"nEvent      = {n_events}\n")
            fh.write(f"nWorkers    = 1\n")
            fh.write(f"save_tsv    = 1\n")
            fh.write(f"tsv_file    = {tsv_path}\n")
            fh.write(f"seed_offset = {seed_offset}\n")

        proc = subprocess.run(
            [BINARY, cfg_path], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Binary exited with code {proc.returncode}.\n"
                f"stderr: {proc.stderr[-400:]}")
        if not os.path.exists(tsv_path):
            raise RuntimeError(
                f"Binary succeeded but produced no TSV at {tsv_path}.")

        return load_tsv(tsv_path)

    finally:
        for p in (cfg_path, tsv_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


# ── JS divergence per observable ───────────────────────────────────────────────

def js_per_obs(A, B, obs_names, bins=80):
    """
    Jensen-Shannon distance per observable column.

    Bin edges are set from the combined [0.5, 99.5] percentile range so
    tails do not dominate.  A small epsilon is added before JS to handle
    empty bins at the extremes.

    Parameters
    ----------
    A, B      : (N_a, n_obs) and (N_b, n_obs) arrays in physical units
    obs_names : list of str, length n_obs

    Returns
    -------
    js : (n_obs,) array in [0, 1]
    """
    n_obs = len(obs_names)
    js = np.zeros(n_obs)
    for i in range(n_obs):
        a = A[:, i]
        b = B[:, i]
        # Drop non-finite values before binning
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            js[i] = np.nan
            continue
        lo = np.percentile(np.concatenate([a, b]), 0.5)
        hi = np.percentile(np.concatenate([a, b]), 99.5)
        if hi <= lo:
            js[i] = 0.0
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(a, bins=edges)
        q, _ = np.histogram(b, bins=edges)
        js[i] = float(jensenshannon(p + 1e-10, q + 1e-10))
    return js


# ── MMD with RBF kernel ────────────────────────────────────────────────────────

def mmd_rbf(X, Y, n_sub=2000, rng=None):
    """
    Biased Maximum Mean Discrepancy with RBF kernel.

    Both sample sets are jointly z-scored before computing.  Each is
    subsampled to n_sub rows to keep computation O(n_sub^2).
    Bandwidth is set by the median heuristic on the combined subsample.

    Returns a non-negative float (0 indicates identical distributions).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # Drop rows with any non-finite entry
    X = X[np.all(np.isfinite(X), axis=1)]
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    if len(X) == 0 or len(Y) == 0:
        return np.nan

    # Joint normalisation
    combined = np.vstack([X, Y])
    mu    = combined.mean(axis=0)
    sigma = combined.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xn = (X - mu) / sigma
    Yn = (Y - mu) / sigma

    # Subsample
    _rng = np.random.default_rng() if rng is None else rng
    nX = min(len(Xn), n_sub)
    nY = min(len(Yn), n_sub)
    Xn = Xn[_rng.choice(len(Xn), nX, replace=False)]
    Yn = Yn[_rng.choice(len(Yn), nY, replace=False)]

    # Median bandwidth heuristic (subsample of the combined set)
    all_sub = np.vstack([Xn, Yn])
    n_bw = min(len(all_sub), 500)
    idx  = _rng.choice(len(all_sub), n_bw, replace=False)
    dmat = cdist(all_sub[idx], all_sub[idx])
    bw   = float(np.median(dmat[dmat > 0]))
    if bw < 1e-8:
        bw = 1.0
    gamma = 1.0 / (2.0 * bw ** 2)

    def K(A, B):
        return float(np.exp(-gamma * cdist(A, B, metric='sqeuclidean')).mean())

    return K(Xn, Xn) + K(Yn, Yn) - 2.0 * K(Xn, Yn)


# ── Observable column extraction ───────────────────────────────────────────────

def extract_obs_cols(data, col_map, obs_selection):
    """
    Extract observable columns from a raw TSV data array in obs_selection order.

    Returns (N, n_obs) float array in physical units.
    """
    from observables import OBSERVABLES
    cols = [col_map[OBSERVABLES[n]['col']] for n in obs_selection]
    return np.asarray(data[:, cols], dtype=float)

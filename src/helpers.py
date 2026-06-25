"""
src/helpers.py
==============
Interpolation helpers and KLD utilities for the SVJ regression model.

Loads svj_scan.npz (new format, from scan_svj.py):
    param_flat, param_offsets, corr_start, obs_names

Also supports the v1 scan (simulated/v1/regression_scan.npz) for KLD utilities.
Both NPZ files are loaded lazily: importing this module does not fail if they
are not present yet.
"""

import numpy as np
import scipy.special as sp
import scipy.stats as st
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════════
# Lazy-loaded NPZ data
# ══════════════════════════════════════════════════════════════════════════════

_v1_data  = None   # v1 regression_scan.npz
_svj_data = None   # svj_scan.npz


def _load_v1():
    global _v1_data
    if _v1_data is None:
        path = _REPO_ROOT / 'simulated/v1/regression_scan.npz'
        try:
            _v1_data = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"regression_scan.npz not found at {path}. "
                "Run the v1 scan first, or use old_version/.")
    return _v1_data


def _load_svj():
    global _svj_data
    if _svj_data is None:
        path = _REPO_ROOT / 'simulated/svj/svj_scan.npz'
        try:
            _svj_data = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"SVJ scan NPZ not found at {path}.\n"
                "Run scan_svj.py first.")
    return _svj_data


# ══════════════════════════════════════════════════════════════════════════════
# V1 interpolator (3-param MVT per point)
# ══════════════════════════════════════════════════════════════════════════════

_v1_interp  = None
_v1_params  = None
_v1_mZ      = _v1_mRho = _v1_rinv = _v1_alphaD = None


def _build_v1_interp():
    global _v1_interp, _v1_params
    global _v1_mZ, _v1_mRho, _v1_rinv, _v1_alphaD
    if _v1_interp is not None:
        return
    reg = _load_v1()
    _v1_mZ     = reg['mZ_vals']
    _v1_mRho   = reg['mRho_vals']
    _v1_rinv   = reg['rinv_vals']
    _v1_alphaD = reg['alphaD_vals']
    _v1_params = np.array(reg['params'])
    _v1_interp = RegularGridInterpolator(
        (_v1_mZ, _v1_mRho, _v1_rinv, _v1_alphaD),
        _v1_params,
        method='linear',
        bounds_error=True,
    )


def interpolate_params(mZ, mRho, rinv, alphaD):
    """
    Return the 10 estimated SVJ parameters at (mZ, mRho, rinv, alphaD)
    via linear interpolation on the v1 precomputed grid.

    Parameters
    ----------
    mZ, mRho, rinv, alphaD : float
        Physical parameters within the grid ranges:
          mZ     in [500, 4000] GeV
          mRho   in [10, 30] GeV
          rinv   in [0.05, 0.70]
          alphaD in [0.1, 0.8]

    Returns
    -------
    params : np.ndarray, shape (10,)
        [mu_pT, mu_logW, mu_logMET, S00, S01, S02, S11, S12, S22, nu]
    """
    _build_v1_interp()
    return _v1_interp([[mZ, mRho, rinv, alphaD]])[0]


# ══════════════════════════════════════════════════════════════════════════════
# KLD utilities (operate on v1 10-param vectors)
# ══════════════════════════════════════════════════════════════════════════════

def kld_params(p1, p2):
    """
    KL divergence KLD(p1 || p2) between two 3D multivariate-t distributions,
    estimated via Monte Carlo sampling from p1.

    Parameters
    ----------
    p1, p2 : array-like, shape (10,)
        [mu_pT, mu_logW, mu_logMET, S00, S01, S02, S11, S12, S22, nu]
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    mu1    = p1[0:3]
    Sigma1 = np.array([[p1[3],p1[4],p1[5]],
                       [p1[4],p1[6],p1[7]],
                       [p1[5],p1[7],p1[8]]])
    nu1    = p1[9]
    mu2    = p2[0:3]
    Sigma2 = np.array([[p2[3],p2[4],p2[5]],
                       [p2[4],p2[6],p2[7]],
                       [p2[5],p2[7],p2[8]]])
    nu2    = p2[9]

    L1 = np.linalg.cholesky(Sigma1)
    n  = 100_000
    samples = np.random.normal(size=(n, 3))
    v       = nu1 / 2
    gammas  = np.random.gamma(shape=v, size=n)
    Y_pre   = mu1 + np.sqrt(v / gammas)[:, None] * (samples @ L1.T)

    delta      = Y_pre - mu2
    Sigma2_inv = np.linalg.inv(Sigma2)
    mchis      = np.sum((delta @ Sigma2_inv) * delta, axis=1) / nu2
    ar         = np.log(1 + mchis) * (nu2 + 3) / 2

    exp2under1 = -np.mean(ar) + exp_terms(nu2, Sigma2)
    exp1under1 = (exp_terms(nu1, Sigma1)
                  - (nu1 + 3) / 2 * (sp.digamma((nu1 + 3) / 2) - sp.digamma(nu1 / 2)))
    return exp1under1 - exp2under1


def KLD(idx1, idx2):
    """KLD between two v1 grid points identified by 4-tuple indices."""
    _build_v1_interp()
    return kld_params(_v1_params[idx1], _v1_params[idx2])


def exp_terms(nu, sigma):
    """Log-normalisation constant of a 3D multivariate-t density."""
    return (sp.gammaln((3 + nu) / 2) - sp.gammaln(nu / 2)
            - 3 / 2 * np.log(np.pi * nu)
            - 1 / 2 * np.log(np.linalg.det(sigma)))


_OFFSETS = [
    (-1,  0,  0,  0), ( 1,  0,  0,  0),
    ( 0, -1,  0,  0), ( 0,  1,  0,  0),
    ( 0,  0, -1,  0), ( 0,  0,  1,  0),
    ( 0,  0,  0, -1), ( 0,  0,  0,  1),
]


def _kld_task(args):
    idx1, idx2 = args
    return KLD(idx1, idx2)


def neighbor_kld_grid(n_workers=10):
    """
    KLD from every interior point of the regression grid to its 8 axis-aligned
    neighbours.

    Returns
    -------
    result : np.ndarray, shape (N0-2, N1-2, N2-2, N3-2, 8)
    """
    _build_v1_interp()
    tasks, keys = [], []
    for i in range(len(_v1_mZ) - 2):
        for j in range(len(_v1_mRho) - 2):
            for k in range(len(_v1_rinv) - 2):
                for l in range(len(_v1_alphaD) - 2):
                    center = (i+1, j+1, k+1, l+1)
                    for n, (di, dj, dk, dl) in enumerate(_OFFSETS):
                        tasks.append((center, (center[0]+di, center[1]+dj,
                                               center[2]+dk, center[3]+dl)))
                        keys.append((i, j, k, l, n))

    shape = (len(_v1_mZ)-2, len(_v1_mRho)-2,
             len(_v1_rinv)-2, len(_v1_alphaD)-2, 8)
    result = np.empty(shape)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        imap = ex.map(_kld_task, tasks)
        if _has_tqdm:
            imap = tqdm(imap, total=len(tasks), desc='neighbor KLD')
        for (i, j, k, l, n), val in zip(keys, imap):
            result[i, j, k, l, n] = val
    return result


def corner_kld_grid(n_workers=10):
    """
    KLD from each of the 16 grid corners to every grid point.

    Returns
    -------
    result : np.ndarray, shape (2, 2, 2, 2, N0, N1, N2, N3)
    """
    _build_v1_interp()
    n0, n1, n2, n3 = (len(_v1_mZ), len(_v1_mRho),
                      len(_v1_rinv), len(_v1_alphaD))
    corner_idx = [[0, n0-1], [0, n1-1], [0, n2-1], [0, n3-1]]
    tasks, keys = [], []
    for c0 in range(2):
        for c1 in range(2):
            for c2 in range(2):
                for c3 in range(2):
                    p1 = (corner_idx[0][c0], corner_idx[1][c1],
                          corner_idx[2][c2], corner_idx[3][c3])
                    for i in range(n0):
                        for j in range(n1):
                            for k in range(n2):
                                for l in range(n3):
                                    tasks.append((p1, (i, j, k, l)))
                                    keys.append((c0, c1, c2, c3, i, j, k, l))

    result = np.empty((2, 2, 2, 2, n0, n1, n2, n3))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        imap = ex.map(_kld_task, tasks)
        if _has_tqdm:
            imap = tqdm(imap, total=len(tasks), desc='corner KLD')
        for key, val in zip(keys, imap):
            result[key] = val
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Legacy transform helpers (kept for v1 notebook compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def BoxCox(datap, l):
    """Box-Cox transform with exponent l (0 → log, else power)."""
    mask = (datap > 0) & (datap < np.inf)
    n_bad = len(datap) - int(np.sum(mask))
    if n_bad:
        print(f"{n_bad} data points <= 0 ignored in Box-Cox.")
    data = datap[mask]
    if l == 0:
        return np.log(data)
    elif 0 < l <= 1:
        return (data**l - 1) / l
    elif l == -1:
        return data
    else:
        raise ValueError('l must be in the range (0,1] or equal to 0 or -1')


def get_common_finite(data1, data2):
    """Return (data1, data2) filtered to rows where both are finite and positive."""
    mask = np.isfinite(data1) & np.isfinite(data2) & (data1 > 0) & (data2 > 0)
    return data1[mask], data2[mask]


def preprocess_data(data):
    """Drop rows with any entry <= 1e-6 or non-finite."""
    return data[np.all((data > 1e-6) & (data < np.inf), axis=1)]


def transform_data(data):
    """Box-Cox + gennorm CDF → standard-normal per column."""
    p_data = preprocess_data(data)
    n_obs  = p_data.shape[1]
    params = np.empty((n_obs, 4))
    tr     = np.empty(p_data.shape)
    for i in range(n_obs):
        x_bc, lam = st.boxcox(p_data[:, i])
        beta, loc, scale = st.gennorm.fit(
            x_bc, 2.0, loc=x_bc.mean(), scale=x_bc.std())
        params[i, :] = lam, beta, loc, scale
        u = np.clip(st.gennorm.cdf(x_bc, beta, loc=loc, scale=scale),
                    1e-10, 1 - 1e-10)
        tr[:, i] = st.norm.ppf(u)
    return tr, params


# ══════════════════════════════════════════════════════════════════════════════
# SVJ scan interpolator (svj_scan.npz)
# ══════════════════════════════════════════════════════════════════════════════

_svj_interp = None
_svj_meta   = {}   # corr_start, param_offsets, obs_names, n_obs


def svj_grid_bounds():
    """
    Return a dict {axis_name: vals_array} for each scan axis in the loaded NPZ.

    The set of axes (and their order) reflects whatever was in [scan] when the
    scan was run — no hardcoded assumption about which parameters were scanned.
    """
    svj        = _load_svj()
    axis_names = list(svj['axis_names'])
    return {name: np.array(svj[f'{name}_vals']) for name in axis_names}


def _build_svj_interp():
    global _svj_interp, _svj_meta
    if _svj_interp is not None:
        return

    svj        = _load_svj()
    axis_names = list(svj['axis_names'])
    axes       = tuple(np.array(svj[f'{name}_vals']) for name in axis_names)

    param_flat    = np.array(svj['param_flat'])
    param_offsets = np.array(svj['param_offsets'], dtype=int)
    corr_start    = int(svj['corr_start'])
    obs_names     = list(svj['obs_names'])
    n_obs         = len(obs_names)
    n_corr        = n_obs * (n_obs - 1) // 2

    _svj_meta = {
        'axis_names':    axis_names,
        'param_offsets': param_offsets,
        'corr_start':    corr_start,
        'obs_names':     obs_names,
        'n_obs':         n_obs,
        'n_corr':        n_corr,
    }

    _svj_interp = RegularGridInterpolator(
        axes,
        param_flat,
        method='linear',
        bounds_error=True,
    )


def interpolate_svj_params(params):
    """
    Return all interpolated SVJ scan parameters at the given physics point.

    Parameters
    ----------
    params : dict[str, float]
        Must contain a value for every scan axis that was used when building
        the NPZ (keys are the names from the [scan] section of the config).
        Extra keys are silently ignored.

    Returns
    -------
    (R_upper, flat_obs_params, param_offsets, obs_names)

    R_upper         : np.ndarray, shape (n_obs*(n_obs-1)//2,)
    flat_obs_params : np.ndarray
    param_offsets   : np.ndarray, shape (n_obs+1,)
    obs_names       : list of str
    """
    _build_svj_interp()
    axis_names = _svj_meta['axis_names']
    point      = np.array([[params[name] for name in axis_names]])
    p          = _svj_interp(point)[0]
    m          = _svj_meta

    cs      = m['corr_start']
    obs_p   = p[:cs]
    R_upper = p[cs:]
    return R_upper, obs_p, m['param_offsets'], m['obs_names']


def sample_svj_new(R_upper, flat_obs_params, param_offsets, obs_names,
                   n_samples=100_000, rng=None):
    """
    Draw samples from the SVJ Gaussian-copula model.

    Uses the observable inverse pipeline from src/observables.py.

    Parameters
    ----------
    R_upper         : array-like, shape (K*(K-1)//2,)
    flat_obs_params : array-like, shape (total_obs_params,)
    param_offsets   : array-like, shape (K+1,)
    obs_names       : list of str

    Returns
    -------
    X : np.ndarray, shape (n_samples, K)  — one column per observable in obs_names
    """
    from observables import OBSERVABLES, inverse_observable_col

    K       = len(obs_names)
    R_upper = np.asarray(R_upper)
    R = np.zeros((K, K))
    R[np.triu_indices(K, k=1)] = R_upper
    R += R.T
    np.fill_diagonal(R, 1.0)

    # Sample MVN(0, R) — Gaussian copula
    if rng is None:
        Z = np.random.multivariate_normal(np.zeros(K), R, size=n_samples)
    else:
        Z = rng.multivariate_normal(np.zeros(K), R, size=n_samples)

    # Map Gaussian marginal CDF N(0,1) → uniform
    u = np.clip(st.norm.cdf(Z), 1e-10, 1.0 - 1e-10)   # (n_samples, K)

    X = np.empty((n_samples, K))
    param_offsets   = np.asarray(param_offsets, dtype=int)
    flat_obs_params = np.asarray(flat_obs_params)

    for i, obs_name in enumerate(obs_names):
        obs_spec = OBSERVABLES[obs_name]
        pipeline = obs_spec['pipeline']
        dist     = obs_spec['distribution']
        p_start  = int(param_offsets[i])
        p_end    = int(param_offsets[i + 1])
        params   = tuple(flat_obs_params[p_start:p_end])
        X[:, i]  = inverse_observable_col(u[:, i], pipeline, dist, params)

    return X

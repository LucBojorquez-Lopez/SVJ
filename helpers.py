#some helpers functions

import numpy as np
import scipy.special as sp
import scipy.stats as st
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor
try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

regression = np.load('simulated/regression_scan.npz', allow_pickle = True)

# ── Build RegularGridInterpolator once at import time ─────────────────────────
_mZ_vals     = regression['mZ_vals']
_mRho_vals   = regression['mRho_vals']
_rinv_vals   = regression['rinv_vals']
_alphaD_vals = regression['alphaD_vals']
_params      = np.array(regression['params'])   # (8, 8, 8, 8, 10) — eagerly loaded into RAM

_interp = RegularGridInterpolator(
    (_mZ_vals, _mRho_vals, _rinv_vals, _alphaD_vals),
    _params,
    method='linear',
    bounds_error=True,   # raises ValueError if query is outside the grid
)

def interpolate_params(mZ, mRho, rinv, alphaD):
    """
    Return the 10 estimated SVJ parameters at (mZ, mRho, rinv, alphaD)
    via linear interpolation on the precomputed grid.

    Parameters
    ----------
    mZ, mRho, rinv, alphaD : float
        Physical parameters. Must lie within the grid ranges:
          mZ     in [500, 4000] GeV
          mRho   in [10, 30] GeV
          rinv   in [0.05, 0.75]
          alphaD in [0.1, 0.8]

    Returns
    -------
    params : np.ndarray, shape (10,)
        [mu_pT, mu_logW, mu_logMET, S00, S01, S02, S11, S12, S22, nu]
    """
    return _interp([[mZ, mRho, rinv, alphaD]])[0]

def kld_params(p1, p2):
    """
    KL divergence E_{p1}[log p1/p2] between two multivariate-t distributions.

    Parameters
    ----------
    p1, p2 : array-like, shape (10,)
        Parameter vectors [mu_pT, mu_logW, mu_logMET,
                           S00, S01, S02, S11, S12, S22, nu].
        p1 is the "true" distribution (we sample from it).

    Returns
    -------
    float
        KL divergence KLD(p1 || p2).
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    mu1 = p1[0:3]
    Sigma1 = np.array([[p1[3],p1[4],p1[5]],[p1[4],p1[6],p1[7]],[p1[5],p1[7],p1[8]]])
    nu1 = p1[9]
    mu2 = p2[0:3]
    Sigma2 = np.array([[p2[3],p2[4],p2[5]],[p2[4],p2[6],p2[7]],[p2[5],p2[7],p2[8]]])
    nu2 = p2[9]

    L1 = np.linalg.cholesky(Sigma1)
    n = 100000
    samples = np.random.normal(size = (n, 3))
    v = nu1/2
    gammas = np.random.gamma(shape = v, size = n)
    Y_pre = mu1 + np.sqrt(v/gammas)[:,None] * (samples @ L1.T)

    st = Y_pre - mu2
    Sigma2_inv = np.linalg.inv(Sigma2)
    mchis = np.sum((st @ Sigma2_inv) * st, axis=1) / nu2
    ar = np.log(1 + mchis) * (nu2 + 3)/2

    exp2under1 = -np.mean(ar) + exp_terms(nu2, Sigma2)
    exp1under1 = exp_terms(nu1, Sigma1) - (nu1 + 3)/2 * (sp.digamma((nu1 + 3)/2) - sp.digamma(nu1/2))

    return exp1under1 - exp2under1


def KLD(idx1, idx2):
    return kld_params(_params[idx1], _params[idx2])

def exp_terms(nu, sigma):
    return sp.gammaln((3 + nu)/2) - sp.gammaln(nu/2) - 3/2 * np.log(np.pi*nu) - 1/2 * np.log(np.linalg.det(sigma))


# ── Neighbor KLD grid ─────────────────────────────────────────────────────────

# Axis-aligned offsets: index n in the last axis of the output corresponds to:
#   0: mZ    - 1    1: mZ    + 1
#   2: mRho  - 1    3: mRho  + 1
#   4: rinv  - 1    5: rinv  + 1
#   6: alphaD- 1    7: alphaD+ 1
_OFFSETS = [
    (-1,  0,  0,  0), ( 1,  0,  0,  0),
    ( 0, -1,  0,  0), ( 0,  1,  0,  0),
    ( 0,  0, -1,  0), ( 0,  0,  1,  0),
    ( 0,  0,  0, -1), ( 0,  0,  0,  1),
]

def _kld_task(args):
    """Top-level worker: compute KLD(idx1 -> idx2). Must be module-level for pickling."""
    idx1, idx2 = args
    return KLD(idx1, idx2)

def neighbor_kld_grid(n_workers=10):
    """
    For every interior point of the 8x8x8x8 regression grid (indices 1..6 in
    each dimension), compute KLD from that point to each of its 8 axis-aligned
    neighbors.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes (default 10).

    Returns
    -------
    result : np.ndarray, shape (6, 6, 6, 6, 8)
        result[i, j, k, l, n] = KLD from interior point (i+1, j+1, k+1, l+1)
        to its n-th neighbor (see _OFFSETS for ordering).
    """
    # Build flat task list and a parallel index list to map results back
    tasks = []
    keys  = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                for l in range(6):
                    center = (i+1, j+1, k+1, l+1)
                    for n, (di, dj, dk, dl) in enumerate(_OFFSETS):
                        neighbor = (center[0]+di, center[1]+dj,
                                    center[2]+dk, center[3]+dl)
                        tasks.append((center, neighbor))
                        keys.append((i, j, k, l, n))

    result = np.empty((6, 6, 6, 6, 8))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        imap = executor.map(_kld_task, tasks)
        if _has_tqdm:
            imap = tqdm(imap, total=len(tasks), desc='neighbor KLD')
        for (i, j, k, l, n), val in zip(keys, imap):
            result[i, j, k, l, n] = val

    return result


# ── Corner-to-all KLD grid ────────────────────────────────────────────────────

def corner_kld_grid(n_workers=10):
    """
    For each of the 16 corners of the grid, compute KLD from that corner
    (as the "true" distribution, p1) to every grid point.

    Corner encoding: c_i = 0 → lower bound (index 0) along axis i,
                     c_i = 1 → upper bound (index -1) along axis i.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes (default 10).

    Returns
    -------
    result : np.ndarray, shape (2, 2, 2, 2, N_mZ, N_mRho, N_rinv, N_alphaD)
        result[c0, c1, c2, c3, i, j, k, l]
            = KLD(corner (c0,c1,c2,c3) → grid point (i,j,k,l))
    """
    n0 = len(_mZ_vals)
    n1 = len(_mRho_vals)
    n2 = len(_rinv_vals)
    n3 = len(_alphaD_vals)

    # Grid index of lower (0) and upper (1) corner along each axis
    corner_idx = [[0, n0 - 1], [0, n1 - 1], [0, n2 - 1], [0, n3 - 1]]

    tasks = []
    keys  = []
    for c0 in range(2):
        for c1 in range(2):
            for c2 in range(2):
                for c3 in range(2):
                    p1_idx = (corner_idx[0][c0], corner_idx[1][c1],
                              corner_idx[2][c2], corner_idx[3][c3])
                    for i in range(n0):
                        for j in range(n1):
                            for k in range(n2):
                                for l in range(n3):
                                    tasks.append((p1_idx, (i, j, k, l)))
                                    keys.append((c0, c1, c2, c3, i, j, k, l))

    result = np.empty((2, 2, 2, 2, n0, n1, n2, n3))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        imap = executor.map(_kld_task, tasks)
        if _has_tqdm:
            imap = tqdm(imap, total=len(tasks), desc='corner KLD')
        for key, val in zip(keys, imap):
            result[key] = val

    return result

def BoxCox(datap, l):
    mask = (datap > 0) & (datap < np.inf)
    print(len(datap) - np.sum(mask), "data points were <= 0 and will be ignored in the Box-Cox transformation")
    data = datap[mask]
    if l==0:
        return np.log(data)
    elif (l <= 1) and (l > 0):
        return (data**l - 1)/l
    elif l==-1:
        return data
    else:
        raise ValueError('l must be in the range (0,1) or equal to 0')
    
def get_common_finite(data1, data2):
    mask = np.isfinite(data1) & np.isfinite(data2) & (data1 > 0) & (data2 > 0)
    return data1[mask], data2[mask]

def preprocess_data(data):
    return data[np.all((data > 0) & (data < np.inf), axis = 1)]

def transform_data(data):
    p_data = preprocess_data(data)
    nobservables = p_data.shape[1]
    params  = np.empty((nobservables, 4))
    transformed_data = np.empty(p_data.shape)
    for i in range(nobservables):
        x_bc, lam = st.boxcox(p_data[:,i])
        beta, loc, scale = st.gennorm.fit(x_bc)
        params[i,:] = lam, beta, loc, scale
        cdf_vals = np.clip(st.gennorm.cdf(x_bc, beta, loc=loc, scale=scale), 1e-10, 1 - 1e-10)
        transformed_data[:,i] = st.norm.ppf(cdf_vals)
    
    return transformed_data, params


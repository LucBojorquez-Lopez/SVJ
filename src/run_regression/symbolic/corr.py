"""Positive-definite-by-construction parameterisation of a correlation matrix.

M8 measured the problem: predicting the 120 upper-triangle entries of R
independently, even with each entry pushed through `arctanh` so it lands inside
(-1, 1), produced a non-positive-definite matrix on **38 of 40** held-out
points.  Bounding entries individually says nothing about the assembled matrix,
because positive definiteness is a joint constraint.  Those 38 then had to be
repaired by eigenvalue clipping before they could be sampled from at all, which
means M8's symbolic scores were measured on repaired matrices rather than on
what the model predicted -- a benchmark of the repair, not of the model.

The fix is to regress in coordinates where every real vector maps to a valid
correlation matrix.  Take the Cholesky factor L of R, with L lower triangular
and unit-norm rows, and read off the *canonical partial correlations*

    z_ij  in (-1, 1),   i > j

via the standard recursion.  Any z in (-1, 1)^(K(K-1)/2) yields a valid R, so
regressing y_ij = arctanh(z_ij) over the unbounded reals cannot leave the
constraint set: tanh brings it back into (-1, 1), and the recursion turns that
into a positive-definite correlation matrix with unit diagonal.  No repair, no
clipping, nothing to explain away.

This is the same device Stan uses for its LKJ-parameterised correlation priors.
"""

import numpy as np


def corr_to_partials(R, eps=1e-12):
    """Correlation matrix -> canonical partial correlations, row-major i>j."""
    K = R.shape[0]
    # jitter only if needed: a fitted corrcoef matrix is PD but can sit close
    # enough to the boundary that Cholesky refuses it (M4 measured a median
    # minimum eigenvalue of ~1.5e-3).
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(R)
        R2 = (V * np.clip(w, 1e-10, None)) @ V.T
        d = np.sqrt(np.diag(R2))
        R2 = R2 / np.outer(d, d)
        np.fill_diagonal(R2, 1.0)
        L = np.linalg.cholesky(R2)
    z = np.empty(K * (K - 1) // 2)
    t = 0
    for i in range(1, K):
        acc = 0.0
        for j in range(i):
            denom = np.sqrt(max(1.0 - acc, eps))
            z[t] = np.clip(L[i, j] / denom, -1 + 1e-9, 1 - 1e-9)
            acc += L[i, j] ** 2
            t += 1
    return z


def partials_to_corr(z, K, ridge=1e-6, zmax=0.9995):
    """Canonical partial correlations -> a valid correlation matrix.

    Two guards, both needed in practice.

    `zmax` keeps |z| off 1.  Real fitted matrices do reach 0.9997, so this
    discards ~0.16% of genuine values -- accepted, because as |z| -> 1 the
    recursion drives the Cholesky diagonal to its floor and R becomes singular
    by construction, and a *predicted* z has no reason to respect the boundary
    as gracefully as a fitted one.

    `ridge` then shrinks toward the identity, (1-d)R + dI, whose eigenvalues are
    (1-d)*lambda_i + d >= d.  That buys a guaranteed PD margin of 1e-6, which
    matters because numpy's multivariate_normal warns and returns garbage for a
    matrix that is only PSD to machine precision -- exactly what saturated
    inputs produce.  The diagonal stays exactly 1, so it is still a correlation
    matrix, and correlations move by O(1e-6).
    """
    L = np.zeros((K, K))
    L[0, 0] = 1.0
    t = 0
    for i in range(1, K):
        acc = 0.0
        for j in range(i):
            zij = float(np.clip(z[t], -zmax, zmax))
            L[i, j] = zij * np.sqrt(max(1.0 - acc, 0.0))
            acc += L[i, j] ** 2
            t += 1
        L[i, i] = np.sqrt(max(1.0 - acc, 1e-12))
    R = L @ L.T
    if ridge > 0:
        R = (1.0 - ridge) * R + ridge * np.eye(K)
    np.fill_diagonal(R, 1.0)
    return R


def upper_to_full(R_upper, K):
    R = np.zeros((K, K))
    R[np.triu_indices(K, k=1)] = R_upper
    R += R.T
    np.fill_diagonal(R, 1.0)
    return R


def full_to_upper(R):
    return R[np.triu_indices(R.shape[0], k=1)]

#!/usr/bin/env python3
"""
interpolation_comparison.py
===========================
Compare five interpolation methods on the SVJ regression scan grid.

Methods:
  1. LinearNDInterpolator          (scipy, Delaunay triangulation)
  2. RBFInterpolator kernel=cubic  (scipy)
  3. RBFInterpolator kernel=gaussian, epsilon=1.0  (scipy)
  4. KNeighborsRegressor k=3       (sklearn)
  5. KNeighborsRegressor k=7       (sklearn)

For each of N_TRIALS trials:
  - Hold out 10% of the 4096 grid points at random.
  - Fit each method on the remaining 90%.
  - Predict at the held-out points.
  - Compute per-point relative squared error:
      d = sum_k [ (pred_k - true_k) / true_k ]^2   over 10 outputs k
  - Accumulate into flat arrays (NaN kept for linear convex-hull misses).

Output: interpolation_comparison.npz
  distances_linear   -- LinearNDInterpolator
  distances_rbf_cub  -- RBF cubic
  distances_rbf_gau  -- RBF gaussian (epsilon=1)
  distances_knn3     -- kNN k=3
  distances_knn7     -- kNN k=7
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from sklearn.neighbors import KNeighborsRegressor
import time

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_FILE    = 'simulated/regression_scan.npz'
OUTPUT_FILE  = 'simulated/interpolation_comparison.npz'
N_TRIALS     = 100
HOLDOUT_FRAC = 0.10
RNG_SEED     = 42
RBF_GAUSS_EPS = 1.0


def rel_sq_err(Y_pred, Y_true):
    """Per-point sum of squared relative errors across all 10 outputs."""
    rel = (Y_pred - Y_true) / Y_true   # (n, 10)
    return np.sum(rel ** 2, axis=1)    # (n,)


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {DATA_FILE} ...")
    d = np.load(DATA_FILE, allow_pickle=True)
    params      = d['params']       # (8, 8, 8, 8, 10)
    mZ_vals     = d['mZ_vals']
    mRho_vals   = d['mRho_vals']
    rinv_vals   = d['rinv_vals']
    alphaD_vals = d['alphaD_vals']

    # ── Flatten grid → (N, 4) inputs, (N, 10) outputs ─────────────────────────
    grid  = np.array(np.meshgrid(mZ_vals, mRho_vals, rinv_vals, alphaD_vals,
                                 indexing='ij'))          # (4, 8, 8, 8, 8)
    X_raw = grid.reshape(4, -1).T                         # (N, 4)
    Y     = params.reshape(-1, 10)                        # (N, 10)
    N     = len(X_raw)
    print(f"Grid: {N} points  x  10 outputs")

    # Drop NaN rows (should be none, but be safe)
    valid = ~np.any(np.isnan(Y), axis=1)
    if valid.sum() < N:
        print(f"WARNING: dropping {N - valid.sum()} NaN points.")
        X_raw, Y = X_raw[valid], Y[valid]

    # ── Normalise inputs to [0, 1] per axis ───────────────────────────────────
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X = (X_raw - X_min) / (X_max - X_min)   # (N, 4)

    n_holdout = max(1, int(round(len(X) * HOLDOUT_FRAC)))
    n_train   = len(X) - n_holdout
    print(f"Each trial: {n_train} train / {n_holdout} held-out")
    print(f"Running {N_TRIALS} trials ...\n")
    print(f"  {'Trial':>7}  {'lin NaN':>7}  {'ETA':>8}")

    rng = np.random.default_rng(RNG_SEED)

    dist_linear  = []
    dist_rbf_cub = []
    dist_rbf_gau = []
    dist_knn3    = []
    dist_knn7    = []

    t0 = time.time()
    for trial in range(N_TRIALS):
        # ── Split ──────────────────────────────────────────────────────────────
        idx_holdout = rng.choice(len(X), size=n_holdout, replace=False)
        idx_train   = np.setdiff1d(np.arange(len(X)), idx_holdout)

        X_tr, Y_tr   = X[idx_train],   Y[idx_train]
        X_ho, Y_true = X[idx_holdout], Y[idx_holdout]

        # ── 1. Linear (Delaunay) ───────────────────────────────────────────────
        lin        = LinearNDInterpolator(X_tr, Y_tr)
        Y_lin      = lin(X_ho)
        n_nan_lin  = int(np.sum(np.isnan(Y_lin[:, 0])))

        # ── 2. RBF cubic ───────────────────────────────────────────────────────
        Y_rbf_cub = RBFInterpolator(X_tr, Y_tr, kernel='cubic')(X_ho)

        # ── 3. RBF gaussian ────────────────────────────────────────────────────
        Y_rbf_gau = RBFInterpolator(X_tr, Y_tr,
                                    kernel='gaussian',
                                    epsilon=RBF_GAUSS_EPS)(X_ho)

        # ── 4 & 5. kNN k=3 and k=7 ────────────────────────────────────────────
        Y_knn3 = KNeighborsRegressor(n_neighbors=3).fit(X_tr, Y_tr).predict(X_ho)
        Y_knn7 = KNeighborsRegressor(n_neighbors=7).fit(X_tr, Y_tr).predict(X_ho)

        # ── Accumulate distances ───────────────────────────────────────────────
        dist_linear .extend(rel_sq_err(Y_lin,     Y_true).tolist())
        dist_rbf_cub.extend(rel_sq_err(Y_rbf_cub, Y_true).tolist())
        dist_rbf_gau.extend(rel_sq_err(Y_rbf_gau, Y_true).tolist())
        dist_knn3   .extend(rel_sq_err(Y_knn3,    Y_true).tolist())
        dist_knn7   .extend(rel_sq_err(Y_knn7,    Y_true).tolist())

        # ── Progress ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        eta     = elapsed / (trial + 1) * (N_TRIALS - trial - 1)
        h, rem  = divmod(int(eta), 3600)
        m, s    = divmod(rem, 60)
        print(f"  {trial+1:>{len(str(N_TRIALS))}}/{N_TRIALS}"
              f"  |  lin NaN: {n_nan_lin:>3}"
              f"  |  ETA {h:02d}:{m:02d}:{s:02d}",
              flush=True)

    # ── Convert and save ───────────────────────────────────────────────────────
    arrs = {
        'distances_linear'  : np.array(dist_linear),
        'distances_rbf_cub' : np.array(dist_rbf_cub),
        'distances_rbf_gau' : np.array(dist_rbf_gau),
        'distances_knn3'    : np.array(dist_knn3),
        'distances_knn7'    : np.array(dist_knn7),
    }

    np.savez(OUTPUT_FILE, **arrs)
    total_time = time.time() - t0
    print(f"\nSaved to {OUTPUT_FILE}  ({total_time/60:.1f} min total)")

    print(f"\nQuick stats (NaNs excluded):")
    labels = [
        ('linear',       arrs['distances_linear']),
        ('rbf cubic',    arrs['distances_rbf_cub']),
        ('rbf gaussian', arrs['distances_rbf_gau']),
        ('knn k=3',      arrs['distances_knn3']),
        ('knn k=7',      arrs['distances_knn7']),
    ]
    for label, arr in labels:
        a = arr[~np.isnan(arr)]
        print(f"  {label:14s}  n={len(a):>6}  median={np.median(a):.4g}"
              f"  mean={np.mean(a):.4g}  p95={np.percentile(a, 95):.4g}")


if __name__ == '__main__':
    main()

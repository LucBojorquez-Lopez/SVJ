#!/usr/bin/env python3
"""
kld_benchmark.py
================
Baseline KLD: how different are two randomly-chosen points in the scanned
parameter space?  Provides a reference scale for the interpolation KLD values
in simulated/validation_results.npy.

Two sampling modes (both reported):
  grid   -- draw two random grid indices uniformly from the 8^4 scan; always
             produces valid (PD) parameter vectors.
  uniform -- sample each of the 10 parameters uniformly in [min, max]; pairs
             where either Sigma is not positive-definite are rejected.

For each valid pair the "true" distribution is chosen at random (50/50).

Usage:
    python kld_benchmark.py [N_pairs]   (default: 200)

Output:
    simulated/kld_benchmark_grid.npy    -- raw KLDs from grid sampling
    simulated/kld_benchmark_uniform.npy -- raw KLDs from uniform sampling
"""

import sys
import numpy as np
import scipy.special as sp

# ── Load scan data ─────────────────────────────────────────────────────────────
scan   = np.load('simulated/regression_scan.npz', allow_pickle=True)
params = np.array(scan['params'])   # (8, 8, 8, 8, 10)
flat   = params.reshape(-1, 10)     # (4096, 10)

p_min = flat.min(axis=0)
p_max = flat.max(axis=0)

NAMES = ['mu_pT', 'mu_logW', 'mu_logMET',
         'S00', 'S01', 'S02', 'S11', 'S12', 'S22', 'nu']

print("Parameter ranges across the scan grid:")
for i, n in enumerate(NAMES):
    print(f"  {n:12s}  [{p_min[i]:.4g}, {p_max[i]:.4g}]")
print()


# ── KLD (mirrors helpers.kld_params) ──────────────────────────────────────────

def sigma_from_p(p):
    return np.array([[p[3], p[4], p[5]],
                     [p[4], p[6], p[7]],
                     [p[5], p[7], p[8]]])

def is_pd(p):
    try:
        np.linalg.cholesky(sigma_from_p(p))
        return True
    except np.linalg.LinAlgError:
        return False

def kld_params(p1, p2, n_mc=100_000):
    mu1, S1, nu1 = p1[0:3], sigma_from_p(p1), p1[9]
    mu2, S2, nu2 = p2[0:3], sigma_from_p(p2), p2[9]

    L1     = np.linalg.cholesky(S1)
    z      = np.random.normal(size=(n_mc, 3))
    v      = nu1 / 2
    g      = np.random.gamma(shape=v, size=n_mc)
    Y      = mu1 + np.sqrt(v / g)[:, None] * (z @ L1.T)

    S2inv  = np.linalg.inv(S2)
    d      = Y - mu2
    mchi   = np.sum((d @ S2inv) * d, axis=1) / nu2
    ar     = np.log1p(mchi) * (nu2 + 3) / 2

    def ent(nu, S):
        return (sp.gammaln((3 + nu) / 2) - sp.gammaln(nu / 2)
                - 1.5 * np.log(np.pi * nu) - 0.5 * np.log(np.linalg.det(S)))

    exp2 = -np.mean(ar) + ent(nu2, S2)
    exp1 =  ent(nu1, S1) - (nu1 + 3) / 2 * (sp.digamma((nu1 + 3) / 2)
                                               - sp.digamma(nu1 / 2))
    return exp1 - exp2


# ── Benchmark ─────────────────────────────────────────────────────────────────

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng()

# ── Mode 1: grid sampling ─────────────────────────────────────────────────────
print(f"Mode 1 — grid sampling  ({N} pairs from 4096 scan points)")
klds_grid = []
for i in range(N):
    ia, ib = rng.integers(0, len(flat), size=2)
    a, b   = flat[ia].copy(), flat[ib].copy()
    if rng.random() < 0.5:
        a, b = b, a
    klds_grid.append(kld_params(a, b))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{N}  running mean: {np.mean(klds_grid):.4f}", flush=True)

klds_grid = np.array(klds_grid)
np.save('simulated/kld_benchmark_grid.npy', klds_grid)

print(f"\n  mean   : {klds_grid.mean():.4f}")
print(f"  std    : {klds_grid.std():.4f}")
print(f"  median : {np.median(klds_grid):.4f}")
print(f"  p5/p95 : {np.percentile(klds_grid, 5):.4f} / {np.percentile(klds_grid, 95):.4f}")
print(f"  min/max: {klds_grid.min():.4f} / {klds_grid.max():.4f}")

# ── Mode 2: uniform sampling in 10D box ──────────────────────────────────────
print(f"\nMode 2 — uniform sampling in [min, max]^10  (attempting {N} valid pairs)")
klds_uni  = []
n_tried   = 0
n_invalid = 0
while len(klds_uni) < N:
    a = rng.uniform(p_min, p_max)
    b = rng.uniform(p_min, p_max)
    n_tried += 2
    if not is_pd(a) or not is_pd(b):
        n_invalid += 2
        continue
    if rng.random() < 0.5:
        a, b = b, a
    klds_uni.append(kld_params(a, b))
    done = len(klds_uni)
    if done % 20 == 0:
        print(f"  {done}/{N}  running mean: {np.mean(klds_uni):.4f}"
              f"  (rejection rate so far: {n_invalid/n_tried:.1%})", flush=True)

klds_uni = np.array(klds_uni)
np.save('simulated/kld_benchmark_uniform.npy', klds_uni)

print(f"\n  Rejection rate (non-PD Sigma): {n_invalid/n_tried:.1%}")
print(f"  mean   : {klds_uni.mean():.4f}")
print(f"  std    : {klds_uni.std():.4f}")
print(f"  median : {np.median(klds_uni):.4f}")
print(f"  p5/p95 : {np.percentile(klds_uni, 5):.4f} / {np.percentile(klds_uni, 95):.4f}")
print(f"  min/max: {klds_uni.min():.4f} / {klds_uni.max():.4f}")

print("\nSaved:")
print("  simulated/kld_benchmark_grid.npy")
print("  simulated/kld_benchmark_uniform.npy")

#!/usr/bin/env python3
"""
validate_fit.py
===============
Sanity check on transform and distribution choices using a single TSV file.
No PYTHIA calls are made — this validates only the parametric fitting step.

For each selected observable the script:
  1. Fits the full pipeline (transforms + distribution) to all valid events.
  2. Draws N1 independent marginal samples from the fitted model.
  3. Computes Jensen-Shannon distance for two comparisons:
       js_fit      : model samples  vs  an evaluation subset of the data (N2 events)
       js_baseline : two other disjoint evaluation subsets compared to each other

The fit uses ALL valid events (including the evaluation subsets), so this is
an in-sample quality check — it tests whether the parametric family can
represent the data shape at all, not generalisation.  js_baseline gives the
statistical floor at sample size N2.  If js_fit ≈ js_baseline the fit is as
good as the data allows; if js_fit >> js_baseline the transform or
distribution is a poor match for the observable's shape.

Requires at least 3 × N2 valid events in the TSV after range-check filtering.

Usage
-----
python src/run_regression/validate_fit.py \\
    simulated/tsv/jets_default.tsv \\
    [--obs leadVisPt,MET,tau1,tau2,tau3] \\
    [--N1 50000] [--N2 5000] [--bins 80] [--seed 0] \\
    --out simulated/svj/validation_fit.npz

Output NPZ keys
---------------
js_fit        (n_obs,)      JS distance: model vs truth subset
js_baseline   (n_obs,)      JS distance: two truth halves (statistical floor)
obs_names     (n_obs,)      observable names
fitted_params (total_obs,)  flat fitted parameter vector (same layout as scan)
param_offsets (n_obs+1,)    start indices into fitted_params per observable
N1, N2, bins  int scalars
"""

import sys
import argparse
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from observables import (
    OBSERVABLES, DEFAULT_SCAN,
    load_tsv, event_valid_mask,
    fit_observable_col, inverse_observable_col,
    param_offsets as obs_param_offsets,
    validate_scan_selection,
)
from _val_utils import js_per_obs


def main():
    ap = argparse.ArgumentParser(
        description='Fit-quality sanity check from a single TSV file.')
    ap.add_argument('tsv',
                    help='Path to jets_default.tsv (or any TSV from the binary)')
    ap.add_argument('--obs', default=None,
                    help='Comma-separated observable names (default: DEFAULT_SCAN)')
    ap.add_argument('--N1', type=int, default=50_000,
                    help='Model samples to draw per observable (default: 50000)')
    ap.add_argument('--N2', type=int, default=5_000,
                    help='Events per evaluation set; need >= 3×N2 valid events'
                         ' (default: 5000)')
    ap.add_argument('--bins', type=int, default=80,
                    help='Histogram bins for JS distance (default: 80)')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for shuffling and sampling (default: 0)')
    ap.add_argument('--out', required=True,
                    help='Output NPZ path')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Observable selection ────────────────────────────────────────────────────
    if args.obs:
        obs_selection = [s.strip() for s in args.obs.split(',')]
    else:
        obs_selection = list(DEFAULT_SCAN)
    validate_scan_selection(obs_selection)
    n_obs   = len(obs_selection)
    offsets = obs_param_offsets(obs_selection)
    print(f"Observables ({n_obs}): {obs_selection}")

    # ── Load TSV and filter ────────────────────────────────────────────────────
    print(f"Loading {args.tsv} ...", flush=True)
    data, col_map = load_tsv(args.tsv)
    mask, n_disc  = event_valid_mask(data, obs_selection, col_map)
    valid = data[mask]
    n_valid = len(valid)
    print(f"  {n_valid:,} / {len(data):,} events pass range checks "
          f"(discarded per-obs: {list(map(int, n_disc))})")

    need = 3 * args.N2
    if n_valid < need:
        print(f"ERROR: need >= {need} valid events (3 × N2={args.N2}), "
              f"got {n_valid}.  Reduce --N2 or use a larger TSV.")
        sys.exit(1)

    # ── Shuffle and partition ──────────────────────────────────────────────────
    # Fit uses ALL valid events (best parameter estimates).
    # Three non-overlapping evaluation subsets of size N2:
    #   truth  — compared against model samples
    #   half_A, half_B — compared against each other (baseline)
    idx    = rng.permutation(n_valid)
    valid  = valid[idx]
    truth  = valid[:args.N2]
    half_A = valid[args.N2:  2 * args.N2]
    half_B = valid[2 * args.N2: 3 * args.N2]

    # ── Fit each observable and collect samples ────────────────────────────────
    all_fitted = np.empty(int(offsets[-1]))
    model_cols = np.empty((args.N1,  n_obs))
    truth_cols = np.empty((args.N2, n_obs))
    halfA_cols = np.empty((args.N2, n_obs))
    halfB_cols = np.empty((args.N2, n_obs))

    for i, obs_name in enumerate(obs_selection):
        obs_spec = OBSERVABLES[obs_name]
        tsv_col  = col_map[obs_spec['col']]

        x_fit = valid[:, tsv_col]
        x_fit = x_fit[np.isfinite(x_fit)]   # guard against any stray non-finite

        _, fitted = fit_observable_col(
            x_fit, obs_spec['pipeline'], obs_spec['distribution'])

        p0, p1 = int(offsets[i]), int(offsets[i + 1])
        all_fitted[p0:p1] = fitted

        # Draw N1 independent marginal samples (no copula; marginals only)
        u = rng.uniform(0.0, 1.0, args.N1)
        model_cols[:, i] = inverse_observable_col(
            u, obs_spec['pipeline'], obs_spec['distribution'], fitted)

        truth_cols[:, i] = truth[:, tsv_col]
        halfA_cols[:, i] = half_A[:, tsv_col]
        halfB_cols[:, i] = half_B[:, tsv_col]

        print(f"  [{i + 1:>{len(str(n_obs))}}/{n_obs}] {obs_name:<20s}  "
              f"params = {[f'{p:.4g}' for p in fitted]}", flush=True)

    # ── JS distances ───────────────────────────────────────────────────────────
    js_fit      = js_per_obs(model_cols, truth_cols,  obs_selection, bins=args.bins)
    js_baseline = js_per_obs(halfA_cols, halfB_cols, obs_selection, bins=args.bins)

    print()
    col_w = max(len(n) for n in obs_selection) + 2
    print(f"{'Observable':<{col_w}}  {'js_fit':>8}  {'js_baseline':>11}  ratio")
    print('-' * (col_w + 34))
    for name, jf, jb in zip(obs_selection, js_fit, js_baseline):
        ratio = jf / jb if jb > 1e-9 else float('inf')
        print(f"{name:<{col_w}}  {jf:8.4f}  {jb:11.4f}  {ratio:5.2f}x")

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        js_fit        = js_fit,
        js_baseline   = js_baseline,
        obs_names     = np.array(obs_selection, dtype=object),
        fitted_params = all_fitted,
        param_offsets = offsets,
        N1            = np.int64(args.N1),
        N2            = np.int64(args.N2),
        bins          = np.int64(args.bins),
    )
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()

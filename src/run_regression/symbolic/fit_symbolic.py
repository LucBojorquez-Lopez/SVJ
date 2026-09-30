#!/usr/bin/env python3
"""Fit a Design A model: scan NPZ -> one expression per parameter component.

    python src/run_regression/symbolic/fit_symbolic.py simulated/svj/svj_scan.npz \
        --out /tmp/model.json --max-terms 8

`--coarsen` trains on a strided subgrid, which is what holdout.py uses to make
the incumbent and the challenger compete on identical, genuinely unseen points.

NOTE on which scan to train against.  M6 in docs/symbolic-regression.md found
the committed svj_scan.npz carries 184 parameters where today's observables.py
produces 186 -- maxMuPt and dPhiMETclose have each since gained a point-mass
weight.  Training on the committed scan is fine for developing the method, but
a model meant for the live sampling path must be trained on a scan produced by
the current code.  This script reads param_offsets from the NPZ, so it stays
correct either way; it just cannot invent the two parameters that are missing.

LOGIN-NODE COURTESY.  Default n_workers is 4, not nproc, and deliberately so.
docs/lxplus.md section 2 already says this for the dependency build ("-j4, not
-j$(nproc): lxplus login nodes are shared and routinely sit above a load
average of their core count") and it applies with equal force here.  A 12-worker
run of this script triggered an automated CPU-pressure warning from the LxPlus
service -- some_avg300 at 68.6%, with these processes named as the top
contributor.  For anything heavier than a quick check, submit to Condor instead
of raising this number.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corr as corrmod                                     # noqa: E402
from basis import build_features, choose_link, fit_target   # noqa: E402
from model import SymbolicModel                       # noqa: E402

DERIVED = {
    'log_mZ': 'np.log(mZ)',
    'mPi':    'mPiOverLambda * LambdaDQCD',
    'mRho':   'LambdaDQCD * (5.76 + 1.5 * mPiOverLambda ** 2) ** 0.5',
    'mq':     'LambdaDQCD * (mPiOverLambda / 5.5) ** 2',
}


def load_grid(npz_path, coarsen=None):
    """Return (points, Y, meta) for a scan NPZ, optionally strided."""
    d = np.load(npz_path, allow_pickle=True)
    axis_names = [str(a) for a in d['axis_names']]
    vals = [np.asarray(d[f'{a}_vals'], dtype=float) for a in axis_names]
    pf   = np.asarray(d['param_flat'])
    if coarsen:
        sl = tuple(slice(None, None, int(s)) for s in coarsen)
        vals = [v[s] for v, s in zip(vals, sl)]
        pf   = pf[sl]
    grid = np.meshgrid(*[np.arange(len(v)) for v in vals], indexing='ij')
    idx  = np.stack([g.ravel() for g in grid], axis=1)
    points = [{a: float(vals[k][idx[n, k]]) for k, a in enumerate(axis_names)}
              for n in range(len(idx))]
    Y = pf.reshape(-1, pf.shape[-1])
    meta = dict(axis_names=axis_names,
                obs_names=[str(o) for o in d['obs_names']],
                param_offsets=[int(x) for x in d['param_offsets']],
                corr_start=int(d['corr_start']),
                n_params=int(pf.shape[-1]),
                axis_vals=vals)
    return points, Y, meta


_G = {}


def _init(F, term_names, Y, max_terms, seed, selector, links):
    _G.update(F=F, tn=term_names, Y=Y, mt=max_terms, seed=seed, sel=selector,
              links=links)


def _one(j):
    return j, fit_target(_G['F'], _G['tn'], _G['Y'][:, j],
                         max_terms=_G['mt'], seed=_G['seed'],
                         selector=_G['sel'], link=_G['links'][j])


def fit_model(points, Y, meta, max_terms=8, extra_inputs=(), pairwise=True,
              n_workers=4, seed=0, quiet=False, selector='omp',
              corr_param='partial'):
    """Fit every component; returns a SymbolicModel."""
    axis_names = meta['axis_names']
    P = np.array([[p[a] for a in axis_names] for p in points], dtype=float)
    names = list(axis_names)
    if extra_inputs:
        env = {a: P[:, i] for i, a in enumerate(axis_names)}
        env['np'] = np
        cols = [P]
        for nm in extra_inputs:
            cols.append(np.asarray(eval(DERIVED.get(nm, nm),
                                        {'__builtins__': {}}, env),
                                   dtype=float).reshape(-1, 1))
            names.append(nm)
        P = np.hstack(cols)
    scale = {'lo': P.min(0).tolist(), 'hi': P.max(0).tolist()}
    F, term_names = build_features(P, names, pairwise=pairwise, scale=scale)
    if not quiet:
        print(f"  training points {len(points)}, basis terms {len(term_names)}, "
              f"targets {Y.shape[1]}")

    cs = meta['corr_start']
    K = len(meta['obs_names'])

    # Re-express the correlation block in canonical-partial coordinates before
    # fitting.  M8 found per-entry regression produced a non-PD matrix on 38 of
    # 40 held-out points; here every real vector maps back to a valid
    # correlation matrix, so nothing needs repairing afterwards.  See corr.py.
    Y = np.array(Y, dtype=float, copy=True)
    if corr_param == 'partial':
        for n in range(Y.shape[0]):
            if np.all(np.isfinite(Y[n, cs:])):
                z = corrmod.corr_to_partials(corrmod.upper_to_full(Y[n, cs:], K))
                Y[n, cs:] = np.arctanh(np.clip(z, -0.9995, 0.9995))

    links = [choose_link(Y[:, j], is_corr=False) if j >= cs
             else choose_link(Y[:, j], is_corr=False) for j in range(Y.shape[1])]
    # The partial-correlation coordinates are already unbounded reals, so they
    # take the identity link -- pushing them through another arctanh would be
    # wrong, not merely redundant.
    if corr_param == 'partial':
        for j in range(cs, Y.shape[1]):
            links[j] = 'identity'
    if not quiet:
        from collections import Counter
        print(f"  links chosen: {dict(Counter(links))}")
    targets = [None] * Y.shape[1]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init,
                             initargs=(F, term_names, Y, max_terms, seed,
                                       selector, links)) as ex:
        for j, res in ex.map(_one, range(Y.shape[1]), chunksize=4):
            targets[j] = res

    spec = dict(axis_names=axis_names, obs_names=meta['obs_names'],
                param_offsets=meta['param_offsets'],
                corr_start=meta['corr_start'], n_params=meta['n_params'],
                pairwise=pairwise, extra_inputs=list(extra_inputs),
                derived_exprs=DERIVED, targets=targets, scale=scale,
                corr_param=corr_param)
    return SymbolicModel(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scan_npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-terms', type=int, default=8)
    ap.add_argument('--coarsen', default=None,
                    help='comma-separated stride per axis, e.g. 2,2,1,1,1,1')
    ap.add_argument('--extra-inputs', default='',
                    help=f'comma-separated, from {sorted(DERIVED)}')
    ap.add_argument('--no-pairwise', action='store_true')
    ap.add_argument('--n-workers', type=int, default=4,
                    help='keep small on a login node; see the module docstring')
    ap.add_argument('--selector', choices=('omp','lasso'), default='omp')
    ap.add_argument('--corr-param', choices=('partial','entry'), default='partial',
                    help="'partial' is PD by construction; 'entry' reproduces "
                         'the broken behaviour measured in M8')
    a = ap.parse_args()

    coarsen = [int(x) for x in a.coarsen.split(',')] if a.coarsen else None
    extra = [x.strip() for x in a.extra_inputs.split(',') if x.strip()]
    points, Y, meta = load_grid(a.scan_npz, coarsen)
    print(f"Design A fit: {a.scan_npz}")
    m = fit_model(points, Y, meta, max_terms=a.max_terms, extra_inputs=extra,
                  pairwise=not a.no_pairwise, n_workers=a.n_workers,
                  selector=a.selector, corr_param=a.corr_param)
    m.save(a.out)

    r2 = np.array([t['r2'] for t in m.targets])
    deg = sum(t.get('degenerate', False) for t in m.targets)
    nt = np.array([len(t['terms']) for t in m.targets])
    cs = m.corr_start
    print(f"\n  in-sample R^2  median {np.median(r2):.4f}   "
          f"marginal {np.median(r2[:cs]):.4f}   corr {np.median(r2[cs:]):.4f}")
    print(f"  R^2 < 0.5 on {int((r2 < 0.5).sum())}/{len(r2)} components; "
          f"degenerate (constant) {deg}")
    print(f"  terms per component: median {int(np.median(nt))}, max {int(nt.max())}")
    print(f"  saved -> {a.out}")
    worst = np.argsort(r2)[:3]
    print("\n  weakest components (these are the ones to distrust):")
    for j in worst:
        print(f"    [{j}] R^2={r2[j]:.3f}  {m.describe(j)[:100]}")


if __name__ == '__main__':
    main()

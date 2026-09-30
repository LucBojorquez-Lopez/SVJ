#!/usr/bin/env python3
"""The cheap benchmark: coarsen the grid, then compete on the points left out.

Why not just run validate_production.py?  Because it re-runs PYTHIA three times
per validation point -- roughly 330 CPU-hours for its 2000 points -- which is
far too slow to tune a basis library against, and would compete with the
production scan for farm slots.

So: train BOTH methods on a strided subgrid and evaluate BOTH at grid points
they never saw, where the true fitted parameter vector is already known.  That
isolates exactly the quantity Design A is trying to improve -- interpolation
error, with per-point fit quality held fixed -- and costs no simulation.

Scoring is in distribution space, not parameter space, because parameter error
is not what anyone cares about: a large error in a weakly-identified shape
parameter may not move the distribution at all, and a small one in a location
parameter certainly does.  At each held-out point we sample from the true
fitted parameters and from each method's prediction, and compare with the same
js_per_obs / mmd_rbf used by the shipped validation scripts.

`js_floor` is the reference that makes the numbers readable: two independent
samples drawn from the SAME true parameters.  It is the noise introduced by
sampling at this n, so a method sitting at js_floor is perfect as far as this
benchmark can see.

    python src/run_regression/symbolic/holdout.py simulated/svj/svj_scan.npz \
        --coarsen 2,2,1,1,1,1 --n-eval 200 --out /tmp/holdout.npz

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
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # _val_utils
sys.path.insert(0, str(HERE.parent.parent))   # src: helpers, observables

from fit_symbolic import DERIVED, fit_model, load_grid   # noqa: E402

_W = {}


def _init(obs_names, offsets, n_samples, seed):
    _W.update(obs=list(obs_names), off=np.asarray(offsets, int),
              n=n_samples, seed=seed)


def _sample(flat, tag):
    """Draw from a predicted parameter vector, repairing R first.

    Both methods get the same treatment, deliberately.  A predicted correlation
    matrix can be non-PD -- Design A predicts its 120 entries independently, and
    linear interpolation on a COARSE grid can extrapolate outside the convex
    hull that kept it safe in M4 -- and numpy's multivariate_normal silently
    warns and returns garbage rather than failing.  Scoring one method on
    garbage would make this benchmark meaningless, so repair both and report how
    often each needed it: the repair rate is itself a quality signal.
    """
    import helpers
    from model import _nearest_pd_corr
    obs, off, n = _W['obs'], _W['off'], _W['n']
    cs, K = int(off[-1]), len(obs)
    Ru = np.clip(np.asarray(flat[cs:], float), -0.999999, 0.999999)
    R = np.zeros((K, K)); R[np.triu_indices(K, k=1)] = Ru
    R += R.T; np.fill_diagonal(R, 1.0)
    R, repaired = _nearest_pd_corr(R)
    Ru = R[np.triu_indices(K, k=1)]
    rng = np.random.default_rng(abs(hash(tag)) % (2 ** 32))
    X = helpers.sample_svj_new(Ru, flat[:cs], off, obs, n_samples=n, rng=rng)
    return X[np.isfinite(X).all(axis=1)], repaired


def _score(args):
    """One held-out point: score linear and symbolic against the truth."""
    i, true_flat, lin_flat, sym_flat = args
    from _val_utils import js_per_obs, mmd_rbf
    obs = _W['obs']
    rng = np.random.default_rng(1000 + i)
    try:
        T1, _   = _sample(true_flat, f't1{i}')
        T2, _   = _sample(true_flat, f't2{i}')
        L, rl   = _sample(lin_flat,  f'l{i}')
        S, rs   = _sample(sym_flat,  f's{i}')
        nmin = min(len(T1), len(T2), len(L), len(S))
        if nmin < 500:
            # Report WHICH method collapsed instead of dropping the point
            # silently -- a method that cannot produce finite draws is failing,
            # and that is a result, not a missing data point.
            return dict(dropped=True, n_true=min(len(T1), len(T2)),
                        n_lin=len(L), n_sym=len(S))
        return dict(dropped=False,
            js_floor=js_per_obs(T1, T2, obs), js_lin=js_per_obs(L, T1, obs),
            js_sym=js_per_obs(S, T1, obs),
            mmd_floor=mmd_rbf(T1, T2, n_sub=1500, rng=rng),
            mmd_lin=mmd_rbf(L, T1, n_sub=1500, rng=rng),
            mmd_sym=mmd_rbf(S, T1, n_sub=1500, rng=rng),
            rep_lin=rl, rep_sym=rs)
    except Exception as e:
        return dict(dropped=True, error=str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scan_npz')
    ap.add_argument('--coarsen', default='2,2,1,1,1,1')
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--n-samples', type=int, default=10000)
    ap.add_argument('--max-terms', type=int, default=8)
    ap.add_argument('--extra-inputs', default='')
    ap.add_argument('--n-workers', type=int, default=4,
                    help='keep small on a login node; see the module docstring')
    ap.add_argument('--selector', choices=('omp','lasso'), default='omp')
    ap.add_argument('--regime', choices=('interior','extrapolate'),
                    default='interior',
                    help='score inside the training hull, or outside it')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    coarsen = [int(x) for x in a.coarsen.split(',')]
    extra = [x.strip() for x in a.extra_inputs.split(',') if x.strip()]

    d = np.load(a.scan_npz, allow_pickle=True)
    axis_names = [str(x) for x in d['axis_names']]
    full_vals = [np.asarray(d[f'{ax}_vals'], float) for ax in axis_names]
    pf = np.asarray(d['param_flat'])
    obs_names = [str(o) for o in d['obs_names']]
    offsets = np.asarray(d['param_offsets'], int)

    print(f"coarsen {coarsen}: "
          f"{'x'.join(str(len(v)) for v in full_vals)} -> "
          f"{'x'.join(str(len(v[::s])) for v, s in zip(full_vals, coarsen))}")

    # ── challenger: symbolic, trained on the coarse subgrid only ──────────────
    tr_points, tr_Y, meta = load_grid(a.scan_npz, coarsen)
    print("fitting symbolic model on the coarse subgrid")
    sym = fit_model(tr_points, tr_Y, meta, max_terms=a.max_terms,
                    extra_inputs=extra, n_workers=a.n_workers, seed=a.seed,
                    selector=a.selector)

    # ── incumbent: linear interpolation on the same coarse subgrid ────────────
    sl = tuple(slice(None, None, s) for s in coarsen)
    lin = RegularGridInterpolator([v[::s] for v, s in zip(full_vals, coarsen)],
                                  pf[sl], bounds_error=False, fill_value=None)

    # ── held-out points: on the full grid, not on the coarse one ──────────────
    # Interior vs extrapolating held-out points, kept apart deliberately.
    #
    # Striding an axis of length 4 by 2 trains on indices {0, 2} and leaves
    # {1, 3} -- and index 3 is BEYOND the last training point, so scoring it
    # measures extrapolation.  Mixing the two silently was a real flaw in the
    # first all-axes run: linear collapsed to 119 finite draws of 8000 because
    # RegularGridInterpolator was extrapolating with fill_value=None, and the
    # resulting invalid parameter vectors were charged against "interpolation".
    #
    # They are also the more interesting comparison apart than together: local
    # interpolation has no defence outside its hull, whereas a global closed
    # form is the one thing that might.
    rng = np.random.default_rng(a.seed)
    last_train = [max(i for i in range(len(v)) if i % st == 0)
                  for v, st in zip(full_vals, coarsen)]
    interior, extrap = [], []
    for _ in range(a.n_eval * 200):
        idx = tuple(int(rng.integers(0, len(v))) for v in full_vals)
        if all(i % st == 0 for i, st in zip(idx, coarsen)):
            continue                       # this one was trained on
        if not np.all(np.isfinite(pf[idx])):
            continue
        (interior if all(i <= lt for i, lt in zip(idx, last_train))
         else extrap).append(idx)
        if len(interior) >= a.n_eval and len(extrap) >= a.n_eval:
            break
    held = (interior if a.regime == 'interior' else extrap)[:a.n_eval]
    print(f"regime={a.regime}: {len(held)} points "
          f"(pool: {len(interior)} interior, {len(extrap)} extrapolating)")
    if not held:
        print("no points in this regime for this coarsening"); return

    pts = [{ax: float(full_vals[k][idx[k]]) for k, ax in enumerate(axis_names)}
           for idx in held]
    sym_pred = sym.predict_flat(pts)
    lin_pred = lin(np.array([[p[ax] for ax in axis_names] for p in pts]))

    tasks = [(i, pf[idx], lin_pred[i], sym_pred[i]) for i, idx in enumerate(held)]
    res, dropped = [], []
    with ProcessPoolExecutor(max_workers=a.n_workers, initializer=_init,
                             initargs=(obs_names, offsets, a.n_samples,
                                       a.seed)) as ex:
        for r in ex.map(_score, tasks, chunksize=2):
            if r:
                (dropped if r.get('dropped') else res).append(r)
    print(f"scored {len(res)} points; dropped {len(dropped)}")
    if dropped:
        errs = [d['error'] for d in dropped if 'error' in d]
        if errs:
            print(f"  errors ({len(errs)}): {errs[0]}")
        coll = [d for d in dropped if 'error' not in d]
        if coll:
            print(f"  collapsed draws: true={np.mean([d['n_true'] for d in coll]):.0f} "
                  f"linear={np.mean([d['n_lin'] for d in coll]):.0f} "
                  f"symbolic={np.mean([d['n_sym'] for d in coll]):.0f} "
                  f"(of {a.n_samples} requested)")
    print()

    J = {k: np.array([r[k] for r in res]) for k in ('js_floor', 'js_lin', 'js_sym')}
    M = {k: np.array([r[k] for r in res]) for k in ('mmd_floor', 'mmd_lin', 'mmd_sym')}

    print(f"{'observable':15s} {'floor':>8s} {'linear':>8s} {'symbolic':>9s} {'change':>9s}")
    print("-" * 54)
    wins = 0
    for k, o in enumerate(obs_names):
        f, l, s = J['js_floor'][:, k].mean(), J['js_lin'][:, k].mean(), J['js_sym'][:, k].mean()
        better = s < l
        wins += better
        print(f"{o:15s} {f:8.4f} {l:8.4f} {s:9.4f} {(s-l)/l*100:+8.1f}%"
              + ("  better" if better else ""))
    print("-" * 54)
    print(f"{'MEAN':15s} {J['js_floor'].mean():8.4f} {J['js_lin'].mean():8.4f} "
          f"{J['js_sym'].mean():9.4f} "
          f"{(J['js_sym'].mean()-J['js_lin'].mean())/J['js_lin'].mean()*100:+8.1f}%")
    print(f"\nsymbolic better on {wins}/{len(obs_names)} observables")
    print(f"joint MMD  floor {M['mmd_floor'].mean():.5f}   "
          f"linear {M['mmd_lin'].mean():.5f}   symbolic {M['mmd_sym'].mean():.5f}")
    rl = sum(r.get('rep_lin', False) for r in res)
    rs = sum(r.get('rep_sym', False) for r in res)
    print(f"PD repair needed:  linear {rl}/{len(res)}   symbolic {rs}/{len(res)}")

    if a.out:
        np.savez(a.out, obs_names=np.array(obs_names, dtype=object), **J, **M)
        print(f"saved -> {a.out}")


if __name__ == '__main__':
    main()

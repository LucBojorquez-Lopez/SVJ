"""The learned model: load, evaluate, serialise.

Design A replaces the *interpolator*, not the marginal family, so the only
contract that matters is the one `helpers.interpolate_svj_params` offers:

    R_upper, flat_obs_params, param_offsets, obs_names = model.interpolate(p)

`SymbolicModel.interpolate` matches it exactly, which is what lets
`sample_svj_new` and every validation script run against either method through
identical downstream code.  Nothing about the incumbent changes.

Serialised as JSON, not a pickle.  Part of the point of this experiment is that
a human can read the result and see how a parameter scales with mZ; a pickle
would defeat that, and would also tie the file to a numpy version.
"""

import json
import warnings
from pathlib import Path

import numpy as np

import corr as corrmod
from basis import LINKS, build_features


def _nearest_pd_corr(R, floor=1e-6):
    """Project a symmetric matrix onto the nearest positive-definite correlation.

    Design A predicts the 120 correlation entries independently, and nothing in
    that constrains the assembled matrix to be positive definite.  M4 measured
    the incumbent: linear interpolation never produces a non-PD matrix, because
    it is a convex combination of PD ones -- but its median minimum eigenvalue
    is only ~1.5e-3, so R sits close to the boundary and an independent
    predictor has every opportunity to cross it.  `sample_svj_new` feeds R
    straight to `multivariate_normal`, so crossing it is not survivable.

    Clip the eigenvalues at a small positive floor and renormalise the diagonal
    back to 1.  Returns (R_fixed, was_repaired).
    """
    w, V = np.linalg.eigh(R)
    if w.min() >= floor:
        return R, False
    R2 = (V * np.clip(w, floor, None)) @ V.T
    d = np.sqrt(np.diag(R2))
    R2 = R2 / np.outer(d, d)
    np.fill_diagonal(R2, 1.0)
    return R2, True


class SymbolicModel:
    """A fitted Design A model: one expression per parameter-vector component."""

    def __init__(self, spec):
        self.axis_names    = list(spec['axis_names'])
        self.obs_names     = list(spec['obs_names'])
        self.param_offsets = np.asarray(spec['param_offsets'], dtype=int)
        self.corr_start    = int(spec['corr_start'])
        self.n_params      = int(spec['n_params'])
        self.targets       = spec['targets']          # list of per-component dicts
        self.pairwise      = bool(spec.get('pairwise', True))
        self.extra_inputs  = list(spec.get('extra_inputs', []))
        self.derived_exprs = dict(spec.get('derived_exprs', {}))
        self.scale         = spec.get('scale')
        self.corr_param    = spec.get('corr_param', 'entry')
        self.n_repaired    = 0
        assert len(self.targets) == self.n_params

    # ── inputs ────────────────────────────────────────────────────────────────
    def _input_matrix(self, points):
        """(N, K') matrix of axis values plus any extra engineered inputs.

        `extra_inputs` exists so the two candidate input sets in
        docs/symbolic-regression.md ("inputs to try side by side") can be
        compared rather than assumed: the bare 6 axes, or the axes augmented
        with log(mZ) and the derived mPi / mRho / mq.  They add no information
        -- they are functions of the axes -- but the true relationship may be
        far simpler in them.
        """
        P = np.array([[float(p[a]) for a in self.axis_names] for p in points],
                     dtype=float)
        if not self.extra_inputs:
            return P, list(self.axis_names)
        env = {a: P[:, i] for i, a in enumerate(self.axis_names)}
        env['np'] = np
        cols, names = [P], list(self.axis_names)
        for nm in self.extra_inputs:
            expr = self.derived_exprs.get(nm, nm)
            cols.append(np.asarray(eval(expr, {'__builtins__': {}}, env),
                                   dtype=float).reshape(-1, 1))
            names.append(nm)
        return np.hstack(cols), names

    # ── evaluation ────────────────────────────────────────────────────────────
    def predict_flat(self, points):
        """(N, n_params) predicted parameter vectors for a list of dicts."""
        X, names = self._input_matrix(points)
        F, term_names = build_features(X, names, pairwise=self.pairwise,
                                       scale=self.scale)
        idx = {t: i for i, t in enumerate(term_names)}
        out = np.empty((len(points), self.n_params))
        for j, t in enumerate(self.targets):
            v = np.full(len(points), t['intercept'], dtype=float)
            for term, c in zip(t['terms'], t['coefs']):
                if term in idx:
                    v += c * F[:, idx[term]]
                else:
                    # A term the current library no longer produces (e.g. a
                    # log dropped because this grid touches zero).  Fail loudly
                    # rather than silently predicting something else.
                    raise KeyError(
                        f"term {term!r} not in the current basis; the model was "
                        f"fitted against a different axis range")
            # Back through the link, so every prediction lands inside the
            # constraint set the parameter actually lives in.
            _, inv = LINKS[t.get('link', 'identity')]
            with np.errstate(all='ignore'):
                out[:, j] = inv(v)

        # Map the correlation block out of partial coordinates.  Any real
        # prediction lands on a positive-definite correlation matrix, which is
        # the whole point -- see corr.py and M8.
        if self.corr_param == 'partial':
            K = len(self.obs_names)
            cs = self.corr_start
            for n in range(out.shape[0]):
                R = corrmod.partials_to_corr(np.tanh(out[n, cs:]), K)
                out[n, cs:] = corrmod.full_to_upper(R)
        return out

    def interpolate(self, params):
        """Same signature and return contract as helpers.interpolate_svj_params."""
        p = self.predict_flat([params])[0]
        obs_p   = p[:self.corr_start]
        R_upper = p[self.corr_start:]

        K = len(self.obs_names)
        R_upper = np.clip(R_upper, -0.999999, 0.999999)
        R = np.zeros((K, K))
        R[np.triu_indices(K, k=1)] = R_upper
        R += R.T
        np.fill_diagonal(R, 1.0)
        R, repaired = _nearest_pd_corr(R)
        if repaired:
            self.n_repaired += 1
            R_upper = R[np.triu_indices(K, k=1)]
        return R_upper, obs_p, self.param_offsets, self.obs_names

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path):
        spec = dict(
            axis_names=self.axis_names, obs_names=self.obs_names,
            param_offsets=[int(x) for x in self.param_offsets],
            corr_start=self.corr_start, n_params=self.n_params,
            pairwise=self.pairwise, extra_inputs=self.extra_inputs,
            derived_exprs=self.derived_exprs, targets=self.targets,
            scale=self.scale, corr_param=self.corr_param)
        Path(path).write_text(json.dumps(spec, indent=1))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    # ── reporting ─────────────────────────────────────────────────────────────
    def describe(self, j, digits=4):
        """The expression for component j, as readable text.

        Variables are the inputs rescaled to [0, 1] over the training range;
        `self.scale` holds the lo/hi used, so a term reading `mZ` means
        (mZ - lo)/(hi - lo).
        """
        t = self.targets[j]
        parts = [f"{t['intercept']:.{digits}g}"]
        for term, c in zip(t['terms'], t['coefs']):
            parts.append(f"{c:+.{digits}g}*{term}")
        return " ".join(parts)

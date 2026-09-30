"""Candidate term library, and sparse selection over it.

Stage 1 of the symbolic-regression experiment (docs/symbolic-regression.md).
This is not free-form symbolic regression: it searches for a sparse linear
combination over a *designed* library of nonlinear terms in the physics axes.
That is a deliberate first cut rather than a compromise --

  * it needs no new dependency (sklearn is in the LCG view, PySR is not),
  * it is deterministic, so a disappointing result is a fact about the data
    rather than about a random seed,
  * it runs in minutes for all ~184 targets, which is what makes the
    coarsen-and-hold-out loop in holdout.py usable, and
  * every fit is already a readable expression.

PySR is the escalation if this leaves headroom, and the interface here
(fit_target returning named terms plus coefficients) is what it would replace.

Selection is two-stage on purpose.  Lasso on standardised columns picks the
support -- standardisation matters because the raw terms span many orders of
magnitude (mZ^2 against a Box-Cox lambda) and an un-standardised penalty would
simply select the largest-scale columns.  The coefficients are then refitted by
ordinary least squares on the RAW selected columns, so what gets stored is in
natural units and reads as physics rather than as z-scores.
"""

import numpy as np

# Terms are (name_template, callable, requires_positive).  Names are formatted
# with the axis name so the stored expression is self-describing.
_UNARY = [
    ('{0}',        lambda x: x,            False),
    ('{0}^2',      lambda x: x * x,        False),
    ('sqrt({0})',  lambda x: np.sqrt(x),   True),
    ('1/{0}',      lambda x: 1.0 / x,      True),
    ('log({0})',   lambda x: np.log(x),    True),
]


def build_features(P, names, pairwise=True, scale=None):
    """Expand a (N, K) array of axis values into a named design matrix.

    Parameters
    ----------
    P : (N, K) array of physics-axis values.
    names : list of K axis names.
    pairwise : include products x_i * x_j for i < j.

    Returns
    -------
    F : (N, n_terms) array
    term_names : list of str, aligned with F's columns
    """
    P = np.asarray(P, dtype=float)
    # Rescale each input to [0, 1] over the training range before expanding.
    #
    # This is not cosmetic.  Un-scaled, the library spans mZ^2 ~ 1e7 against
    # 1/LambdaDQCD ~ 1e-1, so the design matrix has a condition number large
    # enough that the raw-unit least-squares refit returns coefficients like
    # +1e6 next to -3e-26 -- numerical noise dressed as a fit.  Scaled inputs
    # keep every column O(1).  The cost is that stored expressions are in
    # normalised variables; `scale` is saved with the model so they can be read
    # back, and SymbolicModel.describe reports the mapping.
    if scale is not None:
        lo = np.asarray(scale['lo'], float)
        hi = np.asarray(scale['hi'], float)
        rng = np.where(hi > lo, hi - lo, 1.0)
        P = (P - lo) / rng
    cols, term_names = [], []

    for k, nm in enumerate(names):
        x = P[:, k]
        # A term needing positivity is dropped rather than clipped: the
        # committed scan has mPiOverLambda = 0 on its lower edge, and log or
        # 1/x there would silently poison the whole column with -inf.
        pos = np.all(x > 1e-12)
        for tmpl, fn, need_pos in _UNARY:
            if need_pos and not pos:
                continue
            v = fn(x)
            if not np.all(np.isfinite(v)):
                continue
            cols.append(v)
            term_names.append(tmpl.format(nm))

    if pairwise:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cols.append(P[:, i] * P[:, j])
                term_names.append(f'{names[i]}*{names[j]}')

    return np.column_stack(cols), term_names


def fit_target(F, term_names, y, max_terms=8, cv=5, seed=0, selector='omp',
               link='identity'):
    """Select a sparse support for one target, then refit it in raw units.

    Returns
    -------
    dict with 'terms' (list of str), 'coefs' (list of float), 'intercept',
    and 'r2' (in-sample, on the refitted support).
    """
    from sklearn.linear_model import LassoCV, LinearRegression

    finite = np.isfinite(y)
    if finite.sum() < 3 * max_terms or np.std(y[finite]) == 0:
        # Nothing to learn -- fall back to a constant.  Reported honestly
        # rather than dressed up as a fit.
        c = float(np.nanmean(y[finite])) if finite.any() else 0.0
        return {'terms': [], 'coefs': [], 'intercept': c, 'r2': 0.0,
                'degenerate': True, 'link': link}

    Ff, yf = F[finite], y[finite]
    fwd, _ = LINKS[link]
    with np.errstate(all='ignore'):
        yl = fwd(yf)
    ok = np.isfinite(yl)
    if ok.sum() < 3 * max_terms or np.std(yl[ok]) == 0:
        return {'terms': [], 'coefs': [], 'intercept': float(np.mean(yl[ok])) if ok.any() else 0.0,
                'r2': 0.0, 'degenerate': True, 'link': link}
    Ff, yf = Ff[ok], yl[ok]
    mu, sd = Ff.mean(0), Ff.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Ff - mu) / sd
    ymu, ysd = yf.mean(), yf.std()

    # Two selectors.  'omp' fixes the number of terms directly and needs no
    # cross-validation over a penalty path, which makes it ~10x faster -- and
    # since max_terms is set for interpretability anyway, choosing sparsity by
    # CV was solving a problem we did not have.  'lasso' is kept because it can
    # decide that FEWER terms suffice, which is worth knowing per target.
    yz = (yf - ymu) / ysd if ysd > 0 else yf - ymu
    if selector == 'omp':
        from sklearn.linear_model import OrthogonalMatchingPursuit
        k = min(max_terms, Z.shape[1], max(1, Z.shape[0] - 1))
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=True)
        omp.fit(Z, yz)
        coef = omp.coef_
    else:
        las = LassoCV(cv=cv, random_state=seed, max_iter=20000)
        las.fit(Z, yz)
        coef = las.coef_

    order = np.argsort(-np.abs(coef))
    support = [i for i in order if coef[i] != 0][:max_terms]
    if not support:
        return {'terms': [], 'coefs': [], 'intercept': float(ymu), 'r2': 0.0,
                'degenerate': True, 'link': link}

    ols = LinearRegression().fit(Ff[:, support], yf)
    pred = ols.predict(Ff[:, support])
    ss_res = float(np.sum((yf - pred) ** 2))
    ss_tot = float(np.sum((yf - ymu) ** 2))
    return {
        'terms':     [term_names[i] for i in support],
        'coefs':     [float(c) for c in ols.coef_],
        'intercept': float(ols.intercept_),
        'r2':        1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        'degenerate': False,
        'link':      link,          # R^2 above is in LINK space, not raw units
    }


# ── link functions ────────────────────────────────────────────────────────────
#
# The parameters being regressed are NOT unconstrained reals, and treating them
# as such produces predictions that are not distributions at all.  Measured on
# the committed scan: the shape slot is strictly positive in truth (1.05 to
# 8.4e7) yet 2.6% of raw predictions came out <= 0, and the scale slot (0.036 to
# 3.2e5) 10.8%.  A negative scale makes inverse_observable_col return NaN for
# every draw, which is why the first hold-out run scored 9 of 50 points: the
# model had not merely lost accuracy, it had left the space of valid models.
#
# Linear interpolation never has this problem, because a convex combination of
# valid parameter vectors is valid.  Regression has to earn it, by fitting in a
# space that maps back into the constraint set:
#
#   strictly positive  -> log      (also compresses eight orders of magnitude)
#   in (-1, 1)         -> arctanh  (Fisher z; correlations can never escape)
#   in (0, 1)          -> logit    (mixture weights p0)
#   otherwise          -> identity
#
# Choosing per target from the training values rather than by hand keeps this
# honest when the parameter layout changes -- and M6 shows it does change.
LINKS = {
    'identity': (lambda y: y,                      lambda z: z),
    'log':      (lambda y: np.log(y),              lambda z: np.exp(np.clip(z, -700, 700))),
    'logit':    (lambda y: np.log(y / (1.0 - y)),  lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))),
    'arctanh':  (lambda y: np.arctanh(np.clip(y, -0.999999, 0.999999)), lambda z: np.tanh(z)),
}


def choose_link(y, is_corr=False):
    """Pick a link for one target from its training values."""
    v = y[np.isfinite(y)]
    if v.size == 0:
        return 'identity'
    if is_corr and v.min() > -1.0 and v.max() < 1.0:
        return 'arctanh'
    if v.min() > 0.0:
        if v.max() < 1.0:
            return 'logit'
        return 'log'
    return 'identity'

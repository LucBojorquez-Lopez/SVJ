"""Design B: a learned marginal, as a quantile function.

Why the quantile function and not a density: sampling already needs exactly
this and nothing else.  `sample_svj_new` draws Z ~ MVN(0,R), maps u = Phi(Z),
and calls `inverse_observable_col(u, ...)` -- which *is* a quantile function.
So a learned Q(u) drops in without touching the copula, needs no normalisation
constraint (unlike a density), and needs no numerical inversion (unlike a CDF).

The three-stage design is the user's:

  1. discover which components a marginal needs,
  2. take a consensus support shared across grid points,
  3. refit each point over that fixed support,

and stage 3 is what makes the result interpolable at all: every point ends up
with coefficients over the *same* basis, so point i's numbers correspond to
point j's.  Independent per-point symbolic regression cannot give that, which
is what ruled out "Design C" in docs/symbolic-regression.md.

**The simplification that makes stages 1-2 cheap.**  Evaluate every point's
marginal at the *same* u-levels.  Then the design matrix B(u) is shared, and
each point is x_p = B c_p.  Selecting a support S therefore means choosing
columns of B whose span best contains every x_p at once, and the objective
collapses to a Frobenius norm of one projection residual,

    || (I - P_S) X ||_F^2 ,     X = [x_1 ... x_P]

which greedy forward selection minimises in seconds -- no per-point search, no
16 000 independent SR runs.  This is multi-task orthogonal matching pursuit,
and it answers the "one structure, many coefficient sets" question directly
rather than unioning thousands of separate answers.

The basis is a library of standard quantile-function components rather than
free-form expressions: probit for a Gaussian core, logit for logistic tails,
-log(1-u) for exponential, powers of the probit for Cornish-Fisher style skew
and kurtosis corrections, and (1-u)^-a for heavy tails.  Each is the quantile
function of something recognisable, so a fitted combination is readable.
"""

import numpy as np
from scipy.stats import norm


def u_levels(n=199, zmax=3.2):
    """Probit-spaced quantile levels: denser in the tails than uniform.

    Uniform levels waste resolution in the bulk and resolve the tails badly,
    which is where a marginal model usually fails and where `sample_svj_new`'s
    NaN draws come from.
    """
    return norm.cdf(np.linspace(-zmax, zmax, n))


# Each entry is (name, callable).  Every one is the quantile function of a
# recognisable distribution, or a standard correction term, so a fitted
# combination can be read rather than merely evaluated.
_TERMS = [
    ('1',              lambda u, z: np.ones_like(u)),
    ('probit',         lambda u, z: z),
    ('probit^2*sgn',   lambda u, z: z * np.abs(z)),
    ('probit^3',       lambda u, z: z ** 3),
    ('logit',          lambda u, z: np.log(u / (1.0 - u))),
    ('-log(1-u)',      lambda u, z: -np.log(1.0 - u)),
    ('log(u)',         lambda u, z: np.log(u)),
    ('u',              lambda u, z: u),
    ('u^2',            lambda u, z: u ** 2),
    ('(1-u)^-0.5',     lambda u, z: (1.0 - u) ** -0.5 - 1.0),
    ('(1-u)^-0.25',    lambda u, z: (1.0 - u) ** -0.25 - 1.0),
    ('u^-0.25',        lambda u, z: u ** -0.25 - 1.0),
    ('tan(pi(u-.5))',  lambda u, z: np.tan(np.pi * (u - 0.5))),
]


def qbasis(u):
    """(n_levels, n_terms) design matrix and its term names."""
    z = norm.ppf(u)
    cols, names = [], []
    for nm, fn in _TERMS:
        with np.errstate(all='ignore'):
            v = fn(u, z)
        if np.all(np.isfinite(v)):
            cols.append(v)
            names.append(nm)
    return np.column_stack(cols), names


def empirical_quantiles(x, u):
    """Empirical quantiles of one point's events at the shared levels."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return None
    return np.quantile(x, u)


def select_support(X, B, k=6, standardise=True):
    """Greedy multi-task column selection: one support, all points.

    X : (n_levels, n_points) each column a point's empirical quantiles
    B : (n_levels, n_terms)

    Standardising each column first stops large-scale points (HT is hundreds of
    GeV, leadWidth is O(0.1)) from deciding the support on their own.
    """
    Xs = X.astype(float).copy()
    if standardise:
        mu = Xs.mean(0, keepdims=True)
        sd = Xs.std(0, keepdims=True)
        Xs = (Xs - mu) / np.where(sd > 0, sd, 1.0)
    n_terms = B.shape[1]
    support, remaining = [], list(range(n_terms))
    total = float(np.sum(Xs ** 2))
    history = []
    for _ in range(min(k, n_terms)):
        best, best_res = None, np.inf
        for j in remaining:
            S = support + [j]
            Bs = B[:, S]
            # residual of projecting every point onto span(B_S) at once
            coef, *_ = np.linalg.lstsq(Bs, Xs, rcond=None)
            res = float(np.sum((Xs - Bs @ coef) ** 2))
            if res < best_res:
                best, best_res = j, res
        support.append(best)
        remaining.remove(best)
        history.append((best, 1.0 - best_res / total))
    return support, history


def fit_points(X, B, support):
    """Per-point coefficients over the fixed support, in raw units.

    Returns C (n_points, len(support)) and per-point R^2.
    """
    Bs = B[:, support]
    C, *_ = np.linalg.lstsq(Bs, X, rcond=None)
    pred = Bs @ C
    ss_res = np.sum((X - pred) ** 2, axis=0)
    ss_tot = np.sum((X - X.mean(0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)
    return C.T, r2


def evaluate_q(coefs, support, B):
    """Q at the basis' u-levels for one point's coefficients."""
    return B[:, support] @ np.asarray(coefs)


def sample_from_q(coefs, support, u_draw, enforce_monotone=True, u_range=None):
    """Draw values by evaluating Q at given u.

    `u_range` is not optional in practice.  The basis is fitted on a finite set
    of levels -- by default u in [0.0007, 0.9993] -- and several of its terms
    (`u^-0.25`, `log(u)`, `(1-u)^-0.5`) diverge outside that window.  Sampling
    u down to 1e-6 therefore evaluates the fit far outside where it was
    constrained, and the result is not merely inaccurate but non-monotone:
    measured at 19-62% of draws needing repair, against 0-2% once clipped.

    This is a real limitation of a non-parametric quantile function rather than
    a bug to engineer away.  With ~10k events per point the 1e-6 quantile is
    simply not estimable -- it lies below the smallest event -- so the honest
    move is to truncate at the fitted range and say so.  A parametric family
    like `gennorm` extrapolates into the tails analytically, which is exactly
    the advantage it retains here.

    Monotonicity is still repaired by a running maximum over sorted u, and the
    count is returned, because a model needing frequent repair is not a model.
    """
    if u_range is not None:
        u_draw = np.clip(u_draw, u_range[0], u_range[1])
    Bd, _ = qbasis(np.clip(u_draw, 1e-9, 1 - 1e-9))
    # qbasis returns all terms; index the same support
    vals = Bd[:, support] @ np.asarray(coefs)
    if not enforce_monotone:
        return vals, 0
    order = np.argsort(u_draw)
    v = vals[order]
    fixed = np.maximum.accumulate(v)
    n_bad = int(np.sum(fixed != v))
    out = np.empty_like(vals)
    out[order] = fixed
    return out, n_bad


# ── a marginal that handles the two things a bare Q cannot ────────────────────
#
# M10 measured a bare quantile fit failing on exactly two structures, both of
# which M7 had already identified and neither of which is a flaw in Q itself:
#
#   * an ANGULAR FOLD.  dPhiMETfar is peaked at +-pi with an empty middle -- one
#     physical peak split by the phi wrap.  The existing pipeline folds it with
#     `abs_value`, whose inverse "alternates signs; assumes symmetric signed
#     distribution".  Fitting Q on the raw signed column instead asks it to
#     reproduce a two-peaked shape it has no business representing.
#
#   * a BOUNDARY ATOM.  Seven observables put 4-20% of their events on a single
#     value (maxMuPt = 0 with no muon, fInv = 0 for a fully visible jet,
#     dPhiMETclose at +-pi).  A continuous Q must smear that spike across a
#     region, and no number of basis terms fixes a point mass.
#
# Both are handled the way the incumbent handles them: fold first, then model
# the atom as a discrete mixture component and Q on the continuous remainder.

def fit_marginal(x, u, B, support=None, k=9, fold=False, atom=None):
    """Fit a marginal as (optional fold) + (optional atom) + Q on the rest."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if fold:
        x = np.abs(x)
    p0, aval = 0.0, None
    if atom is not None:
        aval = abs(atom['value']) if fold else atom['value']
        tol = atom.get('tol', 1e-10)
        m = np.abs(x - aval) <= tol
        frac = float(m.mean())
        if frac >= atom.get('min_p0', 0.0) and frac > 0:
            p0 = frac
            x = x[~m]
        else:
            aval = None
    if x.size < 200:
        return None
    q = empirical_quantiles(x, u)
    if q is None or not np.all(np.isfinite(q)):
        return None
    if support is None:
        support, _ = select_support(q.reshape(-1, 1), B, k=k)
    C, _ = fit_points(q.reshape(-1, 1), B, support)
    # Where the atom sits decides how the uniform draw is split.
    at_top = aval is not None and aval >= np.median(q)
    return dict(support=support, coefs=C[0], p0=p0, atom=aval,
                at_top=bool(at_top), fold=bool(fold),
                u_range=(float(u.min()), float(u.max())))


def sample_marginal(fit, n, rng):
    """Draw n values from a fit_marginal result."""
    u = rng.uniform(0.0, 1.0, n)
    out = np.empty(n)
    p0 = fit['p0']
    if p0 > 0:
        # The atom takes a slice of the unit interval at whichever end it lives,
        # and the continuous part is rescaled onto the remainder -- the same
        # construction inverse_observable_col uses for point_mass.
        hit = (u >= 1.0 - p0) if fit['at_top'] else (u <= p0)
        out[hit] = fit['atom']
        uc = ((u[~hit] / (1.0 - p0)) if fit['at_top']
              else ((u[~hit] - p0) / (1.0 - p0)))
    else:
        hit = np.zeros(n, dtype=bool)
        uc = u
    if uc.size:
        vals, _ = sample_from_q(fit['coefs'], fit['support'], uc,
                               u_range=fit['u_range'])
        out[~hit] = vals
    if fit['fold']:
        # Undo the fold the way abs_value's inverse does: the signed
        # distribution is assumed symmetric, so restore a random sign.
        out = out * rng.choice([-1.0, 1.0], size=n)
    return out

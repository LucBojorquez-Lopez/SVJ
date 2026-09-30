# Symbolic regression over the scan

An experiment, not yet a component. The goal is to find out whether closed-form
expressions can replace parts of the fit-then-interpolate pipeline, starting
with the part that measurement says is broken.

---

## 1. Why: the interpolator, not the marginals

`simulated/svj/validation_production.npz` already scores the shipped model
against PYTHIA truth at 2000 held-out interior points, alongside two reference
curves: `js_baseline`, the statistical floor at that sample size, and
`js_nearest`, what you get by snapping to the nearest grid point and not
interpolating at all.

Mean Jensen-Shannon distance, 2000 points:

| observable | floor | interp | nearest |
|---|---|---|---|
| Meff | 0.032 | **0.218** | 0.089 |
| leadVisPt | 0.032 | **0.188** | 0.086 |
| HT | 0.032 | **0.158** | 0.082 |
| nConst | 0.032 | **0.123** | 0.076 |
| dPhiMETclose | 0.032 | **0.118** | 0.042 |
| maxMuPt | 0.032 | **0.057** | 0.048 |
| MET | 0.032 | 0.034 | 0.048 |
| jetThrust | 0.032 | 0.041 | 0.084 |
| leadWidth, hemiMass1, e2c, e3c, tau1-3, leadJetMass | 0.032 | 0.045-0.066 | 0.062-0.098 |
| **mean** | **0.0321** | **0.0842** | **0.0758** |

Joint distribution, MMD: baseline 0.00043, **interp 0.0208**, nearest 0.0122.

Read that again: on the joint distribution the linear interpolator is about 70%
**worse** than nearest-grid, and on the per-observable mean it is worse too. It
beats nearest on only 58.7% of (point, observable) pairs and reaches the floor
on 20.7%.

Six observables carry almost all of the damage, and they have something in
common: `HT`, `Meff`, `leadVisPt` are energy scales, `nConst` a multiplicity —
all strongly `mZ`-dependent, on an 8-point **log** axis spanning 500 to 4000 GeV.

The likely mechanism is that linear interpolation of *fitted parameters* is not
interpolation of *distributions*. Each observable carries 4 parameters (a
Box-Cox lambda plus a `gennorm` shape/loc/scale, say). Those interact
nonlinearly, so a straight line between two valid parameter vectors can leave
the manifold of sensible distributions entirely. Nearest-grid at least always
returns a distribution some grid point actually had.

**This is the opportunity.** Not a better distribution family — a better map
from physics to parameters. Symbolic regression produces exactly that: a smooth,
closed-form, extrapolable theta(p), which also regularises away the fit noise
that linear interpolation currently passes through verbatim (re-fitting
identical data moves parameters by ~1e-2; see [raw-store.md](raw-store.md)).

### Is the model *form* the problem? No.

One more measurement, because it decides whether any of this should touch the
copula. At a single grid point, fit the full model — 16 marginals plus the
Gaussian copula — on half the events and compare a sample from it against the
held-out half, using MMD. No interpolation is involved, so this scores the
model form alone:

| point | MMD floor | MMD model | ratio |
|---|---|---|---|
| mZ = 1000 | 0.00033 | 0.00076 | **2.3x** |
| mZ = 3000 | 0.00034 | 0.00045 | **1.3x** |

Against the interpolated model's **48x** the floor at interior points
(0.0208 against 0.00043) from §1. So the joint model form is close to adequate
and interpolation accounts for essentially all of the error.

**Consequence: do not spend effort on a better copula.** A t-copula, a vine, a
richer dependence structure — none of it addresses a 48x error whose form-level
component is 1.3-2.3x. Whatever SR does here, it should be aimed at the map.

(`nConst` is again the worst marginal, at 4.84x and 2.13x the floor, agreeing
with the `validate_fit` diagnostic below. It is excluded from Design A.)

### The diagnostic, and its answer

The table above cannot by itself separate "interpolation is bad" from "the
parameterisation is unsuited to interpolation". `validate_fit.py` settles it: it
reports `js_fit` against `js_baseline` for one point's events, in-sample, so a
ratio near 1 means the per-point fit is already as good as the data allows.

Run on 50 000 fresh events at `mZ = 1000` and `mZ = 3000` (compare ratios, not
absolute JS — the floor depends on `N2`, here 5000 against the 20 000 used for
§1's table):

| observable | js_fit/js_base, mZ=1000 | mZ=3000 | verdict |
|---|---|---|---|
| HT | 0.85x | 0.86x | fit at the floor |
| Meff | 0.98x | 0.74x | fit at the floor |
| leadVisPt | 1.18x | 0.81x | fit essentially at the floor |
| maxMuPt | 1.00x | 0.60x | fit at the floor |
| dPhiMETclose | **1.91x** | 0.78x | **poor at low mZ**, fine at high |
| nConst | **1.96x** | **1.41x** | **fit itself is poor, everywhere** |
| MET / jetThrust / tau1 (well-behaved) | 0.86x / 0.70x / 0.85x | 0.87x / 0.68x / 0.81x | fit at the floor |

This splits §1's six problem observables into two different problems:

- **`HT`, `Meff`, `leadVisPt`, `maxMuPt` — pure map failures.** The per-point
  fit is at the statistical floor, so every bit of their interpolated error
  (`Meff` 0.218 against a 0.032 floor) is the linear parameter map. **Design A
  should fix these outright.** This is the strongest evidence for the plan.
- **`nConst` — a model failure.** Its per-point fit is 1.96x / 1.41x the floor,
  so no interpolator can rescue it. `nConst` is an integer multiplicity and
  Box-Cox to `gennorm` is simply the wrong model for a discrete count. Design A
  will not help; it needs Design B or a discrete-aware treatment.
- **`dPhiMETclose` — both.** Poor fit at low `mZ` (1.91x) but fine at high, so
  part map, part model. Expect partial improvement from A.

So Design A is aimed correctly, but its ceiling is now known in advance: it
cannot recover `nConst`, and only partly `dPhiMETclose`. Predicting that up
front is useful — if SR fixes four of six and leaves those two, that is success,
not shortfall.

---

## 2. Design A — symbolic regression over the parameter map

Learn, for each component j of the 184-vector, an expression

```
theta_j = f_j(mZ, rinv_pion, mPiOverLambda, LambdaDQCD, alphaD, jetR)
```

trained on the committed 16384-point scan (8x8x4x4x4x4, zero NaN points). This
needs **no new data and no waiting on Condor.**

The 184 components are not homogeneous, and the split matters:

- **64 marginal parameters** (4 per observable). Independent scalar targets.
  Straightforward.
- **120 correlation entries** — the upper triangle of a 16x16 correlation
  matrix. Predicting these independently is **not safe**: nothing constrains the
  result to be positive definite, and `sample_svj_new` feeds R straight to
  `multivariate_normal`. Options, in order of preference:
  1. Learn in an unconstrained parameterisation that is PD by construction —
     the Cholesky factor's off-diagonals, or partial correlations — and map back.
  2. Predict entries directly, then project to the nearest PD matrix
     (clip eigenvalues at a small positive floor, renormalise the diagonal).
  Option 2 is a two-line fallback and worth having regardless, since the
  *existing* linear interpolator has the same exposure and no guard at all.
  Whether it currently produces non-PD matrices at interior points is itself
  worth measuring.

**Inputs.** The 6 axes, plus terms that make the physics expressible: `log(mZ)`
(the axis is log-spaced, so this is the natural variable), and the derived
quantities the config already computes — `mPi = mPiOverLambda * Lambda`,
`mRho ~ 2.4 * Lambda`, `mq`. These are deterministic functions of the axes, so
they add no information, but they give the search a basis in which the true
relationship may be far simpler.

**Engine, staged.**

1. **Sparse regression over a designed basis.** A library of candidate terms —
   powers, logs, ratios and low-order products of the inputs — fitted with
   `Lasso`/OMP from the already-present `sklearn`, selecting a handful of terms
   per target. Deterministic, minutes for all 184 targets, immediately readable,
   and **no new dependency**. This is the baseline to beat.
2. **PySR**, only if step 1 leaves headroom. Julia is already in the LCG view
   (`/cvmfs/.../bin/julia`), so this is installable, but `JULIA_DEPOT_PATH` must
   point at EOS — the 2 GB AFS home cannot hold a Julia depot.

---

## 3. Protocol: a cheap inner loop, one expensive confirmation

`validate_production.py` re-runs PYTHIA three times per validation point
(truth, an independent truth replica for the floor, and the nearest grid point).
At 2000 points that is roughly 330 CPU-hours — too expensive to iterate on, and
it would compete with the production scan for slots.

So iterate without PYTHIA at all:

**Coarsen and hold out.** Subsample the grid to a coarse subgrid (every other
point along the dense axes). Fit *both* methods on the coarse subgrid only —
`RegularGridInterpolator` for the incumbent, sparse-basis SR for the challenger
— then evaluate both at the held-out points, where the true fitted parameter
vector is already known.

Score in distribution space, not parameter space, because parameter error is
not what anyone cares about: at each held-out point, sample from the *true*
fitted parameters and from each method's *predicted* parameters, and compute JS
per observable and joint MMD between those two samples. Reuse `js_per_obs` and
`mmd_rbf` from `_val_utils.py` unchanged.

This isolates exactly the quantity of interest — interpolation error, with
per-point fit quality held fixed — costs no simulation, and runs in minutes, so
the basis library and regularisation can be tuned properly.

**Then confirm once.** When the challenger wins the cheap benchmark, run
`validate_production.py --model symbolic:...` at a reduced `--N3` (200-300
points rather than 2000) for a real PYTHIA comparison against the numbers in
§1. That is ~30-50 CPU-hours, submitted the same way as any other array.

**Success is defined by §1's numbers**: mean JS below the linear interpolator's
0.0842 and moving toward the 0.0321 floor, and joint MMD below 0.0208. The
honest bar to clear first is `js_nearest` = 0.0758 / MMD 0.0122, since the
incumbent does not currently clear it.

---

## 4. How it plugs in

A parallel module, so nothing existing changes behaviour:

```
src/run_regression/symbolic/
    __init__.py
    basis.py          candidate term library + sparse fit (stage 1)
    fit_symbolic.py   scan NPZ -> expressions        (CLI)
    model.py          SymbolicModel: load, evaluate, serialise
    holdout.py        the coarsen-and-hold-out benchmark of section 3
configs/symbolic.cfg  basis terms, sparsity, targets, engine
```

The load-bearing design choice is that `model.py` exposes **the same signature
as `helpers.interpolate_svj_params`** —

```python
R_upper, flat_obs_params, param_offsets, obs_names = model.interpolate(params)
```

— so `sample_svj_new` and everything downstream are untouched, and a `--model`
flag on `validate_grid.py` / `validate_production.py` selects incumbent or
challenger. Both then run through *identical* evaluation code, which is the only
way the comparison means anything.

Expressions serialise as strings plus metadata (JSON, or an NPZ of object
arrays) and evaluate through `sympy.lambdify`; `sympy` 1.14 is already present.
Storing the expression text, not a pickle, keeps the model readable and
reviewable — the point of the exercise is partly that a human can look at it.

**Delphes, no-Delphes, and stored data all come free.** Design A's input is a
scan NPZ, and the Delphes stream produces `svj_regression_delphes_scan.npz` with
the same structure and column semantics. `fit_symbolic.py` takes a path; the
stream is which path. Nothing in the module needs to know which generator ran.

---

## 5. Design B — a learned marginal form, in three stages

The design here is the user's, and it is better than fitting one global
`Q(u; p)`: it removes the objection that killed Design C.

**Stage 1 — structure discovery, per point.** For each observable, at each of
many grid points, run SR on that point's marginal and record which *components*
(terms, operators, sub-expressions) the search selects.

**Stage 2 — consensus support.** Take a shared set of components across points:
the union of what stage 1 found, or better, the support that a **multi-task /
group-sparse** selection picks as useful across many points at once. A raw
union over 16 000 independent searches would be enormous and mostly noise;
group sparsity asks the sharper question — "one structure, many coefficient
sets" — and answers it directly.

### How big may the superset be?

There is no constraint tying an observable to 4 or 5 parameters. If the
consensus support wants 10, take 10. The cost is negligible and worth stating
so nobody optimises against a limit that does not exist:

| params/observable | total per point | scan NPZ | SR targets |
|---|---|---|---|
| 4 (today) | 184 | 24 MB | 184 |
| 10 | 280 | 37 MB | 280 |
| 25 | 520 | 68 MB | 520 |

Even 25 per observable is a 68 MB NPZ against the 25 MB shipped today, and the
SR target count scales linearly and stays trivial. Storage, memory and fit
time are all non-issues.

**The binding constraint is identifiability, and M5 measures it directly.** At
today's 4-5 parameters, 15.2% of marginal parameters already have a statistical
sigma larger than their median neighbour-to-neighbour variation — they carry no
recoverable grid-to-grid signal at 20 000 events per point. Adding parameters
divides the *same* 20 000 events among more of them, so that fraction only
grows. A superset large enough to fit each point beautifully in isolation can
still be useless, because stage 3 has nothing smooth left to regress.

So the superset size should be chosen by a stopping rule rather than a guess,
and M5 supplies it: **a candidate component earns its place only if its
coefficient varies across the grid by more than its own statistical noise.**
Concretely — bootstrap the fit at a handful of grid points, compute
sigma/|delta| per coefficient as in M5, and grow the support while the
noise-dominated fraction stays acceptable (25% is a defensible line; today's
model already sits at 15%). This is cheap, it reuses M5's machinery unchanged,
and it makes "how many components?" an empirical question with an answer
instead of a matter of taste.

It also suggests a lever if the answer is unsatisfying: identifiability scales
with events per point, so a superset that is unidentifiable at `nEvent = 20000`
may be fine at 50 000. That trades simulation time for model capacity, and the
raw store means the trade can be evaluated before committing to it.

**Stage 3 — refit with the support fixed.** Every grid point now gets
coefficients over the *same* fixed-length basis. That is what makes this work,
and it is precisely what naive per-point SR (Design C) cannot give: a
correspondence between point *i*'s numbers and point *j*'s, and hence something
that can be interpolated or regressed over physics at all. It is the same
contract the current pipeline has — 4 numbers per observable per point — with a
learned functional form instead of Box-Cox to `gennorm`, and a superset of
components rather than a fixed four.

**Design A is stage 3.** Once the coefficient vector exists, mapping it to
physics is exactly Design A's problem on different numbers. So A is not a
detour before B; A builds the half of B that B would otherwise still need. That
is the real reason to do it first, beyond it being cheaper.

### What to regress: the quantile function

For a marginal, the choice of target function matters more than the choice of
SR engine:

- **Quantile function `x = Q(u)`, `u` in (0,1) — recommended.** Sampling needs
  exactly this and nothing else; `inverse_observable_col` *is* this function
  today, so it drops in without touching the copula. The only structural
  constraint is monotonicity in `u`. No binning, no normalisation.
- **PDF `f(x)`** — needs the integral to be 1, which free-form SR will not
  respect, and needs binning choices that bias the tails.
- **CDF `F(x)`** — monotone and bounded, but sampling then needs a numerical
  inverse per draw, which is slower and reintroduces the tail failures that
  `sample_svj_new`'s NaN draws already come from.

Point masses still need a mixture: a learned p0(p) for the atom at the boundary
(`maxMuPt` = 0 with no muon, `fInv` = 0 for a fully visible jet) plus Q on the
continuous part.

Gated on the raw store, which is being written now.

### SR on the joint, and why it stops at the parameters

Worth being precise, because "do SR on the joint" sounds like it should mean
one expression for the 16-dimensional density, and it cannot.

The joint is represented as 16 marginals plus a Gaussian copula with a 16x16
correlation matrix — 120 free numbers. Design A regresses those 120 alongside
the 64 marginal parameters, so **SR is applied to the joint's parameters**. What
it does not do is change the joint's functional form.

Free-form SR of a 16-dimensional density is not practical, and the reason is
structural rather than a tooling gap. SR searches for an expression in its input
variables; a density over 16 correlated observables has no natural scalar target
to regress, and estimating one non-parametrically in 16 dimensions needs data
that grows exponentially in the dimension. The copula decomposition exists
precisely to dodge this: it factorises one 16-dimensional problem into sixteen
one-dimensional problems plus a correlation matrix, each of which SR *can* handle.

Tractable things one could still do to the dependence structure, in rough order
of value:

1. Regress `R_ij(p)` — already in Design A.
2. Regress **Kendall's tau_ij(p)** instead of Pearson `R_ij(p)`, and convert.
   tau is copula-invariant and better behaved under transforms, so it is
   plausibly a smoother SR target than the Pearson entries.
3. Fit a richer copula and regress *its* parameters (a t-copula's nu(p), vine
   pair parameters).

Given the measurement above — form 1.3-2.3x the floor against interpolation's
48x — only item 1 is currently justified, and item 2 is a cheap variant of it
worth trying side by side.

### Inputs to try side by side

Two input sets, compared rather than assumed: the 6 raw axes, and the 6 axes
augmented with `log(mZ)` and the derived `mPi`, `mRho`, `mq`. The augmented set
adds no information — they are deterministic functions of the axes — but the
true relationship may be far simpler in them, and which one wins is an
empirical question, not a judgement call.

## 6. What could make this not work

- If `validate_fit.py` shows the per-point fits are themselves poor for the six
  bad observables, Design A cannot fix them and §1's framing is wrong.
- The 4 parameters per observable may be so ill-conditioned (near-degenerate
  Box-Cox / `gennorm` combinations) that no smooth theta(p) exists to be found.
  A tell would be sparse-basis SR fitting the well-behaved observables easily
  and failing on exactly the six bad ones.
- ~~Fit non-determinism is label noise that may swamp the signal.~~
  **Measured and cleared — see M5.** Signal-to-noise is roughly 4:1 for the
  median parameter and the fit is exactly reproducible in-process. The residual
  concern is narrower: 15% of marginal parameters are noise-dominated, so those
  need regularisation and honest reporting rather than a claimed fit.
- Design A improves the map but inherits the family. There may be a ceiling
  well above the statistical floor that only Design B can pass.

---

## 7. Measurement log

Every claim in this document traces to one of these. Each entry records what
was measured, how, and what it forces.

### M1 — the incumbent is worse than nearest-grid  (2026-09-02)

Read directly from the shipped `simulated/svj/validation_production.npz`
(2000 held-out interior points, PYTHIA truth). Mean JS 0.0842 for the linear
interpolator against 0.0758 for nearest-grid and a 0.0321 floor; joint MMD
0.0208 against 0.0122 and 0.00043. Beats nearest on 58.7% of (point,
observable) pairs; at the floor on 20.7%.

**Forces:** the target is the parameter map, not the distribution family.

### M2 — the model form is nearly adequate  (2026-09-02)

50 000 fresh events at `mZ` = 1000 and 3000. Fit all 16 marginals plus the
Gaussian copula on half, sample, MMD against the held-out half. Result 2.3x and
1.3x the floor, against interpolation's 48x.

**Forces:** do not build a better copula. Spend nothing on t-copulas or vines.

### M3 — which observables are map failures, which are model failures  (2026-09-02)

`validate_fit.py`, `js_fit / js_baseline`, same two points. `HT` 0.85x/0.86x,
`Meff` 0.98x/0.74x, `leadVisPt` 1.18x/0.81x, `maxMuPt` 1.00x/0.60x — all at the
floor, so their interpolated error is entirely the map. `nConst` 1.96x/1.41x —
the fit itself is poor. `dPhiMETclose` 1.91x/0.78x — poor at low `mZ` only.

**Forces:** Design A should fix four of §1's six; `nConst` is excluded from A;
`dPhiMETclose` should be expected to improve only partly.

### M4 — the interpolated correlation matrices are positive definite  (2026-09-02)

300 exact grid points, 300 random interior points, 300 grid midpoints. Zero
non-PD in all three sets; worst minimum eigenvalue 1.79e-4, 4.14e-4, 4.62e-4;
all entries inside [-1, 1].

**Forces:** not a current bug, so M1's poor scores are not explained by broken
correlation matrices. But the median minimum eigenvalue is only ~1.5e-3, so R
sits close to the PD boundary — linear interpolation stays inside because it is
a convex combination of PD matrices, and SR predicting 120 entries
independently has no such guarantee. Keep the PD-safe parameterisation for the
challenger.

### M5 — there is signal to fit  (2026-09-02)

The question that could have invalidated Design A: if neighbouring grid points
differ by less than the noise on each, no smooth map can be recovered. Measured
at `n` = 20000 to match the production scan — five refits of identical events
for algorithmic noise, twelve independent bootstrap draws for statistical
noise — against the median absolute neighbour difference on the committed scan.

| | median sigma/&#124;delta&#124; | frac > 1 | frac > 0.5 |
|---|---|---|---|
| 66 marginal params, algorithmic | **0.00** | 0.0% | 0.0% |
| 66 marginal params, statistical | **0.25** | 15.2% | 31.8% |
| 120 correlation entries, algorithmic | **0.00** | 0.0% | 0.0% |
| 120 correlation entries, statistical | **0.26** | 0.0% | 3.3% |

Median neighbour difference 0.073 against a statistical sigma of 0.017 for the
marginals (0.018 against 0.0051 for the correlations) — roughly **4:1
signal-to-noise** for the median parameter.

**Forces:** Design A is viable. Two riders. Fifteen percent of marginal
parameters *are* noise-dominated, so SR must be regularised rather than fitted
to convergence, and those parameters should be identified and reported rather
than silently trusted. The correlation entries are the cleaner target — nothing
noise-dominated at all — which is mildly surprising given they were the part
flagged as risky in M4.

Also: within a single process the fit is **exactly** reproducible (sigma = 0.00,
both groups). An earlier cross-run comparison in
[raw-store.md](raw-store.md) found ~7.6e-3 between two separate `fit_raw.py`
invocations on identical data; that is not in-process optimiser noise and its
cause is unresolved — plausibly BLAS thread-order differing between processes.
It does not affect that document's conclusion, since float32's relative error of
2.3e-8 is orders of magnitude below either figure.

### M6 — the committed scan is stale relative to `observables.py`  (2026-09-02)

The committed `svj_scan.npz` carries 184 parameters; today's code produces 186
for the same 16 observables. Located exactly: `maxMuPt` and `dPhiMETclose` have
each since gained a 5th parameter, the point-mass mixture weight.

| observable | committed | today | has point_mass |
|---|---|---|---|
| maxMuPt | 4 | **5** | yes |
| dPhiMETclose | 4 | **5** | yes |
| the other 14 | 4 | 4 | no |

Nothing breaks at runtime — `sample_svj_new` reads `param_offsets` from the NPZ
rather than assuming them. But two consequences matter:

1. **Those two are among §1's six worst.** `dPhiMETclose` is the single worst
   relative to nearest-grid (0.118 against 0.042). The committed scan modelled
   both *without* the boundary atom that the current code fits, so part of
   their poor score is a stale model rather than interpolation. M1 therefore
   understates today's code for these two. Testable, and worth testing before
   attributing their error to the map.
2. **Design A's deliverable must be trained on the new production scan**, not
   the committed one, since the current sampling path expects 186. The
   committed scan remains fine for prototyping the method.

### M7 — there is essentially no genuine bimodality; the real problem is discreteness  (2026-09-02)

Motivated by a reasonable expectation: SR should handle a single well-behaved
distribution and struggle with bimodal ones, and any bimodality would be
pointing at physics worth understanding. Measured on 40 grid points from the
production raw store, all 28 observables — KDE mode count (5% prominence),
bimodality coefficient, largest-repeated-value fraction, and unique-value count.

The first pass flagged five observables as multimodal. **On inspection all of
them were artefacts of the detector, not real bimodality.** The 12-bin
histograms:

| observable | apparent | what it actually is |
|---|---|---|
| `metPhi` | 1.62 modes | **flat.** Counts ~1500 in every bin, KS against uniform p = 0.04. Uniform in phi by rotational symmetry, exactly as it should be — KDE noise on a flat density crossed the prominence threshold. |
| `dPhiMETfar` | 1.70 modes | **one peak, split by the coordinate.** Counts `[7580, 1396, 388, 0×6, 391, 1449, 7520]` — peaked at both -pi and +pi with an empty middle. The far jet is by construction near pi from MET; the two peaks are the same physical peak either side of the phi wrap. The pipeline already folds it with `abs_value`, so the *fitted* variable is unimodal. |
| `nJets` | 2.20 modes | **discrete.** Integer 1-10, mode at 2 (dijet-dominated, as expected). The interleaved zero bins are a histogram artefact of binning integers, not a second mode. |
| `nInvClose` | 2.30 modes | discrete, 12 unique values. Same artefact. |
| `ptBal` | 1.38 modes | **unimodal with an atom.** Monotonically falling from a peak at 0 with an 8% atom there. |

So the expectation does not bite here: **the 16 currently-fitted observables are
all unimodal (1.00 modes, 0% multimodal)**, and after accounting for flat
densities, integer binning and angular wrap-around, none of the other 12 shows
genuine continuous bimodality either.

What the scan *does* reveal is a different and real taxonomy:

1. **Continuous unimodal** — 11 of the 16 fitted. SR-friendly, and M3 confirms
   they already fit at the statistical floor.
2. **Continuous with a boundary atom** — `maxElePt` 13.4%, `fInv` 19.8%,
   `dPhiMETdijet` 17.7%, `hemiMass2` 13.1%, `RT`/`ptBal` 8.0%, `maxMuPt` and
   `dPhiMETclose` 3.8-4.1%. Handled today by the `point_mass` mixture; any SR
   marginal needs the same mixture, not a single expression.
3. **Discrete counts** — `nConst` 138 unique values, `nJets` 11, `nInvClose` 12.
   **This, not bimodality, explains `nConst`'s 1.96x fit ratio in M3**:
   `nConst` is unimodal (1.00 modes) but integer-valued, and Box-Cox to
   `gennorm` is a continuous model for a discrete count.
4. **Binary** — `closeJetIsLead`, 2 values, 85% in one of them. Not a
   distribution-fitting problem at all, consistent with it having no pipeline.

**Design implications.**

- SR should be applied to the variable *after* the folding/abs stage of the
  pipeline, or its operator set must include `abs`. On the raw `dPhiMETfar`
  column an expression search would waste itself rediscovering the phi fold.
- The atoms are not optional detail: seven observables carry one, up to 20% of
  events. A learned marginal is a mixture — p0(p) for the atom plus Q on the
  continuous part — and this was already the plan in section 5.
- `nConst` needs a discrete treatment, not a better map or a better search. It
  stays excluded from Design A.
- Nothing here requires a multimodal-capable marginal model, which removes a
  risk from the Design B search rather than adding one.

> **A note on the method.** The mode-count detector was wrong on five of five
> flagged observables. Anyone re-running this should look at the histograms
> before believing a mode count: KDE prominence on a flat density, integer
> binning, and angular wrap-around each produce confident false positives.

### M8 — Design A built and benchmarked; it loses globally, wins exactly where the incumbent fails  (2026-09-02)

First end-to-end run of `src/run_regression/symbolic/` on the committed scan.
Trained on a coarse subgrid (8x8x4x4x4x4 -> 4x4x4x4x4x4, 4096 points), both
methods evaluated at 50 grid points outside it, 8000 draws each, scored against
samples from the true fitted parameters.

**Two bugs had to be fixed before any number meant anything.**

*Conditioning.* Un-scaled, the term library spans `mZ^2` ~ 1e7 against
`1/LambdaDQCD` ~ 1e-1, and the raw-unit least-squares refit returned
coefficients like +1e6 beside -3e-26 — numerical noise dressed as a fit.
Rescaling every input to [0, 1] over its training range fixed it.

*Constraint violation — the important one.* The parameters are not
unconstrained reals. The shape slot is strictly positive in truth (1.05 to
8.4e7); 2.6% of raw predictions came out <= 0. The scale slot (0.036 to 3.2e5);
10.8%. A negative scale is not a distribution, so `inverse_observable_col`
returns NaN for every draw — which is why the first run scored **9 of 50
points** with *zero* finite symbolic draws at the rest. The model had not lost
accuracy, it had left the space of valid models. Linear interpolation is immune
by construction: a convex combination of valid vectors is valid.

Fixed with per-target link functions chosen from the training values —
`log` for strictly positive, `arctanh` for correlations, `logit` for weights in
(0,1), identity otherwise. Chosen automatically rather than by hand, because M6
shows the parameter layout changes. Result: 41 log, 120 arctanh, 23 identity;
in-sample R^2 median 0.786 -> 0.802, components with R^2 < 0.5 from 11 to 5, and
40 of 50 points now score.

**The result, and it is not a win.**

| | mean JS | joint MMD |
|---|---|---|
| floor (sampling noise at n=8000) | 0.0961 | 0.00051 |
| linear on the coarse subgrid | **0.1527** | 0.05002 |
| symbolic on the coarse subgrid | **0.2820** | 0.15994 |

Symbolic is 84.7% worse on average, better on 3 of 16 observables.

**But the failure is structured, and the structure is the finding.** Ordering
observables by how far linear sits from the floor:

| linear/floor | observables | SR change |
|---|---|---|
| > 2.8x | maxMuPt, Meff, HT, leadVisPt | Meff **+21%**, leadVisPt **+6%**, HT -25%, maxMuPt -103% |
| 1.15-1.25x | dPhiMETclose | **+3%** |
| < 1.15x | the other 10 | -26% to -212% |

correlation(linear's gap from the floor, SR's improvement) = **+0.574**.
Linear is already within 15% of the floor on **10 of 16** observables, and on
those SR averages -148%: there is nothing to win and a lot to lose, because a
global 8-term expression cannot compete with local interpolation where local
interpolation is essentially exact. On the 6 where linear is genuinely off the
floor, SR averages -48% and wins on 3.

**So "replace the interpolator" is the wrong framing.** The defensible one is a
hybrid: keep linear where it already reaches the floor, and use SR only for the
observables where it does not. That is testable with exactly this harness.

**Three caveats that keep this from being a verdict.**

1. **PD repair was needed on 38 of 40 symbolic predictions**, against 2 of 40
   for linear. `arctanh` keeps each entry inside (-1, 1) but says nothing about
   the assembled matrix being positive definite — the per-entry link does not
   solve the joint constraint. Every one of those 38 was perturbed by eigenvalue
   clipping before sampling, so the symbolic column above is scored on repaired
   matrices, not on what the model actually predicted. The Cholesky /
   partial-correlation parameterisation from section 2 is now a prerequisite,
   not a precaution.
2. **This benchmark is easier than M1.** Coarsening `2,2,1,1,1,1` leaves axes
   3-6 fully sampled, so linear interpolates in at most two dimensions here
   against all six in M1 — which is why linear scores 1.59x the floor here and
   2.62x in M1. Coarsening all six axes is needed before the two numbers can be
   compared.
3. **`maxMuPt` is the worst SR result (-103%) and the largest linear gap
   (3.88x)** — and it is one of M6's two stale observables, fitted in the
   committed scan *without* its point-mass weight. Its parameters do not mean
   what the current code thinks they mean, so it should be excluded or refitted
   before being counted either way.

**Cost:** ~4 minutes at 4 workers for fit plus 50 scored points, which is what
the harness was for. One operational note: an earlier 12-worker run of this
script triggered an automated CPU-pressure warning from the LxPlus service
(`some_avg300` 68.6%, these processes named top contributor). Defaults are now
4 workers, and anything larger belongs on Condor.

### M9 — PD fixed, regimes separated, and a hybrid that actually beats the incumbent  (2026-09-02)

Three changes on top of M8, in the order they mattered.

**1. Positive definiteness, by construction.** The correlation block is now
regressed in canonical partial-correlation coordinates (`corr.py`): take the
Cholesky factor of R, read off the partial correlations z in (-1,1) via the
standard recursion, and fit y = arctanh(z) over the unbounded reals. Any real
prediction maps back to a valid correlation matrix, so nothing needs repairing.
Verified on 3000 deliberately wild predictions (sigma = 5): zero invalid, min
eigenvalue exactly the 1e-6 ridge, zero `multivariate_normal` warnings.
Round-trip error on real fitted matrices 1.1e-4, comfortably below M5's
statistical sigma of 5.1e-3 for correlation entries.

Result on the identical M8 configuration: **PD repair 38/40 -> 0/40**, and the
JS scores essentially unchanged (mean 0.2820 both times). So M8's numbers were
sound after all — a useful negative result, since it was previously an
assumption. In-sample correlation R^2 falls 0.810 -> 0.729: partial
correlations are a harder target, but a valid one.

**2. Interpolation and extrapolation were being mixed, and that was my error.**
Striding an axis of length 4 by 2 trains on indices {0, 2} and leaves {1, 3} —
index 3 is *beyond* the last training point. Half the "held-out" points were
therefore extrapolations, `RegularGridInterpolator` was extrapolating with
`fill_value=None`, and the invalid parameter vectors that produced (linear
collapsed to 119 finite draws of 8000) were charged against interpolation.
`--regime interior|extrapolate` now separates them.

**3. The interior result, with all six axes coarsened** (8x8x4x4x4x4 ->
4x4x2x2x2x2, 256 training points, 38 scored interior points):

| | mean JS | vs floor | joint MMD |
|---|---|---|---|
| floor | 0.1053 | 1.00x | 0.00053 |
| linear | 0.2258 | 2.14x | 0.09960 |
| symbolic | 0.2521 | 2.39x | 0.10975 |

Symbolic is **11.7% worse**, down from 86.4% in the two-axis case, and now wins
on **5 of 16** — `HT` -41%, `Meff` -37%, `nConst` -27%, `leadVisPt` -26%,
`dPhiMETclose` -2%. Joint MMD is near parity. This is M8's +0.574 correlation
playing out: the harder the interpolation problem, the better a global closed
form looks, and the observables it wins are the ones M1 flagged.

**4. Extrapolation: symbolic is much worse, not better.** Outside the training
hull, symbolic is 86.7% worse and wins only 2 of 16, while linear barely
degrades. This kills an assumption worth stating plainly, because it was mine:
"a global closed form should extrapolate better than local interpolation" is
false *for this basis*. The library is dominated by degree-2 polynomials
(`mZ^2`, pairwise products), and polynomials are the worst possible
extrapolators — inputs are rescaled to [0,1] over the *training* range, so an
extrapolated point has scaled value > 1 where squares and products amplify.
Extrapolation would need a basis restricted to asymptotically tame terms (logs,
fitted power laws), which is an argument for PySR with a constrained operator
set rather than for this stage-1 library. Only 13 points scored, so treat the
magnitude as indicative.

**5. The hybrid, and this is the result worth keeping.** Choose per observable
between linear and symbolic on one half of the interior points, then score on
the other, disjoint half:

| | mean JS | vs floor |
|---|---|---|
| floor | 0.1017 | 1.00x |
| linear | 0.2378 | 2.34x |
| symbolic | 0.2532 | 2.49x |
| **hybrid** | **0.2065** | **2.03x** |

**13.2% better than linear**, 18.5% better than symbolic. Symbolic was selected
for `leadVisPt`, `HT`, `Meff`, `nConst` — four of M1's six worst observables,
and physically coherent: three energy scales and a multiplicity, all strongly
`mZ`-dependent, are the quantities a closed form in `mZ` should capture and a
coarse log-spaced grid should struggle with. The shape observables (`tau1-3`,
`e2c`, `e3c`, `jetThrust`) keep linear, which is right — M8 showed linear
already sits near the floor for those.

**Caveats.** Selection on 19 points is noisy: `leadJetMass` and
`dPhiMETclose` were assigned to linear but symbolic beat it on the evaluation
half, so the hybrid is if anything understated. And `maxMuPt` remains one of
M6's two stale observables, so its numbers should not be read either way until
it is refitted with its point-mass weight.

**Where this leaves Design A.** Not a replacement for the interpolator — M8
settled that. But a per-observable hybrid is a real, honestly-measured 13%
improvement on the incumbent, concentrated exactly where the incumbent is
known to fail, and it costs one JSON file and no simulation.

### M10 — Design B: a learned quantile function beats the parametric family on most observables  (2026-09-04)

The production raw store finished (see [raw-store.md](raw-store.md) for
completeness), so Design B became possible. Implemented in
`src/run_regression/symbolic/quantile.py`.

**The simplification that makes stages 1-2 cheap.** Evaluating every point's
marginal at the *same* u-levels makes the design matrix B(u) shared, so each
point is x_p = B c_p and choosing a support means choosing columns of B whose
span best contains every x_p at once. The objective collapses to a single
Frobenius residual, `||(I - P_S) X||_F^2`, which greedy forward selection
minimises in seconds. No per-point search, no 16 000 independent SR runs — the
"one structure, many coefficient sets" question answered directly rather than
by unioning thousands of separate answers.

**Stage 1-2 result: 6 terms reproduce the marginals almost exactly.** Median
R^2 across observables **0.9994**, 21 of 28 above 0.999, on 512 grid points.
And the selected supports read as physics:

| pattern | observables | reading |
|---|---|---|
| `probit` chosen first | most | a Gaussian core |
| `(1-u)^-0.5` | leadVisPt, MET, HT, Meff, hemiMass1/2, leadJetMass, nConst | heavy right tail — falling spectra |
| `logit` first | HT, Meff, hemiMass1/2, leadJetMass, e3c | heavier-than-Gaussian both tails |
| `u^-0.25` | leadWidth, tau1-3, e2c, transSphericity | bounded-below shape variables |

Two observables sit far below the rest, and both were predicted by M7:
`dPhiMETfar` (0.978, the phi-wrap artefact) and `closeJetIsLead` (0.901,
binary).

**One bug of mine, worth recording because it inverted the conclusion.** The
basis is fitted on u in [0.0007, 0.9993], and I sampled u in [1e-6, 1-1e-6] —
evaluating the fit far outside where it was constrained, in the region where
`u^-0.25`, `log(u)` and `(1-u)^-0.5` all diverge. The result was not merely
inaccurate but *non-monotone*: 19-62% of draws needed the running-maximum
repair, and Q looked 6.9x the floor against `gennorm`'s 3.3x. Clipping draws
to the fitted range dropped the repair rate to 0.3-8% for the well-behaved
observables and reversed the verdict.

This is a genuine limitation, not an implementation detail: with ~10k events a
point, the 1e-6 quantile lies below the smallest event and is simply not
estimable. A non-parametric Q must truncate at its fitted range, whereas
`gennorm` extrapolates into the tails analytically. That is the one structural
advantage the parametric family keeps.

**Stage 3 comparison, fit on half a point's events and scored on the other
half, 60 grid points:**

| | mean JS | vs floor | beats gennorm |
|---|---|---|---|
| floor | 0.0430 | 1.00x | — |
| gennorm (incumbent) | 0.1057 | 3.32x | — |
| Q, 6 terms | 0.1802 | 5.03x | 1/26 |
| **Q, 9 terms** | **0.1372** | **4.06x** | **15/26** |

The means are dragged around by a few catastrophic observables, so the
per-observable picture matters more. On the **16 observables the scan actually
fits, Q9 wins 11**, and brings several close to the floor:

| observable | floor | gennorm | Q9 |
|---|---|---|---|
| HT | 0.0427 | 0.0637 | **0.0491** (1.15x floor) |
| Meff | 0.0427 | 0.0786 | **0.0497** (1.16x) |
| leadVisPt | 0.0430 | 0.0767 | **0.0576** |
| tau3 | 0.0463 | 0.0748 | **0.0587** |
| tau1 | 0.0471 | 0.0618 | **0.0525** |
| e2c | 0.0461 | 0.0590 | **0.0522** |
| leadWidth | 0.0470 | 0.0622 | **0.0520** |
| metPhi | 0.0458 | 0.1135 | **0.0481** |

**And it fails exactly where M7 said it would.** Every failure is an atom, a
discreteness, or an angular fold:

| observable | gennorm | Q9 | why |
|---|---|---|---|
| dPhiMETclose | 0.0858 | 0.2563 | point mass at +-pi, not modelled by Q |
| dPhiMETfar | (fit fails) | 0.6200 | phi-wrap bimodality (M7) |
| nJets | 0.6698 | 0.6604 | discrete count; both fail |
| dPhiMETdijet | (fit fails) | 0.4191 | 17.7% atom |
| ptBal, RT, fInv, maxElePt | mixed | worse | 8-20% atoms |

The monotonicity repair rate tracks this precisely: 0.3-8% for the continuous
unimodal observables, 15-52% for the atom and fold cases.

**Correction to the table above, and a result in its own right.** M10's
comparison selected the support *per grid point* (`select_support` was called on
a single column), not the consensus support from stages 1-2. That measures an
upper bound, and worse, it measures the version that cannot be interpolated --
per-point supports are precisely the Design C dead end. Re-run properly, with
the support selected on 200 points and scored on 60 disjoint ones:

| | vs floor | beats gennorm |
|---|---|---|
| gennorm | 3.70x | — |
| **Q9, shared support** | **3.52x** | 15/26 |
| Q9, per-point support | 3.56x | 16/26 |

**The shared support costs nothing** -- it is as good as letting every point
choose its own. That is the load-bearing assumption of the whole three-stage
design, and it holds. Per observable, with the consensus support:

| observable | floor | gennorm | Q9 shared | vs floor |
|---|---|---|---|---|
| transSphericity | 0.0446 | 0.0915 | **0.0484** | 1.09x |
| metPhi | 0.0440 | 0.1158 | **0.0456** | 1.04x |
| HT | 0.0423 | 0.0634 | **0.0481** | 1.14x |
| Meff | 0.0412 | 0.0679 | **0.0474** | 1.15x |
| leadVisPt | 0.0412 | 0.0737 | **0.0477** | 1.16x |
| fInv | 0.0442 | 0.2138 | **0.0722** | — |
| ptBal | 0.0447 | 0.1931 | **0.0823** | — |
| dPhiMETdijet | 0.0443 | 0.8145 | **0.1942** | — |
| nConst | 0.0433 | 0.0683 | 0.0692 | tie |
| MET, hemiMass1, jetThrust, tau2, tau3 | ~0.044 | ~0.046 | ~0.049 | slightly worse |
| dPhiMETclose | 0.0445 | 0.0526 | 0.1071 | worse (atom) |
| RT | 0.0451 | 0.0706 | 0.0918 | worse (atom) |
| dPhiMETfar | 0.0314 | 0.0411 | 0.5685 | far worse (phi wrap) |
| nJets | 0.0177 | 0.6786 | 0.6895 | both fail (discrete) |

Note the shape of it: where `gennorm` is already near the floor (`MET`,
`hemiMass1`, `jetThrust`, `tau2/3`) Q is marginally worse, and where `gennorm`
is far off (`transSphericity` 2.05x, `metPhi` 2.63x, `fInv` 4.84x, `ptBal`
4.32x, `dPhiMETdijet` 18.4x) Q is dramatically better -- the same
"wins-where-the-incumbent-fails" pattern M8 found for Design A, now at the
marginal level rather than the interpolation level. It suggests the same
per-observable hybrid conclusion.

**So the missing piece is the mixture, which was already in the plan.** Q needs
p0(p) for the boundary atom plus Q on the continuous part, and the angular
observables need the `abs` fold applied before fitting rather than after. Until
then Design B is a real improvement on the continuous unimodal majority and a
regression on the rest.

**Not yet done: Design B's own stage 3 over physics.** Each point now has 9
coefficients over a shared basis — a fixed-length, interpolable vector, which
was the whole point — but mapping those to physics is Design A's machinery
applied to different numbers, and has not been run. That is the natural next
step, and it is also the measurement that decides whether the superset is
*better conditioned* for interpolation than Box-Cox + `gennorm`, which is
Design B's real promise beyond per-point fit quality.

---

## Appendix: terminology, and how symbolic regression actually works

Written because this document uses two quite different things that both get
called "symbolic regression", and because the quantile vocabulary in section 5
is not standard in this project's other docs.

### Quantile functions and "u-levels"

A distribution can be written three equivalent ways: the density `f(x)`, the
cumulative distribution `F(x) = P(X <= x)`, and the **quantile function**
`Q(u) = F^-1(u)` — the value below which a fraction `u` of the distribution
lies. `Q(0.5)` is the median, `Q(0.99)` the 99th percentile. scipy calls it
`ppf`, the percent-point function.

A **u-level** is simply one of the probabilities `u` at which we choose to
evaluate `Q`. To fit `Q` as a function we need (input, output) pairs, so we pick
a grid of `u` values and, at each grid point of the scan, read off that point's
*empirical* quantile from its 20 000 events: `u = 0.5` gives the median of those
events, `u = 0.99` the 99th percentile. Those pairs are the training data.

`u_levels()` spaces them by **probit** rather than uniformly: `u = Phi(z)` for
`z` evenly spaced in [-3.2, 3.2], giving 199 levels spanning
`u` in [0.00069, 0.99931]. Uniform spacing would put half the levels inside the
central 50% of the distribution, where nothing interesting happens, and one or
two in the tails, where the model actually fails and where `sample_svj_new`'s
NaN draws come from.

The range also explains M10's bug. With ~20k events per point the smallest
estimable quantile is about 1/20000 = 5e-5; asking for `Q(1e-6)` is asking about
an event that is not in the sample. Fitting on [0.0007, 0.9993] and then
*sampling* at `u = 1e-6` evaluates the basis outside its support, where terms
like `u^-0.25` diverge.

**Why regress `Q` and not `f` or `F`.** Sampling by inverse transform is
`u ~ Uniform(0,1)`, `x = Q(u)` — so `Q` is the sampling primitive, and this
pipeline already computes it: `inverse_observable_col` *is* `Q`. Regressing a
density instead imposes `integral f = 1`, a constraint free-form regression will
not respect, and requires binning that biases the tails. Regressing a CDF gives
a conveniently bounded monotone target but then needs a numerical inversion per
draw. `Q`'s only structural requirement is that it increase in `u`.

### How symbolic regression works

The idea is to fit not the *parameters* of a fixed functional form, but the
functional **form** itself — searching over expressions built from variables,
constants and operators for one that is both accurate and short. Two families
exist, and this project uses the second while the literature mostly means the
first.

**(a) Genetic programming / evolutionary search.** An expression is a tree:
operators at the nodes, variables and constants at the leaves. Keep a population
of trees; score each on accuracy *and* complexity (node count); evolve by
mutation (swap an operator, replace a subtree) and crossover (exchange subtrees);
numerically optimise the constants inside each candidate. The output is not one
expression but a **Pareto front** trading accuracy against complexity, from
which a human picks. This is what PySR does, and what people usually mean by
"symbolic regression".

**(b) Sparse regression over a fixed library.** Pre-specify a dictionary of
candidate terms and fit a *linear* combination of them,
`y = sum_j c_j phi_j(x)`, using sparse selection (Lasso, orthogonal matching
pursuit, greedy forward selection) to keep only a few. Far faster and fully
deterministic, but it can only express combinations of terms you thought to
include — it cannot invent `exp(-x^2/a)` with a fitted `a` inside.

Both designs here are type (b):

| | library | selection | targets |
|---|---|---|---|
| Design A (`basis.py`) | powers, logs, reciprocals and pairwise products of the 6 physics axes | OMP or LassoCV | the 184 fitted parameters |
| Design B (`quantile.py`) | 13 standard quantile-function components (probit, logit, `-log(1-u)`, `(1-u)^-1/2`, ...) | greedy multi-task | each grid point's empirical quantiles |

PySR is the escalation to (a), and stage 1's interface — return named terms plus
coefficients — is deliberately what it would replace.

**Orthogonal matching pursuit**, used above, is the greedy version: start from
an empty support; repeatedly add whichever library term most reduces the
residual; refit all coefficients on the chosen set at each step. Lasso instead
adds an L1 penalty and lets the penalty strength decide how many terms survive.

**Multi-task selection** is what makes Design B's stages 1-2 cheap. There are
thousands of targets (one per grid point) but they share the *same* design
matrix `B(u)`, because every point is evaluated at the same u-levels. Asking
"how well does support `S` serve all of them at once" is then just the residual
of projecting the whole data matrix onto `span(B_S)` — one Frobenius norm rather
than thousands of separate searches. That is the "one structure, many
coefficient sets" requirement answered directly.

### Reading

Author, title and venue below are reliable; the arXiv numbers are from memory
and worth checking.

*Symbolic regression.*
- Koza, *Genetic Programming* (MIT Press, 1992) — the origin of tree-based GP.
- Schmidt & Lipson, "Distilling Free-Form Natural Laws from Experimental Data",
  *Science* **324** (2009) — the paper that made SR-for-physics visible; the
  Eureqa system.
- Cranmer et al., "Interpretable Machine Learning for Science with PySR and
  SymbolicRegression.jl" (arXiv ~2305.01582) — the tool to read if we escalate.
- Cranmer et al., "Discovering Symbolic Models from Deep Learning with Inductive
  Biases", NeurIPS 2020 — SR applied downstream of a neural network.
- La Cava et al., "Contemporary Symbolic Regression Methods and their Relative
  Performance" (SRBench), NeurIPS 2021 Datasets & Benchmarks — honest
  head-to-head benchmarking, useful for calibrating expectations.

*Sparse-library regression, which is what is actually implemented here.*
- Brunton, Proctor & Kutz, "Discovering governing equations from data by sparse
  identification of nonlinear dynamical systems" (SINDy), *PNAS* **113** (2016).
  Design A's `basis.py` is SINDy's method applied to a parameter map instead of
  a dynamical system.
- Tropp & Gilbert on orthogonal matching pursuit (2007); Yuan & Lin (2006) on
  the group lasso, for the multi-task variant.

*Quantile functions — the literature closest to Design B.*
- Gilchrist, *Statistical Modelling with Quantile Functions* (Chapman & Hall,
  2000) — the standard book-length treatment.
- Keelin & Powley, "Quantile-Parameterized Distributions", *Decision Analysis*
  (2011), and Keelin, "The Metalog Distributions", *Decision Analysis* (2016).
  **Design B is essentially fitting a metalog**: the metalog family is a linear
  combination of terms in `u` and `logit(u)`, very close to `quantile.py`'s
  basis. That literature also treats the monotonicity problem under the name
  *feasibility*, which is exactly M10's repair-rate issue and is worth reading
  before hand-rolling a fix.
- Tukey's g-and-h and lambda families — earlier flexible quantile-function
  distributions, and the ancestors of the `probit`, `probit^2*sgn`, `probit^3`
  terms in the basis (Cornish-Fisher style skew and kurtosis corrections).

*Copulas and correlation matrices, for context on the joint model.*
- Nelsen, *An Introduction to Copulas* — Sklar's theorem and the Gaussian
  copula this project uses.
- Lewandowski, Kurowicka & Joe (2009) on vines and the extended onion method —
  the origin of the canonical partial-correlation parameterisation in
  `corr.py`, and the same device Stan uses for LKJ correlation priors.

### M11 — Design B with the fold and the atom: a 23-31% improvement, and M7's diagnosis confirmed  (2026-09-04)

M10 left Q failing on exactly two structures, both of which M7 had already
named. Both are now handled the way the incumbent handles them — fold first
(`abs_value`'s inverse "alternates signs; assumes symmetric signed
distribution"), then model a boundary atom as a discrete mixture component with
Q on the continuous remainder (`fit_marginal` / `sample_marginal`).

Support selected on 200 grid points, method chosen on 60 more, scored on 60
**disjoint** points:

| | mean JS | vs floor |
|---|---|---|
| floor | 0.0417 | 1.00x |
| gennorm (incumbent) | 0.1229 | 3.69x |
| **Q9 + fold + atom** | **0.0846** | **2.83x** |
| hybrid (per-observable pick) | 0.0842 | 2.81x |

**-31.5% on mean JS against the incumbent** (-23% on the ratio-to-floor
aggregation). Note what the hybrid does *not* buy: 2.81x against Q's 2.83x. Per-
observable selection adds essentially nothing here, unlike Design A where it was
the whole result. Q with the two fixes is simply better across the board, so the
honest recommendation is Q everywhere rather than a per-observable table --
fewer moving parts, and no selection noise to relitigate as the grid grows.

**The fold fix confirms M7 outright.** `dPhiMETfar` went from 0.5685 (bare Q,
M10) to **0.0395** — better than `gennorm`'s 0.0484 — purely by folding before
fitting. M7 had predicted this from the histogram: one physical peak split by
the phi wrap, not a bimodality. Nothing about the basis changed.

**The atom fix does the same for the mixture cases:**

| observable | gennorm | bare Q (M10) | Q + fix |
|---|---|---|---|
| dPhiMETfar | 0.0484 | 0.5685 | **0.0395** |
| dPhiMETdijet | 0.8130 | 0.4191 | **0.1876** |
| fInv | 0.2039 | 0.1361 | **0.1109** |
| ptBal | 0.1659 | 0.1921 | **0.0931** |
| metPhi | 0.1119 | 0.0481 | **0.0455** |
| RT | 0.0742 | 0.1853 | **0.0629** |
| transSphericity | 0.0880 | 0.0550 | **0.0481** |

**Q is selected for 16 of 26** observables: leadVisPt, leadWidth, MET, maxMuPt,
transSphericity, ptBal, dPhiMETdijet, e2c, tau1, dPhiMETfar, metPhi, HT, RT,
Meff, leadJetMass, fInv. `gennorm` is retained where it already sits near the
floor (`hemiMass1` 1.10x, `jetThrust` 1.05x, `e3c` 1.12x, `maxElePt` 1.00x) —
the same pattern as M8: neither method wins everywhere, and the incumbent is
hard to beat where it is already as good as the data allows.

**Two things still not fixed.** `nJets` (0.667 for both) is discrete and drags
the mean badly; excluding it would move every aggregate. `nConst` (gennorm
0.0833, Q 0.0799) is also discrete and only marginally improved. Neither is a
quantile-function problem — a discrete observable needs a discrete model, and
that remains M7's open item.

**Status: exploratory, not final.** This is measured on 320 grid points from 16
shards of the truth stream, one split, one choice of k = 9. It has not been run
on the Delphes stream, the u-level count and basis size have not been tuned, and
Design B's stage 3 — mapping the 9 coefficients over physics — is still not
done. That last one remains the measurement that decides whether the superset is
*better conditioned* for interpolation than Box-Cox + `gennorm`, which is B's
real promise beyond per-point fit quality.

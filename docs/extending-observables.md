# Adding transforms, distributions, and observables

Everything lives in `src/observables.py`.

## Adding a new invertible transform

Add an entry to the `TRANSFORMS` dict:

```python
TRANSFORMS['my_transform'] = {
    'fit':      lambda x, **fixed: (transformed_x, (fitted_param1, ...)),
    'forward':  lambda x, params, **fixed: transformed_x,
    'inverse':  lambda y, params, **fixed: original_x,
    'n_fitted': 0,     # number of data-fitted params (0 if all fixed)
    'requires': lambda **fixed: (lo, hi),   # input range, or None
    'desc':     'Human-readable description',
}
```

- `fit(x, **fixed_params)` applies the transform to data `x` AND fits any
  free parameters; returns `(y_array, tuple_of_fitted_params)`.
- `forward(x, params, **fixed_params)` applies with known `params` (no fitting).
- `inverse(y, params, **fixed_params)` maps back to original space.
- `requires` returns `(lo, hi)` — events outside this range are discarded
  *before* this transform is applied. Return `None` if there is no restriction.

## Adding a new fitting distribution

Add an entry to the `DISTRIBUTIONS` dict:

```python
DISTRIBUTIONS['my_dist'] = {
    'dist':        scipy.stats.my_dist,   # scipy continuous_rv_generic
    'n_params':    3,                      # number of params returned by .fit()
    'fit_init':    lambda x: ((), {'floc': 0.0}),  # initial args/kwargs for MLE
    'input_range': (0.0, np.inf),          # informational — support of the dist
    'desc':        'Human-readable description',
}
```

The `fit_init` callable receives the data column and returns `(args, kwargs)`
forwarded to `dist.fit(x, *args, **kwargs)`. Use `f<n>=` kwargs to fix
parameters during MLE (e.g. `floc=0` to fix location to zero).

## Adding a new observable

Adding a new observable is a two-step process.

**Step 1 — C++: `src/generate_events/svj_observables_common.h` + both binaries**

The observable computation itself lives in **one shared, header-only file**,
`svj_observables_common.h`, used identically by the truth-level binary
(`svj_regression.cc`) and the Delphes-level binary
(`svj_regression_delphes.cc`) — a new observable is computed once, not twice,
so the two streams can never define it differently.

1. Add a field to the `SvjObservables` struct and compute it inside
   `computeSvjObservables()` in `svj_observables_common.h`, using the already-
   classified `SvjJetInputs` (clustered jets, constituent arrays, sphericity
   sums, etc. — the same inputs both binaries build).
2. Append its name to `OBS_NAMES` at the top of **each** binary
   (`svj_regression.cc` and `svj_regression_delphes.cc`), in the same
   position in both files:
   ```cpp
   static const std::vector<std::string> OBS_NAMES = {
       ...,
       "myObs",   // ← add here, same position in both files
   };
   ```
3. Append the field to `data.push_back({obs. ...})` in **each** binary's event
   loop, in the same position as its `OBS_NAMES` entry.
4. Rebuild both: `make svj_regression svj_regression_delphes` from the project
   root.

If the new observable is meaningless for Delphes-eflow-only input (i.e. it
needs truth-level invisible-particle info, like `nInvClose`/`fInv`), it's
fine to still add it to `svj_regression_delphes.cc`'s `OBS_NAMES` — passing
an empty `invis_ptcls` list (as that binary always does) makes such fields
evaluate to a well-defined `0` rather than requiring a separate code path;
just don't select it in a Delphes scan's `--obs`/`DEFAULT_SCAN` if a
non-trivial value matters for your analysis. See `docs/setup_delphes.md` for
background on the truth/Delphes split.

**Step 2 — Python: `src/observables.py`**

Add an entry to the `OBSERVABLES` dict:

```python
OBSERVABLES['myObs'] = {
    'col':             'myObs',    # must match the name in OBS_NAMES (step 1)
    'pipeline':        [('boxcox', {})],
    'distribution':    'gennorm',
    'default_include': False,
    'label':           r'My observable',
    'desc':            'One-line physics description',
}
```

For a **derived** observable (ratio of two base observables; GUI display only,
no regression):

```python
OBSERVABLES['obsA/obsB'] = {
    'col':             None,
    'pipeline':        None,
    'distribution':    None,
    'default_include': False,
    'label':           r'$a/b$',
    'desc':            'Derived: obsA / obsB.  GUI display only.',
    'derive_cols':     ('obsA', 'obsB'),  # observable names (numerator, denominator)
}
```

For derived observables, `default_include` controls **GUI visibility only** —
derived observables are never part of the regression scan regardless of this
flag.  Set `'default_include': True` to show the ratio in the explorer
dropdowns and cut-slider panel by default; leave it `False` to hide it (it will
not appear unless you edit `observables.py` and re-run the explorer).

To include a **base** observable in the default regression scan, set
`'default_include': True`, or add its name to `DEFAULT_SCAN` at the bottom of
`observables.py`. Then re-run `scan_svj.py`.

## Assigning a custom transform pipeline to an observable

The `pipeline` is an ordered list of `(transform_name, fixed_params_dict)` pairs.
Transforms are applied left-to-right; the fitting distribution receives the output
of the last transform. Example for a signed-angle observable:

```python
'pipeline': [('abs_value', {}), ('boxcox', {})],
```

Example with a fixed-parameter flip (for thrust-like variables):

```python
'pipeline': [('affine_flip', {'a': 1.0}), ('boxcox', {})],
```

The parameter `a=1.0` is the fixed upper bound of `affine_flip`; changing `a`
requires editing the observable entry in `observables.py`.

## Modelling an observable with a point mass at a boundary value

Some observables have a physical spike at a specific boundary value — for
example, `maxMuPt = 0` when no muon is produced, or `fInv = 0` when no jet
constituents are invisible. These events would ordinarily be discarded by the
Box-Cox range check.

Adding an optional `point_mass` key enables a mixture-distribution model:

```python
OBSERVABLES['maxMuPt'] = {
    ...
    'point_mass': {
        'value':   0.0,    # location of the point mass
        'tol':     1e-10,  # |x - value| < tol → event is a PM event
        'symmetric': False,  # if True, also treat x ≈ -value as a PM event
        'min_p0':  0.01,   # minimum fraction required to activate the mixture
    },
}
```

**How it works.**
Let `p0` be the fraction of events within tolerance of the boundary value.
The fitted CDF is:

```
F_mix(x) = p0                          at x = boundary_value
F_mix(x) = p0 + (1 - p0) * F_cont(x)  for x in the continuous part
```

`F_cont` is the usual pipeline-transform + parametric-distribution CDF fitted
to the non-PM events only. The probability integral transform then yields proper
`U[0, 1]` marginals (point-mass events receive a uniform draw from `[0, p0]`
during fitting, and the deterministic midpoint `p0/2` during forward evaluation).

**Parameter layout.** When `point_mass` is set, `p0` is stored as the *first*
element of that observable's parameter block in `param_flat`.
`n_fitted_params` returns one more than it would without the key.

**`min_p0` threshold.** If the measured `p0 < min_p0` (default 0.01), the
mixture is skipped at runtime (the observable is treated as purely continuous)
but `p0` is still stored so that interpolation over the grid remains smooth.
If `p0 = 1` (all events are at the boundary) a `ValueError` is raised.

**`symmetric=True`.** For observables like `dPhiMETclose` that spike at both
`+π` and `−π`, set `symmetric: True`. Events within tolerance of either
`+value` or `−value` are treated as PM events; the inverse samples the sign
uniformly at random.

**`rng` parameter.** `fit_observable_col` accepts an optional `rng` keyword
(a `numpy.random.Generator`) to make the randomised PIT reproducible:

```python
rng = np.random.default_rng(42)
y_std, params = fit_observable_col(x_col, pipeline, dist_name,
                                   point_mass=pm_spec, rng=rng)
```

**Disabling the mixture.** To revert an observable to purely continuous
behaviour, remove the `point_mass` key (or set it to `None`). No other code
changes are needed.

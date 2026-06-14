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

**Step 1 — C++: `src/generate_events/svj_regression.cc`**

1. Declare a local variable for the new quantity inside `runWorker()` (near the
   other observable declarations, e.g. `double myObs = 0.0;`).
2. Compute its value using the existing local variables available at that point
   (jet kinematics, constituent arrays, particle-loop sums, etc.).
3. Append its name to `OBS_NAMES` at the top of the file:
   ```cpp
   static const std::vector<std::string> OBS_NAMES = {
       ...,
       "myObs",   // ← add here
   };
   ```
4. Append the variable to `data.push_back({...})` at the end of `runWorker()`,
   in the same position as its `OBS_NAMES` entry.
5. Rebuild: `make svj_regression` from the project root.

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

To include a new observable in the default regression scan, set
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

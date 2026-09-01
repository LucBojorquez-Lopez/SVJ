# Parameter interpolation API

`src/helpers.py` provides fast interpolation so the joint observable
distribution can be sampled at any parameter point without re-running PYTHIA.

## The default scan

Out of the box, the API loads the **working example** shipped with the
repository:

```
simulated/svj/working_example/svj_scan.npz
```

A complete 4-axis × 11-observable scan, validated end to end:

| Axis | Range | Points |
|------|-------|--------|
| `mZ` | 500 – 4000 GeV | 16 (log-spaced) |
| `rinv_pion` | 0.05 – 0.70 | 8 |
| `mRho` | 15.5 – 30.0 GeV | 8 |
| `alphaD` | 0.1 – 0.6 | 8 |

Held fixed: `mq = 4.0`, `jetR = 1.0`, `Brmu = 0.3`.
Derived: `mPi = mRho·(8/15.5)`, `LambdaDQCD = mRho·(5/15.5)`, `rinv_rho = rinv_pion`.

Observables: `leadVisPt, leadWidth, MET, maxMuPt, jetThrust, hemiMass1, e2c,
e3c, tau1, tau2, tau3`.

## Basic use

```python
import sys; sys.path.insert(0, 'src')
import helpers

# Grid bounds: dict {axis_name: vals_array}
grid_bounds = helpers.svj_grid_bounds()
# {'mZ': array([500., ..., 4000.]), 'rinv_pion': ..., 'mRho': ..., 'alphaD': ...}

# Interpolate fitted parameters anywhere inside the grid
R_upper, obs_params, param_offsets, obs_names = helpers.interpolate_svj_params(
    {'mZ': 1500, 'rinv_pion': 0.3, 'mRho': 20.0, 'alphaD': 0.4})

# Sample the joint distribution
X = helpers.sample_svj_new(R_upper, obs_params, param_offsets, obs_names,
                           n_samples=50_000)
# X: (50000, 11) in original physical units, columns ordered as obs_names
```

The dict keys must match the scan axis names stored in the NPZ. Extra keys are
ignored; a **missing** key raises `KeyError`, and a value outside the grid
raises `ValueError: One of the requested xi is out of bounds`.

## Using a different scan

`set_svj_scan_path` swaps the loaded NPZ and clears all caches. The larger
6-axis scan is shipped alongside the default:

```python
helpers.set_svj_scan_path('simulated/svj/svj_scan.npz')
list(helpers.svj_grid_bounds())
# ['mZ', 'rinv_pion', 'mPiOverLambda', 'LambdaDQCD', 'alphaD', 'jetR']

helpers.set_svj_scan_path(None)   # restore the default working example
```

Note that the axis **names** differ between scans — the 6-axis scan
parameterises the dark sector by `mPiOverLambda` and `LambdaDQCD` rather than
`mRho`, so parameter dicts are not interchangeable. Always build the dict from
`svj_grid_bounds()` keys rather than hardcoding names.

Two module constants record the default:

```python
helpers.DEFAULT_SCAN_DIR   # .../simulated/svj/working_example
helpers.DEFAULT_SCAN_NPZ   # .../simulated/svj/working_example/svj_scan.npz
```

## Known limitation — NaNs in the extreme tail

`sample_svj_new` returns a small fraction of `NaN` values: about **0.01 %** on
the working example and **0.03 %** on the 6-axis scan.

This is inherent to the Box-Cox marginal model. For a fitted exponent λ > 0,
the Box-Cox inverse is only defined where `λy + 1 > 0`, i.e. `y > -1/λ`, but
the generalized-normal fitted in that space has unbounded support and places a
little probability mass below the bound. Those draws invert to `NaN` rather
than to a physical value. `leadVisPt` (largest λ, ~0.75) dominates the effect.

The values are genuinely undefined rather than merely extreme, so mask them:

```python
X = helpers.sample_svj_new(R_upper, obs_params, param_offsets, obs_names,
                           n_samples=50_000)
X = X[np.isfinite(X).all(axis=1)]        # drop affected rows
```

Because the copula ties the columns together, dropping whole rows (rather than
per-column masking) is what preserves the joint structure. Draw ~1 % extra
samples if you need an exact count.

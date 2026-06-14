# Parameter interpolation API

Once a scan NPZ is present, `src/helpers.py` provides fast interpolation so that
the joint observable distribution can be sampled at any parameter point without
re-running PYTHIA.

```python
import sys; sys.path.insert(0, 'src')
import helpers

# Get grid bounds: dict {axis_name: vals_array} for each scan axis
grid_bounds = helpers.svj_grid_bounds()
# e.g. {'mZ': array([500,...,4000]), 'mRho': array([10,...,30]), ...}

# Interpolate fitted parameters at any point inside the grid.
# Pass a dict with a value for each scan axis.
R_upper, nu, obs_params, param_offsets, obs_names = helpers.interpolate_svj_params(
    {'mZ': 1500, 'mRho': 20, 'rinv_pion': 0.3, 'alphaD': 0.4})

# Sample from the interpolated distribution
X = helpers.sample_svj_new(R_upper, nu, obs_params, param_offsets, obs_names,
                            n_samples=50_000)
# X has shape (n_samples, n_obs) in original physical units
```

The dict keys must match the scan axis names stored in the NPZ (i.e. whatever was
in `[scan]` when the scan was run). Extra keys are ignored.

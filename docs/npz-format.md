# NPZ file formats

## New format (`simulated/svj/svj_scan.npz`)

The number and names of scan axes are dynamic (set by `[scan]` in the config).
The example below shows the default 4-axis scan.

| Key | Shape | Description |
|-----|-------|-------------|
| `axis_names` | `(K,)` | Names of the K scan axes (e.g. `['mZ','jetR','rinv_pion','Brmu']`) |
| `{name}_vals` | `(N_k,)` | Grid values for each axis (one array per axis name) |
| `param_flat` | `(N_0, …, N_{K-1}, total_params)` | All fitted parameters, flat over the K-D grid |
| `param_offsets` | `(n_obs+1,)` | `param_flat[..., offsets[i]:offsets[i+1]]` = params for obs i |
| `corr_start` | scalar int | `param_flat[..., corr_start:]` = Gaussian copula correlation upper-triangle |
| `obs_names` | `(n_obs,)` | Observable names in regression order |
| `scan_params` | `(N_0, …, N_{K-1}, n_phys)` | All resolved physics params (scan + derived) per point |
| `scan_param_names` | `(n_phys,)` | Names for the `scan_params` columns |

`total_params = param_offsets[-1] + n_corr` where `n_corr = n_obs*(n_obs-1)//2`.

The companion `svj_scan_meta.json` stores `scan_axes`, `fixed_params`, and
`derived_exprs` so that downstream tools can reconstruct any parameter value.

## V1 format (`simulated/v1/regression_scan.npz`)

Older 3-observable MVT fit (pT, jet width, MET). Used by `old_version/helpers.py` only.

| Key | Shape | Description |
|-----|-------|-------------|
| `params` | `(grid..., 10)` | `[mu_pT, mu_logW, mu_logMET, S00, S01, S02, S11, S12, S22, nu]` |
| `param_names` | `(10,)` | Names for the `params` columns |
| `scan_params` | `(grid..., 6)` | `[mZ, mRho, mPi, LambdaQCD, rinv, alphaD]` per point |
| `scan_param_names` | `(6,)` | Names for the `scan_params` columns |
| `mZ_vals`, `mRho_vals`, `rinv_vals`, `alphaD_vals` | 1-D | Grid axis values |

> **Note:** The gennorm + MVN copula format (keys `corr_params`, `transform_params`, `obs_names`)
> is in `old_version/simulated/gennorm/gennorm_scan.npz`, used by the `old_version/` GUI.

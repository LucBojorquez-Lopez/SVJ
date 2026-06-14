# NPZ file formats

## New format (`simulated/svj/svj_scan.npz`)

The number and names of scan axes are dynamic (set by `[scan]` in the config).
The example below shows the default 4-axis scan.

| Key | Shape | Description |
|-----|-------|-------------|
| `axis_names` | `(K,)` | Names of the K scan axes (e.g. `['mZ','mRho','rinv_pion','alphaD']`) |
| `{name}_vals` | `(N_k,)` | Grid values for each axis (one array per axis name) |
| `param_flat` | `(N_0, …, N_{K-1}, total_params)` | All fitted parameters, flat over the K-D grid |
| `param_offsets` | `(n_obs+1,)` | `param_flat[..., offsets[i]:offsets[i+1]]` = params for obs i |
| `corr_start` | scalar int | `param_flat[..., corr_start:corr_start+n_corr]` = MVT corr upper-triangle |
| `nu_idx` | scalar int | `param_flat[..., nu_idx]` = MVT degrees of freedom ν |
| `obs_names` | `(n_obs,)` | Observable names in regression order |
| `scan_params` | `(N_0, …, N_{K-1}, n_phys)` | All resolved physics params (scan + derived) per point |
| `scan_param_names` | `(n_phys,)` | Names for the `scan_params` columns |

`total_params = param_offsets[-1] + n_corr + 1` where `n_corr = n_obs*(n_obs-1)//2`.

The companion `svj_scan_meta.json` stores `scan_axes`, `fixed_params`, and
`derived_exprs` so that downstream tools can reconstruct any parameter value.

## V1 format (`simulated/v1/regression_scan.npz`)

12-observable Box-Cox + gennorm marginals, MVN copula. Used by `old_version/` only.

| Key | Shape | Description |
|-----|-------|-------------|
| `corr_params` | `(grid..., 66)` | MVN correlation upper-triangle (12 observables) |
| `transform_params` | `(grid..., 12, 4)` | `[lam, beta, loc, scale]` per observable |
| `obs_names` | `(12,)` | Observable names |
| `scan_params` | `(grid..., 6)` | `[mZ, mRho, mPi, LambdaQCD, rinv, alphaD]` per point |
| `mZ_vals`, ... | 1-D | Grid axis values |

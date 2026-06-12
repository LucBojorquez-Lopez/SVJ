# mySVJ — SVJ Interpolation Framework

Semi-Visible Jet (SVJ) parameter-space interpolation using PYTHIA8 + FastJet.
The framework scans a 4-D parameter grid, fits a per-observable transform pipeline
+ Multivariate-t copula at each point, and provides fast interpolation so that
the joint observable distribution can be sampled at any parameter point without
re-running PYTHIA.

---

## Directory layout

```
mySVJ/
├── src/
│   ├── generate_events/
│   │   ├── svj_regression.cc      C++ event generator (PYTHIA8 + FastJet)
│   │   ├── svj_regression.cfg     Default physics + run parameters
│   │   └── Makefile               Build target: make svj_regression
│   ├── run_regression/
│   │   ├── scan_svj.py            Main scan script (grid → svj_scan.npz)
│   │   ├── scan_regression.py     Legacy 3-param MVT scan (regression_scan.npz)
│   │   ├── scan_regression.cfg    Shared config for both scan scripts
│   │   └── fit_raw.py             Re-fit from saved raw events (no re-simulation)
│   ├── gui/
│   │   └── svj_explorer.py        Jupyter notebook interactive explorer
│   ├── observables.py             Observable / transform / distribution registry
│   ├── helpers.py                 NPZ interpolation helpers + KLD utilities
│   └── diagnostics.py             Transform pipeline diagnostic plots
├── old_version/                   Archived v1 (gennorm + MVN copula, self-contained)
│   ├── helpers.py
│   ├── svj_explorer.py
│   ├── gui.ipynb
│   └── simulated/{v1,gennorm}/   Copies of the v1 NPZ files
├── simulated/
│   ├── svj/svj_scan.npz           Output of scan_svj.py  (new format)
│   └── v1/regression_scan.npz    Output of scan_regression.py (v1 format)
└── data/regression/               TSV files written by the event generator
```

---

## 1 · Running a scan

### 1.1 Build the event generator

```bash
cd src/generate_events
make svj_regression
cd ../..
```

### 1.2 Configure the scan

All scan parameters live in `src/run_regression/scan_regression.cfg`.
Edit this file before running.

**Grid axes** — these define the 4-D hypercube that will be scanned:

| Key | Default | Description |
|-----|---------|-------------|
| `mZ_min`, `mZ_max`, `mZ_n` | 500, 4000, 8 | Z′ mass range (GeV) and number of grid points |
| `mRho_min`, `mRho_max`, `mRho_n` | 10, 30, 8 | Dark rho mass range (GeV) |
| `rinv_min`, `rinv_max`, `rinv_n` | 0.05, 0.70, 8 | Dark pion invisible BR |
| `alphaD_min`, `alphaD_max`, `alphaD_n` | 0.10, 0.80, 8 | Dark coupling α_D |

**Fixed physics parameters** (same at every grid point):

| Key | Default | Description |
|-----|---------|-------------|
| `mq` | 4.0 | Dark quark mass (GeV) |
| `Brl` | 0.3 | Fraction of *visible* dark-rho decays going to μ⁺μ⁻; `lep_br = Brl*(1−rinv2)` |
| `jetR` | 1.0 | Anti-kT jet radius |
| `nEvent` | 50000 | Events per grid point |
| `nWorkers` | 1 | Internal C++ threads per grid point (keep 1 when outer parallelism is used) |

**Derived parameters** (computed automatically, not set here):

| Parameter | Formula | Notes |
|-----------|---------|-------|
| `mPi` | `mRho × 8/15.5` | Dark pion mass |
| `LambdaDQCD` | `mRho × 5/15.5` | Dark QCD confinement scale |
| `rinv2` | `= rinv` | Dark rho invisible BR (tied to dark pion invisible BR in v2 scan; see note) |

> **Note on `rinv2`**: The current scan ties the dark-rho invisible BR to the dark-pion invisible BR
> (`rinv2 = rinv`).  This is a deliberate model choice recorded in
> `scan_regression.cfg`.  The C++ generator uses the new BR parameterisation where
> `Brl` is the *fraction of the visible* decays that go to leptons, so
> `lep_br = Brl*(1−rinv2)` and `bott_br = (1−Brl)*(1−rinv2)` always sum to 1.

**Parallelisation parameters** (`scan_svj.py` only):

| Key | Default | Description |
|-----|---------|-------------|
| `n_outer_workers` | 48 | Simultaneous grid points (Python `ProcessPoolExecutor`) |
| `checkpoint_every` | 200 | Save intermediate NPZ every N completions |

### 1.3 Run the scan — standard (no raw saving)

```bash
# From the project root (mySVJ/):
python src/run_regression/scan_svj.py src/run_regression/scan_regression.cfg
```

Output is written to `simulated/svj/svj_scan.npz` and `simulated/svj/svj_scan_meta.json`.

Progress is printed to stdout — one line per completed grid point:
```
[  1/4096]  (0,0,0,0)  ok    ETA 01:23:45  disc=12
[  2/4096]  (0,0,0,1)  ok    ETA 01:22:30  disc=0
...
```
`disc=N` is the number of events discarded at that grid point due to range checks
(events for which any observable's pipeline requirements are not met).

**Resume after interruption**: just re-run the same command.  Finished grid points
are detected via `np.isfinite` checks and skipped automatically.

### 1.4 Run the scan — with raw event saving (`--save-raw`)

The `--save-raw` flag saves the pre-transform observable values for every grid point
alongside the fitted parameters.  This allows re-fitting without re-running PYTHIA.

```bash
python src/run_regression/scan_svj.py src/run_regression/scan_regression.cfg \
    --save-raw
```

Additional output: `simulated/svj/svj_scan_raw.npz`
```
raw_flat       shape (total_events, n_obs)   — pre-transform values
raw_grid_flat  shape (total_events, 4)        — grid indices [i,j,k,l]
```

> **Storage estimate**: with `nEvent=2000` and 12 observables over a 8⁴ grid,
> raw data is ~4096 × 2000 × 12 × 8 bytes ≈ **786 MB**.

### 1.5 Select which observables to scan

The default set is `DEFAULT_SCAN` in `src/observables.py`
(currently 12 observables: leadVisPt, leadWidth, MET, maxMuPt, jetThrust,
hemiMass1, hemiMass2, e2c, e3c, tau1, tau2, tau3).

Override from the command line with `--obs`:

```bash
python src/run_regression/scan_svj.py scan_regression.cfg \
    --obs "leadVisPt,MET,tau1,tau2,tau3"
```

The observable names must match keys in the `OBSERVABLES` dict in `src/observables.py`,
and each must have a non-`None` `pipeline` and `distribution`.

### 1.6 SLURM array jobs

Split the grid across multiple array jobs:

```bash
# Submit N=4 jobs (job indices 0..3)
for i in 0 1 2 3; do
    sbatch --export=JOB_INDEX=$i,N_JOBS=4 my_slurm_script.sh
done

# In the SLURM script:
python src/run_regression/scan_svj.py scan_regression.cfg \
    --job-index $JOB_INDEX --n-jobs $N_JOBS

# After all jobs finish, merge:
python src/run_regression/scan_svj.py --merge --n-jobs 4
```

### 1.7 Re-fit from saved raw data (`fit_raw.py`)

If you saved raw events with `--save-raw`, you can re-fit with a different
observable selection or transform pipeline without re-simulating:

```bash
python src/run_regression/fit_raw.py \
    simulated/svj/svj_scan_raw.npz \
    --scan-npz simulated/svj/svj_scan.npz \
    --obs "leadVisPt,MET,tau1,tau2" \
    --out-npz simulated/svj/svj_scan_refit.npz \
    --n-workers 8
```

**All `fit_raw.py` arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `raw_npz` (positional) | — | Path to `*_raw.npz` from `--save-raw` |
| `--scan-npz` | sibling `svj_scan.npz` | Scan NPZ to read grid axes from |
| `--obs` | from scan NPZ `obs_names` | Comma-separated observable names |
| `--out-npz` | `svj_scan_refit.npz` | Output NPZ path |
| `--mvt-iters` | 200 | Max EM iterations for MVT fitting |
| `--n-workers` | 8 | Parallel worker processes |

---

## 2 · Adding transforms, distributions, and observables

Everything lives in `src/observables.py`.

### 2.1 Adding a new invertible transform

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
  *before* this transform is applied.  Return `None` if there is no restriction.

### 2.2 Adding a new fitting distribution

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
forwarded to `dist.fit(x, *args, **kwargs)`.  Use `f<n>=` kwargs to fix
parameters during MLE (e.g. `floc=0` to fix location to zero).

### 2.3 Adding a new observable

Add an entry to the `OBSERVABLES` dict:

```python
OBSERVABLES['myObs'] = {
    'col':             18,         # column index in the 22-col TSV (None if derived)
    'pipeline':        [('boxcox', {})],   # ordered list of (transform_name, fixed_params)
    'distribution':    'gennorm',  # key in DISTRIBUTIONS
    'default_include': False,      # include in DEFAULT_SCAN?
    'label':           r'My observable',   # LaTeX label for plots
    'desc':            'One-line physics description',
}
```

For a **derived** observable (computed as a ratio of two base observables):

```python
OBSERVABLES['obs_a/obs_b'] = {
    'col':             None,
    'pipeline':        None,
    'distribution':    None,
    'default_include': False,
    'label':           r'$a/b$',
    'desc':            'Derived: obs_a / obs_b.  GUI display only.',
    'derive_cols':     (col_a, col_b),   # TSV column indices (numerator, denominator)
}
```

To include a new observable in the default regression scan, set
`'default_include': True`, or add its name to `DEFAULT_SCAN` at the bottom of
`observables.py`.  Then re-run `scan_svj.py`.

### 2.4 Assigning a custom transform pipeline to an observable

The `pipeline` is an ordered list of `(transform_name, fixed_params_dict)` pairs.
Transforms are applied left-to-right; the fitting distribution receives the output
of the last transform.  Example for a signed-angle observable:

```python
'pipeline': [('abs_value', {}), ('boxcox', {})],
```

Example with a fixed-parameter flip (for thrust-like variables):

```python
'pipeline': [('affine_flip', {'a': 1.0}), ('boxcox', {})],
```

The parameter `a=1.0` is the fixed upper bound of `affine_flip`; changing `a`
requires editing the observable entry in `observables.py`.

---

## 3 · Diagnostics

`src/diagnostics.py` produces 3-panel figures for each selected observable:
- **Left**: raw distribution histogram
- **Middle**: after all invertible transforms, with the fitted distribution overlaid
- **Right**: final standard-normal mapped output, with N(0,1) overlaid

A summary line reports how many events passed the range checks.

### 3.1 Basic call

```python
from src.diagnostics import plot_observable_transforms

figs = plot_observable_transforms('data/regression/jets_default.tsv')
```

This uses `DEFAULT_SCAN` and shows all 12 default observables.

### 3.2 All arguments

```python
figs = plot_observable_transforms(
    tsv_path,           # str or Path — required; 22-column TSV
    obs='default',      # 'default' | list of names | comma-separated string
    n_events=None,      # int — subsample to first N valid events (default: all)
    figsize_per_obs=(12, 3),   # (width, height) per observable figure
    bins=60,            # histogram bin count
)
```

| Argument | Type | Description |
|----------|------|-------------|
| `tsv_path` | str/Path | Path to the 22-column TSV written by `svj_regression` |
| `obs` | str/list | `'default'` = DEFAULT_SCAN; `'leadVisPt,MET'` or `['leadVisPt','MET']` |
| `n_events` | int/None | Subsample after filtering (useful for quick checks) |
| `figsize_per_obs` | tuple | Figure dimensions per observable row |
| `bins` | int | Histogram bin count |

Returns a list of `matplotlib.Figure` objects (one per observable).

### 3.3 Show immediately

```python
from src.diagnostics import show_observable_transforms
show_observable_transforms('data/regression/jets_default.tsv', obs='leadVisPt,MET,tau1')
```

### 3.4 Workflow: generate a test TSV first

The diagnostics require a TSV file (raw PYTHIA output).  Generate one with:

```bash
# From the project root:
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
# writes data/regression/jets_default.tsv
```

All physics parameters for this run are in `src/generate_events/svj_regression.cfg`:

| Key | Default | Description |
|-----|---------|-------------|
| `mZ` | 1000.0 | Z′ mass (GeV) |
| `mq` | 4.0 | Dark quark mass (GeV) |
| `mPi` | 7.742 | Dark pion mass (GeV) — update manually if mRho changes |
| `mRho` | 15.0 | Dark rho mass (GeV) |
| `rinv` | 0.45 | Dark pion invisible BR |
| `rinv2` | 0.70 | Dark rho invisible BR |
| `Brl` | 0.3 | Fraction of visible dark-rho decays → μ⁺μ⁻ |
| `alphaD` | 0.1 | Dark coupling α_D |
| `nEvent` | 50000 | Number of events |
| `jetR` | 1.0 | Anti-kT jet radius |
| `LambdaDQCD` | 4.839 | Dark QCD scale (GeV) — update if mRho changes |
| `nWorkers` | 14 | Parallel C++ threads |
| `save_tsv` | 1 | Write `jets_default.tsv` (set 0 to skip) |
| `vis_jet_pt_min` | 100.0 | Minimum visible jet pT threshold (GeV) |
| `full_obs` | 1 | Compute all 22 observables (0 = only leadVisPt/leadWidth/MET; for debugging) |

> `mPi` and `LambdaDQCD` in this file are hardcoded floats.  If you change
> `mRho`, update them manually: `mPi = 8/15.5 * mRho`, `LambdaDQCD = 5/15.5 * mRho`, if you want to keep the ratio the same.

---

## 4 · Interactive GUI (Jupyter)

### 4.1 Quick start

Open `gui.ipynb` (or a new notebook) from the project root and run:

```python
%matplotlib widget
import sys
sys.path.insert(0, 'src/gui')
from svj_explorer import show
show()
```

The GUI loads `simulated/svj/svj_scan.npz` automatically (or falls back to the
old-format `simulated/gennorm/gennorm_scan.npz` if the new file is not present).

### 4.2 GUI controls

| Control | Description |
|---------|-------------|
| **mZ′ slider** | Z′ mass in GeV (grid range) |
| **mRho slider** | Dark rho mass in GeV (grid range) |
| **rinv slider** | Dark pion invisible BR (grid range) |
| **alphaD slider** | Dark coupling α_D (grid range) |
| **Feature X / Y dropdowns** | Choose any two observables for the joint plot |
| **Fixed / Auto toggle** | Fixed axes use percentile-1/99 ranges from grid corners; Auto rescales to the current sample |
| **N model** | Number of model samples drawn from the interpolated distribution |
| **N validate** | Events for the VALIDATE PYTHIA run |
| **VALIDATE button** | Runs `svj_regression` at the current slider point and overlays the true distribution in crimson |
| **Cuts panel** | Per-observable range sliders to filter both model and true events; **Reset cuts** restores all |

Moving any physics slider clears the validation overlay (since the true data is now at a different point).

### 4.3 `show()` arguments

```python
show(n_samples=10_000)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | 10 000 | Initial number of model samples per draw |

### 4.4 Observable coverage

The GUI dropdown includes all observables that are in the loaded NPZ plus any
derived observables (ratios) whose components are present.  The set is determined
automatically when the module is imported.

---

## 5 · Parameter interpolation API (Python)

Once a scan NPZ is present, `src/helpers.py` provides:

```python
import sys; sys.path.insert(0, 'src')
import helpers

# Get grid bounds
mZ_vals, mRho_vals, rinv_vals, alphaD_vals = helpers.gn_grid_bounds()

# Interpolate fitted parameters at any (mZ, mRho, rinv, alphaD) inside the grid
result = helpers.interpolate_gennorm_params(mZ=1500, mRho=20, rinv=0.3, alphaD=0.4)

# Sample — format-agnostic (handles both old and new NPZ)
if len(result) == 5:
    # New MVT format: (R_upper, nu, obs_params_flat, param_offsets, obs_names)
    X = helpers.sample_svj_new(*result, n_samples=50_000)
else:
    # Old gennorm format: (corr_66, tf_12x4)
    X = helpers.sample_svj(*result, n_samples=50_000)
# X has shape (n_samples, n_obs) in original physical units
```

---

## 6 · Old version

The `old_version/` directory is a self-contained snapshot of the v1 pipeline
(12 fixed observables, Box-Cox + gennorm marginals, Gaussian copula).

```python
# From the project root:
%matplotlib widget
import sys
sys.path.insert(0, 'old_version')
from svj_explorer import show
show()
```

This loads `old_version/simulated/gennorm/gennorm_scan.npz` and uses
`old_version/helpers.py`.  It cannot be re-generated from the old code
(the binary now uses the new BR parameterisation), but all GUI features work
from the saved NPZ.

---

## 7 · NPZ file formats

### New format (`simulated/svj/svj_scan.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `param_flat` | `(N_mZ, N_mRho, N_rinv, N_alphaD, total_params)` | All fitted parameters, flat |
| `param_offsets` | `(n_obs+1,)` | `param_flat[..., offsets[i]:offsets[i+1]]` = params for obs i |
| `corr_start` | scalar int | `param_flat[..., corr_start:corr_start+n_corr]` = MVT correlation upper-triangle |
| `nu_idx` | scalar int | `param_flat[..., nu_idx]` = MVT degrees of freedom ν |
| `obs_names` | `(n_obs,)` | Observable names in regression order |
| `scan_params` | `(N_mZ, N_mRho, N_rinv, N_alphaD, 6)` | `[mZ, mRho, mPi, LambdaQCD, rinv, alphaD]` per point |
| `mZ_vals`, `mRho_vals`, `rinv_vals`, `alphaD_vals` | 1-D | Grid axis values |

`total_params = param_offsets[-1] + n_corr + 1` where `n_corr = n_obs*(n_obs-1)//2`.

### Old format (`simulated/gennorm/gennorm_scan.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `corr_params` | `(grid..., 66)` | MVN correlation upper-triangle (12 observables) |
| `transform_params` | `(grid..., 12, 4)` | `[lam, beta, loc, scale]` per observable |
| `obs_names` | `(12,)` | Observable names |
| `scan_params` | `(grid..., 6)` | Same as above |
| `mZ_vals`, ... | 1-D | Grid axis values |

---

*Environment setup (PYTHIA8, FastJet, Python packages) will be documented separately.*

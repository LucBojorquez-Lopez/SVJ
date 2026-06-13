# mySVJ — SVJ Interpolation Framework

Semi-Visible Jet (SVJ) parameter-space exploration using PYTHIA8 + FastJet.
The framework scans an arbitrary subset of SVJ physics parameters (configured via
an INI-style config file), fits a per-observable transform pipeline + Multivariate-t
copula at each grid point, and provides fast interpolation so that the joint
observable distribution can be sampled at any parameter point without re-running PYTHIA.

---

## 0 · Setup

### 0.1 Prerequisites

| Requirement | Minimum version | Notes |
|-------------|-----------------|-------|
| Linux x86-64 | — | Binary is ELF 64-bit; macOS not tested |
| g++ | 9 | C++17 `<filesystem>` required |
| Python | 3.8 | |
| PYTHIA | 8.317 | built in-place; see §0.3 |
| FastJet | 3.5.1 | installed with `--prefix`; see §0.4 |

### 0.2 Required directory layout

The Makefile hardcodes `PYTHIA_DIR = ../pythia8317` and `FASTJET_DIR = ../fastjet3`,
so both libraries must live **as siblings** of this repository:

```
<parent>/
├── pythia8317/    # PYTHIA 8.317 built in-place (§0.3)
├── fastjet3/      # FastJet ≥ 3 installed with --prefix (§0.4)
└── mySVJ/         # this repo  ← you are here
```

### 0.3 Install PYTHIA 8.317

```bash
# Run from <parent>/ (one level above mySVJ/)
wget https://pythia.org/download/pythia83/pythia8317.tgz
tar xzf pythia8317.tgz        # extracts to pythia8317/
cd pythia8317
./configure
make -j$(nproc)
cd ..
```

This builds PYTHIA in-place; no `make install` step is needed.
The shared library ends up at `pythia8317/lib/libpythia8.so`.

### 0.4 Install FastJet 3.5.1

```bash
# Run from <parent>/
wget http://fastjet.fr/repo/fastjet-3.5.1.tar.gz
tar xzf fastjet-3.5.1.tar.gz
cd fastjet-3.5.1
./configure --prefix="$(pwd)/../fastjet3"
make -j$(nproc) install
cd ..
```

This installs the library to `fastjet3/` and the config binary to
`fastjet3/bin/fastjet-config`, which the Makefile queries for compiler/linker flags.

### 0.5 Install Python packages

```bash
pip install numpy "scipy>=1.6" matplotlib ipywidgets ipympl jupyterlab
```

`scipy ≥ 1.6` is required for `scipy.stats.multivariate_t`.
`ipympl` enables the `%matplotlib widget` backend used by the Jupyter GUI.

### 0.6 Create the Makefile

The `Makefile` is **not tracked in git** (it is listed in `.gitignore`) because it
contains paths that are local to each machine.  Create it at the repository root:

```bash
# From mySVJ/ root:
cat > Makefile << 'EOF'
PYTHIA_DIR  = ../pythia8317
FASTJET_DIR = ../fastjet3
-include $(PYTHIA_DIR)/examples/Makefile.inc

CXX_COMMON += -std=c++17 -pthread

CXX_COMMON := $(OBJ_COMMON) -I$(PYTHIA_DIR)/include $(CXX_COMMON) $(GZIP_LIB)
CXX_COMMON += -L$(PYTHIA_DIR)/lib -Wl,-rpath,$(PYTHIA_DIR)/lib -lpythia8 -ldl
CXX_COMMON += $(shell $(FASTJET_DIR)/bin/fastjet-config --cxxflags --libs)
CXX_COMMON += -lstdc++fs

src/generate_events/svj_regression: src/generate_events/svj_regression.cc
	$(CXX) $< -o $@ $(CXX_COMMON)

.PHONY: svj_regression
svj_regression: src/generate_events/svj_regression
EOF
```

Adjust `PYTHIA_DIR` and `FASTJET_DIR` if your installations live elsewhere.
The indented recipe line above uses a **tab** character — `cat << 'EOF'` preserves it. Running the previous command from the terminal might lose this tab indentation for some; if so, vim into the Makefile and add that tab yourself.

### 0.7 Build the event generator

```bash
# From the mySVJ/ root:
make svj_regression
```

This compiles `src/generate_events/svj_regression.cc` against PYTHIA8 and FastJet
and writes the binary to `src/generate_events/svj_regression`.

> **Note**: A pre-compiled binary is tracked in git for convenience, but it has
> absolute library paths baked in from the original build machine.  Always
> re-run `make svj_regression` after a fresh clone before trying to run or scan.

### 0.8 Verify the build

```bash
# From the mySVJ/ root (the binary resolves ../pythia8317 relative to here):
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
```

A successful run writes `simulated/tsv/jets_default.tsv` (~50 000 events).
The output directory is created automatically.

---

## Directory layout

```
mySVJ/
├── Makefile                       Build target: make svj_regression  (not in git; see §0.6)
├── run_svj_scan.sh                SLURM array job: parameter scan → svj_scan.npz
├── run_svj_tsv.sh                 SLURM array job: batch TSV generation → jets_default.tsv
├── merge_svj_tsv.sh               Merge per-job TSV shards after run_svj_tsv.sh
├── src/
│   ├── generate_events/
│   │   ├── svj_regression.cc      C++ event generator (PYTHIA8 + FastJet)
│   │   └── svj_regression.cfg     Default physics + run parameters
│   ├── run_regression/
│   │   ├── scan_svj.py            Main scan script (grid → svj_scan.npz)
│   │   ├── scan_regression.cfg    Config for scan_svj.py (grid axes, physics params)
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
└── simulated/
    ├── svj/svj_scan.npz           Output of scan_svj.py  (new format; created on first scan run)
    ├── tsv/jets_default.tsv       TSV output of single binary run (created on first run)
    └── v1/regression_scan.npz    V1 archived data (gennorm + MVN copula, 12 obs)
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

The config uses **INI-style sections**.  Every physics parameter must appear in
exactly one of three zones:

```ini
[scan]
# axes of the N-D grid; format: name = min, max, n[, spacing]
# spacing = linear (default) or log
mZ        = 500, 4000, 8
mRho      = 10, 30, 8
rinv_pion = 0.05, 0.70, 8
alphaD    = 0.10, 0.80, 8

[fixed]
# constant at every grid point
mq   = 4.0
Brmu = 0.3
jetR = 1.0

[derived]
# arithmetic expressions evaluated from scan+fixed values at each point
# Only +, -, *, / and parentheses are allowed (no imports or builtins)
mPi        = mRho * (8.0 / 15.5)
LambdaDQCD = mRho * (5.0 / 15.5)
rinv_rho   = rinv_pion              # tie dark-rho BR to dark-pion BR

[simulation]
nEvent           = 50000   # events per grid point
nWorkers         = 1       # C++ threads per point (keep 1 with outer parallelism)
n_outer_workers  = 48      # simultaneous grid points (ProcessPoolExecutor)
checkpoint_every = 200     # save NPZ every N completions
output_dir       = simulated
```

**To change which parameters are scanned** — move them between sections:
- Move a param from `[fixed]` or `[derived]` into `[scan]` to add a scan axis.
- Move a param from `[scan]` into `[fixed]` to hold it constant.
- Move a param from `[fixed]` into `[derived]` to tie it to another parameter
  (e.g. `rinv_rho = rinv_pion` makes them always equal without scanning both. NOTE: Derived parameters can only be written in terms of scanning parameters, not fixed ones).
- Any number of scan axes is supported (1-D, 2-D, 5-D, …).

> **Note on `rinv_pion` / `rinv_rho`**: The C++ binary uses separate keys
> `rinv_pion` (dark pion invisible BR) and `rinv_rho` (dark rho invisible BR).
> In the default scan these are tied via `rinv_rho = rinv_pion` in `[derived]`;
> move `rinv_rho` to `[scan]` to scan them independently.  The BR decomposition
> is `rho_mu_br = Brmu*(1−rinv_rho)`, `rho_bott_br = (1−Brmu)*(1−rinv_rho)`,
> `inv_br = rinv_rho`; together they sum to 1.

**Log-spacing**: append `, log` to a `[scan]` entry to use `np.logspace` instead
of `np.linspace`:
```ini
mZ = 500, 4000, 8, log
```

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

A ready-to-use SLURM array script is at the project root:

```bash
sbatch run_svj_scan.sh          # submits a 4-task array (tasks 0–3)

# After all tasks finish, merge their partial NPZs:
python src/run_regression/scan_svj.py --merge --n-jobs 4
```

Edit `run_svj_scan.sh` to change the array size, partition, or resource requests.
To split manually:

```bash
# In any SLURM script body:
python src/run_regression/scan_svj.py scan_regression.cfg \
    --job-index $SLURM_ARRAY_TASK_ID --n-jobs $N_JOBS

# After all jobs finish, merge:
python src/run_regression/scan_svj.py --merge --n-jobs $N_JOBS
```

### 1.7 Batch TSV generation (`run_svj_tsv.sh`)

To generate a large TSV with more events than a single node can produce in one
run, use the batch TSV script.  Each SLURM array task runs the full binary
with the same physics parameters but a unique `seed_offset`, so the events are
statistically independent.  The total event count is `N_JOBS × nEvent`
(where `nEvent` comes from `svj_regression.cfg`).

```bash
# 1. Submit the array (default: 4 tasks × nEvent events each):
sbatch run_svj_tsv.sh

# 2. After all tasks finish, merge the shards:
bash merge_svj_tsv.sh 4          # 4 = N_JOBS in run_svj_tsv.sh
```

The merge script concatenates the `#`-header from shard 0 and all data rows
into `simulated/tsv/jets_default.tsv` (and `jets_kinematics.tsv`), then
removes the per-job shard files.  Pass `--keep-shards` to retain them.

```bash
bash merge_svj_tsv.sh 4 --keep-shards
```

Or submit the merge as a dependent job at `sbatch` time:

```bash
ARRAY_JID=$(sbatch --parsable run_svj_tsv.sh)
sbatch --dependency=afterok:${ARRAY_JID} --wrap "bash merge_svj_tsv.sh 4"
```

**How seed isolation works:** the binary computes each worker's PYTHIA seed as

```
seed = workerID + 1 + seed_offset × nWorkers
```

With `nWorkers = 14` and `N_JOBS = 4`:

| Task | seed_offset | PYTHIA seeds used |
|------|-------------|-------------------|
| 0 | 0 | 1 – 14 |
| 1 | 1 | 15 – 28 |
| 2 | 2 | 29 – 42 |
| 3 | 3 | 43 – 56 |

No overlap, no duplicated events.  The parameter scan (`run_svj_scan.sh`) is
unaffected — it never sets `seed_offset`, so it defaults to 0 and each
scan-point simulation uses seeds 1..1 (one inner worker), which is correct and
reproducible.

### 1.8 Re-fit from saved raw data (`fit_raw.py`)

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
%load_ext autoreload 
%autoreload 2 # so that changes in observables.py are reflected upon rerun, without requiring a kernel restart.

from src.diagnostics import plot_observable_transforms

figs = plot_observable_transforms('simulated/tsv/jets_default.tsv')
```

This uses `DEFAULT_SCAN` and shows all 12 default observables.

### 3.2 All arguments

```python
figs = plot_observable_transforms(
    tsv_path,           # str or Path — required
    obs='default',      # 'default' | list of names | comma-separated string
    n_events=None,      # int — subsample to first N valid events (default: all)
    figsize_per_obs=(12, 3),   # (width, height) per observable figure
    bins=60,            # histogram bin count
)
```

| Argument | Type | Description |
|----------|------|-------------|
| `tsv_path` | str/Path | Path to the TSV written by `svj_regression` |
| `obs` | str/list | `'default'` = DEFAULT_SCAN; `'leadVisPt,MET'` or `['leadVisPt','MET']` |
| `n_events` | int/None | Subsample after filtering (useful for quick checks) |
| `figsize_per_obs` | tuple | Figure dimensions per observable row |
| `bins` | int | Histogram bin count |

Returns a list of `matplotlib.Figure` objects (one per observable).

### 3.3 Show immediately

```python
from src.diagnostics import show_observable_transforms
show_observable_transforms('simulated/tsv/jets_default.tsv', obs='leadVisPt,MET,tau1')
```

### 3.4 Workflow: generate a test TSV first

The diagnostics require a TSV file (raw PYTHIA output).  Generate one with:

```bash
# From the project root:
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
# writes simulated/tsv/jets_default.tsv
```

All physics parameters for this run are in `src/generate_events/svj_regression.cfg`:

| Key | Default | Description |
|-----|---------|-------------|
| `mZ` | 1000.0 | Z′ mass (GeV) |
| `mq` | 4.0 | Dark quark mass (GeV) |
| `mPi` | 7.742 | Dark pion mass (GeV) |
| `mRho` | 15.0 | Dark rho mass (GeV) |
| `rinv_pion` | 0.45 | Dark pion invisible BR |
| `rinv_rho` | 0.70 | Dark rho invisible BR |
| `Brmu` | 0.3 | Fraction of VISIBLE dark-rho decays that go to → μ⁺μ⁻ |
| `alphaD` | 0.1 | Dark coupling α_D |
| `nEvent` | 50000 | Number of events |
| `jetR` | 1.0 | Anti-kT jet radius |
| `LambdaDQCD` | 4.839 | Dark QCD scale (GeV) |
| `nWorkers` | 14 | Parallel C++ threads |
| `save_tsv` | 1 | Write `jets_default.tsv` (set 0 to skip) |
| `vis_jet_pt_min` | 100.0 | Minimum visible jet pT threshold (GeV) |

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

The GUI loads `simulated/svj/svj_scan.npz` automatically.

### 4.2 GUI controls

| Control | Description |
|---------|-------------|
| **Parameter sliders** | One slider per scan axis, built dynamically from the loaded NPZ — adding or removing axes in `scan_regression.cfg` automatically updates the GUI |
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

# Get grid bounds: dict {axis_name: vals_array} for each scan axis
grid_bounds = helpers.svj_grid_bounds()
# e.g. {'mZ': array([500,...,4000]), 'mRho': array([10,...,30]), ...}

# Interpolate fitted parameters at any point inside the grid
# Pass a dict with a value for each scan axis
R_upper, nu, obs_params, param_offsets, obs_names = helpers.interpolate_svj_params(
    {'mZ': 1500, 'mRho': 20, 'rinv_pion': 0.3, 'alphaD': 0.4})

# Sample from the interpolated distribution
X = helpers.sample_svj_new(R_upper, nu, obs_params, param_offsets, obs_names,
                            n_samples=50_000)
# X has shape (n_samples, n_obs) in original physical units
```

The dict keys must match the scan axis names stored in the NPZ (i.e. whatever was
in `[scan]` when the scan was run).  Extra keys are ignored.

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

### V1 format (`simulated/v1/regression_scan.npz`)

12-observable Box-Cox + gennorm marginals, MVN copula.  Used by `old_version/` only.

| Key | Shape | Description |
|-----|-------|-------------|
| `corr_params` | `(grid..., 66)` | MVN correlation upper-triangle (12 observables) |
| `transform_params` | `(grid..., 12, 4)` | `[lam, beta, loc, scale]` per observable |
| `obs_names` | `(12,)` | Observable names |
| `scan_params` | `(grid..., 6)` | `[mZ, mRho, mPi, LambdaQCD, rinv, alphaD]` per point |
| `mZ_vals`, ... | 1-D | Grid axis values |

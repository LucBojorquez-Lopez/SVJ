# SVJ Interpolation Framework

Much of the existing work on Hidden Valley (HV) phenomenology constrains physics by picking a few event or jet observables, varying one or two HV parameters while holding others fixed at "reasonable" benchmark values. With an infinite-dimensional HV parameter space, constraints derived at benchmark points offer no guarantee of representativeness away from those benchmarks. Perhaps more importantly, nothing ensures that separate studies, conducted at different benchmarks, are mutually consistent.

This framework enables fast, interactive exploration of the full HV parameter space. The user has full control over which parameters to scan, which observables to define, how to transform and fit each observable's marginal distribution, and how to apply cuts. An interactive GUI exposes physics sliders so that the joint observable distribution can be explored dynamically, with cuts applied in real time, since observable cuts are central to these analyses.

We hope this flexibility in characterizing the HV parameter space at truth-level contributes to a more coherent and unified approach to Hidden Valley phenomenology.

For questions, suggestions, or requests: luc_bojorquezlopez@college.harvard.edu or mfarrington@g.harvard.edu.

---

## How it works

The pipeline has four stages.

**Simulation.** PYTHIA8 generates events over a user-defined grid in HV parameter space. Grid axes, ranges, and densities are fully configurable; a denser grid yields a more accurate interpolation. At each grid point, a configurable set of event and jet observables (e.g. MET, leading jet p_T, N-subjettiness) is computed and stored.

**Marginal fitting.** For each observable at each grid point, a user-defined sequence of invertible transforms is applied (e.g. an affine flip followed by Box-Cox, to symmetrize a bounded left-skewed distribution). Any free transform parameters are fitted to the data and stored. The transformed observable is then fitted to a user-chosen parametric distribution (e.g. generalized normal); its parameters are stored alongside the transform parameters. Finally, the probability integral transform maps each marginal to U[0,1].

**Copula.** The per-observable U[0,1] values are mapped through the probit to N(0,1). A Gaussian copula is fitted to the resulting vectors to capture inter-observable correlations; its correlation matrix R is estimated via Pearson's `corrcoef` and stored alongside the marginal parameters.

**Interpolation and sampling.** All stored parameters are interpolated jointly over the grid using a `LinearGridInterpolator`. At any new parameter point inside the grid, interpolated parameters are recovered in milliseconds. Sampling from the resulting joint distribution and inverting all transforms yields estimated observable samples in original physical units.

The GUI exposes this pipeline interactively: physics sliders trigger a re-interpolation and re-sample, updating the joint distribution plot in real time. Observable cuts filter the shared sample, so their effect on all other marginals is immediately visible.

> **Runtime note.** Full scans are compute-intensive and should be run on a cluster. As a reference, an 8×8×8×8 grid with 10 000 PYTHIA events per point took approximately 12 hours on 100 cores; fitting time is negligible relative to simulation. A practical strategy is to define all observables of interest upfront and use `--save-raw` (see [running-a-scan.md](docs/running-a-scan.md)), so that PYTHIA runs only once and observable fits can be re-tuned without re-simulation.

---

## Minimal example

A scan is shipped with the repository, so sampling works straight after
cloning — no simulation and no C++ build required:

```python
import numpy as np, sys; sys.path.insert(0, 'src')
import helpers

R_upper, obs_params, param_offsets, obs_names = helpers.interpolate_svj_params(
    {'mZ': 1500, 'rinv_pion': 0.3, 'mRho': 20.0, 'alphaD': 0.4})

X = helpers.sample_svj_new(R_upper, obs_params, param_offsets, obs_names,
                           n_samples=50_000)
X = X[np.isfinite(X).all(axis=1)]   # see docs/api.md — extreme-tail NaNs
# X: (~50000, 11) array in original physical units, columns ordered as obs_names
```

This loads `simulated/svj/working_example/` — a complete 4-axis
(`mZ`, `rinv_pion`, `mRho`, `alphaD`) × 11-observable scan. A larger 6-axis
scan is also included; see [docs/api.md](docs/api.md) to switch between them.

To generate your own events and scans, complete [setup](docs/setup.md), then:

```bash
# 1. Build the C++ event generator
make svj_regression

# 2. Run a single-point simulation (useful for diagnosing observable transforms)
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
# → simulated/tsv/jets_default.tsv

# 3. Run the parameter scan (edit scan_regression.cfg first to configure the grid)
python src/run_regression/scan_svj.py src/run_regression/scan_regression.cfg
# → simulated/svj/svj_scan.npz
```

Or launch the interactive GUI:

```python
# In a Jupyter notebook from the project root:
%matplotlib widget
import sys; sys.path.insert(0, 'src/gui')
from svj_explorer import show
show()
```

---

## Directory layout

```
SVJ/
├── Makefile                       make svj_regression | check-deps | clean
├── run_svj_scan.sh                SLURM array job: parameter scan → svj_scan.npz
├── run_svj_tsv.sh                 SLURM array job: batch TSV generation
├── run_svj_validation.sh          SLURM array job: production validation
├── merge_svj_tsv.sh               Merge per-job TSV shards after run_svj_tsv.sh
├── merge_svj_validation.py        Merge per-task NPZs after run_svj_validation.sh
├── condor/                        HTCondor equivalents for lxplus (see docs/lxplus.md)
├── src/
│   ├── generate_events/
│   │   ├── svj_regression.cc      C++ event generator (PYTHIA8 + FastJet)
│   │   └── svj_regression.cfg     Default physics + run parameters
│   ├── run_regression/
│   │   ├── scan_svj.py            Main scan script (grid → svj_scan.npz)
│   │   ├── scan_regression.cfg    Config for scan_svj.py (grid axes, physics params)
│   │   ├── fit_raw.py             Re-fit from saved raw events (no re-simulation)
│   │   ├── validate_fit.py        Marginal fit quality from a single TSV
│   │   ├── validate_grid.py       Interpolation quality vs PYTHIA truth
│   │   ├── validate_production.py Full benchmark incl. nearest-grid + MMD
│   │   └── _val_utils.py          Shared validation utilities (JS, MMD, sampling)
│   ├── gui/
│   │   └── svj_explorer.py        Jupyter notebook interactive explorer
│   ├── observables.py             Observable / transform / distribution registry
│   ├── helpers.py                 NPZ interpolation helpers + KLD utilities
│   └── diagnostics.py             Transform pipeline + validation plots
├── tests/                         pytest suite (211 tests, no PYTHIA needed)
├── docs/                          Extended documentation (see below)
└── simulated/
    ├── svj/working_example/       DEFAULT scan: 4 axes × 11 observables
    │   ├── svj_scan.npz             + svj_scan_meta.json
    │   └── validation_production.npz
    ├── svj/svj_scan.npz           Larger 6-axis × 16-observable scan
    ├── svj/validation_production.npz   2000-point validation of the above
    ├── tsv/                       Generated TSVs (gitignored; created on first run)
    └── v1/regression_scan.npz     Archived v1 grid; backs the KLD utilities only
```

---

## Documentation

Full reference documentation is in [`docs/`](docs/):

| Topic | File |
|-------|------|
| Installation & build | [docs/setup.md](docs/setup.md) |
| Running on CERN lxplus | [docs/lxplus.md](docs/lxplus.md) |
| Running and configuring scans | [docs/running-a-scan.md](docs/running-a-scan.md) |
| Adding observables, transforms, distributions | [docs/extending-observables.md](docs/extending-observables.md) |
| Diagnostic plots | [docs/diagnostics.md](docs/diagnostics.md) |
| Interactive GUI | [docs/gui.md](docs/gui.md) |
| Interpolation API | [docs/api.md](docs/api.md) |
| NPZ file formats | [docs/npz-format.md](docs/npz-format.md) |
| Validation | [docs/validation.md](docs/validation.md) |
| Running the unit tests | [docs/testing.md](docs/testing.md) |

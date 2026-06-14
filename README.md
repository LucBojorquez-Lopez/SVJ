# mySVJ — SVJ Interpolation Framework

<!-- TODO: introduce the database here -->

---

## How it works

<!-- TODO: describe the pipeline here -->

---

## Minimal example

After completing [setup](docs/setup.md), the full workflow is:

```bash
# 1. Build the C++ event generator
make svj_regression

# 2. Run a single-point simulation to get a TSV (useful for diagnostics)
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
# → simulated/tsv/jets_default.tsv

# 3. Run the parameter scan (edit scan_regression.cfg first to set grid axes)
python src/run_regression/scan_svj.py src/run_regression/scan_regression.cfg
# → simulated/svj/svj_scan.npz
```

Once the scan NPZ is ready, sample from the interpolated distribution at any point:

```python
import sys; sys.path.insert(0, 'src')
import helpers

R_upper, nu, obs_params, param_offsets, obs_names = helpers.interpolate_svj_params(
    {'mZ': 1500, 'mRho': 20, 'rinv_pion': 0.3, 'alphaD': 0.4})

X = helpers.sample_svj_new(R_upper, nu, obs_params, param_offsets, obs_names,
                            n_samples=50_000)
# X: (50000, n_obs) array in original physical units
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
mySVJ/
├── Makefile                       Build target: make svj_regression  (not in git; see docs/setup.md)
├── run_svj_scan.sh                SLURM array job: parameter scan → svj_scan.npz
├── run_svj_tsv.sh                 SLURM array job: batch TSV generation
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
├── docs/                          Extended documentation (see below)
├── old_version/                   Archived v1 (gennorm + MVN copula, self-contained)
└── simulated/
    ├── svj/svj_scan.npz           Output of scan_svj.py  (created on first scan run)
    ├── tsv/jets_default.tsv       TSV output of single binary run (created on first run)
    └── v1/regression_scan.npz    V1 archived data (gennorm + MVN copula, 12 obs)
```

---

## Documentation

| Topic | File |
|-------|------|
| Installation & build | [docs/setup.md](docs/setup.md) |
| Running and configuring scans | [docs/running-a-scan.md](docs/running-a-scan.md) |
| Adding observables, transforms, distributions | [docs/extending-observables.md](docs/extending-observables.md) |
| Diagnostic plots | [docs/diagnostics.md](docs/diagnostics.md) |
| Interactive GUI | [docs/gui.md](docs/gui.md) |
| Interpolation API | [docs/api.md](docs/api.md) |
| NPZ file formats | [docs/npz-format.md](docs/npz-format.md) |
| Old version (v1) | [docs/old-version.md](docs/old-version.md) |

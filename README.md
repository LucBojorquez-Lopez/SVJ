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

**Copula.** The per-observable U[0,1] values are mapped through the probit to N(0,1). A multivariate-t distribution is fitted to the resulting vectors to capture inter-observable correlations; its parameters (correlation matrix, degrees of freedom ν) are stored alongside the marginal parameters.

**Interpolation and sampling.** All stored parameters are interpolated jointly over the grid using a `LinearGridInterpolator`. At any new parameter point inside the grid, interpolated parameters are recovered in milliseconds. Sampling from the resulting joint distribution and inverting all transforms yields estimated observable samples in original physical units.

The GUI exposes this pipeline interactively: physics sliders trigger a re-interpolation and re-sample, updating the joint distribution plot in real time. Observable cuts filter the shared sample, so their effect on all other marginals is immediately visible.

> **Runtime note.** Full scans are compute-intensive and should be run on a cluster. As a reference, an 8×8×8×8 grid with 10 000 PYTHIA events per point took approximately 12 hours on 100 cores; fitting time is negligible relative to simulation. A practical strategy is to define all observables of interest upfront and use `--save-raw` (see [running-a-scan.md](docs/running-a-scan.md)), so that PYTHIA runs only once and observable fits can be re-tuned without re-simulation.

---

## Minimal example

After completing [setup](docs/setup.md), the full workflow is:

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

Full reference documentation is in [`docs/`](docs/):

| Topic | File |
|-------|------|
| Installation & build | [docs/setup.md](docs/setup.md) |
| Running and configuring scans | [docs/running-a-scan.md](docs/running-a-scan.md) |
| Adding observables, transforms, distributions | [docs/extending-observables.md](docs/extending-observables.md) |
| Diagnostic plots | [docs/diagnostics.md](docs/diagnostics.md) |
| Interactive GUI | [docs/gui.md](docs/gui.md) |
| Interpolation API | [docs/api.md](docs/api.md) |
| NPZ file formats | [docs/npz-format.md](docs/npz-format.md) |
| Validation | [docs/validation.md](docs/validation.md) |
| Old version (v1) | [docs/old-version.md](docs/old-version.md) |

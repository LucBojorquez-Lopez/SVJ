# Diagnostics

`src/diagnostics.py` produces 4-panel figures for each selected observable:
- **Panel 1**: raw distribution histogram
- **Panel 2**: after all invertible transforms, with the fitted distribution overlaid
- **Panel 3**: uniformized (PIT output) — should be U(0,1); includes a KS-test p-value
- **Panel 4**: final standard-normal mapped output, with N(0,1) overlaid and χ²/dof

A summary line reports how many events passed the range checks.

For observables that use a `point_mass` specification, the figure additionally
shows a dashed vertical line at the boundary value with the measured `p0` fraction
labelled, and the suptitle includes `p0=<value>`. The middle panel is computed on
continuous (non-PM) events only, so the transform is never applied to the boundary
value itself.

## Generate a test TSV first

The diagnostics require a TSV file (raw PYTHIA output). Generate one with:

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
| `vis_jet_pt_min` | 20.0 | Minimum visible jet pT threshold (GeV) |

> `mPi` and `LambdaDQCD` in this file are hardcoded floats. If you change
> `mRho`, update them manually: `mPi = 8/15.5 * mRho`, `LambdaDQCD = 5/15.5 * mRho`.

## Basic call

```python
%load_ext autoreload
%autoreload 2  # picks up changes in observables.py without kernel restart

from src.diagnostics import plot_observable_transforms

figs = plot_observable_transforms('simulated/tsv/jets_default.tsv')
```

This uses `DEFAULT_SCAN` and shows all 12 default observables.

## All arguments

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

## Show immediately

```python
from src.diagnostics import show_observable_transforms
show_observable_transforms('simulated/tsv/jets_default.tsv', obs='leadVisPt,MET,tau1')
```

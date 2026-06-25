# Running the unit tests

The test suite lives in `tests/` and covers the pure-Python analysis code
(observable registry, transforms, Gaussian copula fitting, interpolation helpers, and
validation utilities). No PYTHIA binary or pre-built NPZ is required.

## Dependencies

```bash
pip install numpy "scipy>=1.6" tqdm pytest
```

These are the same packages installed by GitHub CI (see `.github/workflows/ci.yml`).

## Run all tests

From the project root (`mySVJ/`):

```bash
pytest tests/ -v
```

`-v` prints each test name as it runs. Drop it for a shorter summary.

## Run a single test file

```bash
pytest tests/test_observables.py -v
pytest tests/test_transforms.py -v
pytest tests/test_helpers.py -v
pytest tests/test_scan_svj.py -v
pytest tests/test_val_utils.py -v
```

## Run a single test by name

```bash
pytest tests/test_observables.py -v -k "test_forward_inverse_roundtrip"
```

## What each file covers

| File | What it tests |
|------|---------------|
| `test_transforms.py` | `TRANSFORMS` registry: forward, inverse, `n_fitted`, `requires` |
| `test_observables.py` | `OBSERVABLES`/`DISTRIBUTIONS` registry integrity; `fit_observable_col`, `event_valid_mask`, forward/inverse roundtrip, `load_tsv` |
| `test_helpers.py` | `BoxCox`, `get_common_finite`, `preprocess_data`, `exp_terms`, `kld_params`, `sample_svj_new` |
| `test_scan_svj.py` | Config parsing (`read_scan_cfg`, `resolve_point`), Gaussian copula fitter (`fit_mvn_corr`) |
| `test_val_utils.py` | `resolve_point`, `sample_interior_points`, `js_per_obs`, `mmd_rbf`, `extract_obs_cols` |

## CI behaviour

GitHub Actions runs `pytest tests/ -v` automatically on every push to `main`
and on every pull request targeting `main`. The badge status reflects the most
recent run on `main`. To check CI results locally before pushing, just run the
command above — the test environment is identical (Python 3.11, same pip packages).

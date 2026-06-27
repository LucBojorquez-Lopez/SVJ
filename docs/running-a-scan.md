# Running a scan

## Configure the scan

All scan parameters live in `src/run_regression/scan_regression.cfg`.
Edit this file before running.

The config uses **INI-style sections**. Every physics parameter must appear in
exactly one of three zones:

```ini
[scan]
# axes of the N-D grid; format: name = min, max, n[, spacing]
# spacing = linear (default) or log
mZ            = 500, 4000, 8, log
rinv_pion     = 0.05, 0.70, 8
mPiOverLambda = 0.0, 2.0, 8        # dimensionless ratio mPi / LambdaDQCD
LambdaDQCD    = 2.0, 15.0, 8       # dark QCD confinement scale (GeV)
alphaD        = 0.10, 0.80, 8

[fixed]
# constant at every grid point
Brmu = 0.3
jetR = 1.0

[derived]
# arithmetic expressions evaluated from scan+fixed values at each point
# Operators +, -, *, /, ** and parentheses are supported (no imports or builtins)
mq       = LambdaDQCD * (mPiOverLambda / 5.5) ** 2
mPi      = mPiOverLambda * LambdaDQCD
mRho     = LambdaDQCD * (5.76 + 1.5 * mPiOverLambda ** 2) ** 0.5
rinv_rho = rinv_pion              # tie dark-rho BR to dark-pion BR

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
  (e.g. `rinv_rho = rinv_pion` makes them always equal without scanning both).
  Derived expressions may reference scan params, fixed params, or other derived params —
  all are available at evaluation time.
- Any number of scan axes is supported (1-D, 2-D, 5-D, …).

> **Note on `rinv_pion` / `rinv_rho`**: The C++ binary uses separate keys
> `rinv_pion` (dark pion invisible BR) and `rinv_rho` (dark rho invisible BR).
> In the default scan these are tied via `rinv_rho = rinv_pion` in `[derived]`;
> move `rinv_rho` to `[scan]` to scan them independently. The BR decomposition
> is `rho_mu_br = Brmu*(1−rinv_rho)`, `rho_bott_br = (1−Brmu)*(1−rinv_rho)`,
> `inv_br = rinv_rho`; together they sum to 1.

**Log-spacing**: append `, log` to a `[scan]` entry to use `np.logspace` instead
of `np.linspace`:
```ini
mZ = 500, 4000, 8, log
```

## Run the scan — standard (no raw saving)

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

**Resume after interruption**: just re-run the same command. Finished grid points
are detected via `np.isfinite` checks and skipped automatically.

## Run the scan — with raw event saving (`--save-raw`)

The `--save-raw` flag saves the pre-transform observable values for every grid point
alongside the fitted parameters. This allows re-fitting without re-running PYTHIA.

```bash
python src/run_regression/scan_svj.py src/run_regression/scan_regression.cfg \
    --save-raw
```

Additional output: `simulated/svj/svj_scan_raw.npz`
```
raw_flat       shape (total_events, n_obs)   — pre-transform values
raw_grid_flat  shape (total_events, K)        — grid indices per axis (K = number of scan axes)
```

> **Storage estimate**: with `nEvent=2000` and 12 observables over a 8⁴ grid,
> raw data is ~4096 × 2000 × 12 × 8 bytes ≈ **786 MB**.

## Select which observables to scan

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

## SLURM array jobs

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

## Batch TSV generation (`run_svj_tsv.sh`)

To generate a large TSV with more events than a single node can produce in one
run, use the batch TSV script. Each SLURM array task runs the full binary
with the same physics parameters but a unique `seed_offset`, so the events are
statistically independent. The total event count is `N_JOBS × nEvent`
(where `nEvent` comes from `svj_regression.cfg`).

```bash
# 1. Submit the array (default: 4 tasks × nEvent events each):
sbatch run_svj_tsv.sh

# 2. After all tasks finish, merge the shards:
bash merge_svj_tsv.sh 4          # 4 = N_JOBS in run_svj_tsv.sh
```

The merge script concatenates the `#`-header from shard 0 and all data rows
into `simulated/tsv/jets_default.tsv` (and `jets_kinematics.tsv`), then
removes the per-job shard files. Pass `--keep-shards` to retain them.

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

No overlap, no duplicated events. The parameter scan (`run_svj_scan.sh`) is
unaffected — it never sets `seed_offset`, so it defaults to 0 and each
scan-point simulation uses seeds 1..1 (one inner worker), which is correct and
reproducible.

## Re-fit from saved raw data (`fit_raw.py`)

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
| `--n-workers` | 8 | Parallel worker processes |

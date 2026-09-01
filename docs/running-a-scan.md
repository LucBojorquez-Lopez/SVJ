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
# From the project root (SVJ/):
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

> **Storage estimate**: with `nEvent=2000` and 12 observables over an 8⁴ grid,
> raw data is ~4096 × 2000 × 12 × 8 bytes ≈ **786 MB**. `*_raw.npz` is
> gitignored for this reason.

## Select which observables to scan

The default set is `DEFAULT_SCAN` in `src/observables.py` — every observable
whose entry has `default_include: True`. It currently holds 16:

```
leadVisPt, leadWidth, MET, maxMuPt, jetThrust, hemiMass1, e2c, e3c,
tau1, tau2, tau3, dPhiMETclose, HT, Meff, leadJetMass, nConst
```

28 base observables are available in total; see
[extending-observables.md](extending-observables.md).

Override from the command line with `--obs`:

```bash
python src/run_regression/scan_svj.py scan_regression.cfg \
    --obs "leadVisPt,MET,tau1,tau2,tau3"
```

The observable names must match keys in the `OBSERVABLES` dict in `src/observables.py`,
and each must have a non-`None` `pipeline` and `distribution`.

## Condor array jobs

Submit files live in [`condor/`](../condor/); the full lxplus walkthrough,
including the rules that apply because the repository sits on EOS, is in
[lxplus.md](lxplus.md) §7.

```bash
source setup_env.sh
condor_submit condor/smoke.sub      # one short job — always do this first
condor_submit condor/scan.sub       # the 16-slice array

# After all slices finish, merge their partial NPZs:
python src/run_regression/scan_svj.py --merge --n-jobs 16
```

The array size is set in **two** places inside `condor/scan.sub` that must
agree — the `arguments = "scan 16 $(ProcId)"` line and `queue 16` — and the
same number goes to `--merge`. Set `n_outer_workers` in the cfg to
`request_cpus`.

`--merge` resolves `output_dir` from the cfg you pass it, so if you redirected
`output_dir`, pass the same cfg to the merge:

```bash
python src/run_regression/scan_svj.py my_scan.cfg --merge --n-jobs 16
```

To split manually, in any batch script body:

```bash
python src/run_regression/scan_svj.py scan_regression.cfg \
    --job-index $TASK --n-jobs $N_JOBS      # Condor: pass $(ProcId) as $TASK
python src/run_regression/scan_svj.py scan_regression.cfg --merge --n-jobs $N_JOBS
```

## Batch TSV generation (`condor/tsv.sub`)

To generate a large TSV with more events than a single node can produce in one
run, use the TSV workflow. Each array task runs the full binary with the same
physics parameters but a unique `seed_offset`, so the events are statistically
independent. The total event count is `N_JOBS × nEvent` (where `nEvent` comes
from `svj_regression.cfg`).

```bash
# 1. Submit the array (default: 4 tasks × nEvent events each):
condor_submit condor/tsv.sub

# 2. After all tasks finish, merge the shards:
bash merge_svj_tsv.sh 4          # 4 = `queue N` in condor/tsv.sub
```

The merge script concatenates the `#`-header from shard 0 and all data rows
into `simulated/tsv/jets_default.tsv` (and `jets_kinematics.tsv`), then
removes the per-job shard files. Pass `--keep-shards` to retain them.

```bash
bash merge_svj_tsv.sh 4 --keep-shards
```

There is no DAGMan file, so the merge is a manual step after the array drains.

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

No overlap, no duplicated events. The parameter scan (`condor/scan.sub`) is
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

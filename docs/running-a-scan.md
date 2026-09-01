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
# binary            = svj_regression   # optional; see "Truth vs. Delphes" below
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

## Truth vs. Delphes: choosing the simulation engine

By default the scan runs the truth-level binary (`svj_regression`): jets are
clustered directly from stable Pythia8 final-state particles, with no
detector resolution or reconstruction efficiency. A separate, parallel
**Delphes-level stream** (`svj_regression_delphes`) is also available: it
runs the same Hidden-Valley physics through a particle-level-only Delphes
detector card (tracking/calorimeter smearing + efficiency, no jet-finder/
MET/b-tagging/pileup) before computing the identical observable set on the
reconstructed candidates. See `docs/setup_delphes.md` for the Delphes/ROOT
build steps and architecture notes.

Switch engines with one key in `[simulation]`:

```ini
[simulation]
binary = svj_regression_delphes   # omit entirely (or set to svj_regression)
                                    # for the default truth-level stream
```

`binary` is resolved relative to `src/generate_events/`. Switching away from
the default truth binary automatically renames the output NPZ to
`{binary}_scan.npz` (e.g. `svj_regression_delphes_scan.npz`) instead of
`svj_scan.npz`, so the two streams can never collide or silently overwrite
each other — you can keep both a truth scan and a Delphes scan side by side
in the same `output_dir` and compare them directly.

The scan's `svj_scan_meta.json` records which binary produced it, so
`validate_grid.py` / `validate_production.py` automatically validate against
the matching binary — no need to pass `--binary` yourself unless overriding
(see `docs/validation.md`).

Delphes-specific cfg keys (`delphes_card`, `jets_vis_only`, `dijet_only`,
`vis_jet_pt_min`, `seed_offset`) live in `svj_regression_delphes.cfg`'s own
defaults and are not part of `scan_regression.cfg`'s `[scan]`/`[fixed]`/
`[derived]` zones; the Delphes binary is single-threaded (`nWorkers` is
ignored), so all scan-level parallelism still comes from `n_outer_workers`.

## Select which observables to scan

The default set is `DEFAULT_SCAN` in `src/observables.py` — every observable
whose entry has `default_include: True`. It currently holds 16:

```
leadVisPt, leadWidth, MET, maxMuPt, jetThrust, hemiMass1, e2c, e3c,
tau1, tau2, tau3, dPhiMETclose, HT, Meff, leadJetMass, nConst
```

28 base observables are available in total; see
[extending-observables.md](extending-observables.md).

Both `svj_regression` and `svj_regression_delphes` compute the full
28-observable set, so any `DEFAULT_SCAN` or `--obs` selection works
identically with either binary.

Override from the command line with `--obs`:

```bash
python src/run_regression/scan_svj.py scan_regression.cfg \
    --obs "leadVisPt,MET,tau1,tau2,tau3"
```

The observable names must match keys in the `OBSERVABLES` dict in `src/observables.py`,
and each must have a non-`None` `pipeline` and `distribution`.

## How dense does the grid need to be?

Dense enough that interpolating between grid points means something. The
pipeline interpolates **fitted distribution parameters**, not samples, and the
transform parameters combine multiplicatively when the transforms are inverted
— so a linear blend of parameters is not the blend of the corresponding
distributions. Too coarse a grid does not degrade gracefully; it produces
confident nonsense between the points it was fitted at.

A 2x2x2x2x2x2 grid makes this concrete. Every grid point fits fine, but walking
`mZ` between two of them, with all other axes pinned at a grid point:

| mZ | `leadVisPt` median (GeV) |
|---|---|
| 1000 | 348  <- grid point |
| 1200 | 3867 |
| 1500 | 4711 |
| 2200 | 2457 |
| 3000 | 1019  <- grid point |

Both endpoints are physical; everything between is inflated by up to ~14x, in a
smooth arch that looks like a result rather than a failure. Nothing errors and
nothing is NaN.

So: two levels per axis is a **pipeline test, not a physics grid**. The shipped
`working_example` uses 8x8x4x4 for this reason. When adding an axis, check a
mid-grid point against a direct single-point simulation before trusting it —
`validate_grid.py` does exactly this comparison, see [validation.md](validation.md).

## Reproducibility across runs

Two runs of the same config agree physically but are **not bit-identical**, and
the same is true of a scan split across Condor jobs versus run in one process.
Sampled observable medians match to every digit you are likely to print;
fitted parameters differ by ~1e-14 relative in the median, with a tail to ~1e-5.

The cause is that grid axis values can differ in the last bit — `mZ` coming out
as `3000.000000000001` versus `3000.0000000000014` — which is written into each
per-point config as text, and the fit optimizers (Box-Cox `lambda` especially)
amplify it. It is floating-point noise, not non-determinism in the physics:
per-point PYTHIA seeds are fixed (`write_point_cfg` never sets `seed_offset`,
so every point runs with seed 1).

Worth knowing if you ever diff two NPZs and expect zeros. `fit_raw.py` re-fit
from the same raw events *is* exactly reproducible — zero difference.

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
independent. The total event count is `N_JOBS × nEvent`, where `nEvent` comes
from the cfg belonging to the binary in use (`svj_regression.cfg`, or
`svj_regression_delphes.cfg` when `SVJ_BINARY` selects the Delphes build).

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

**Delphes option.** Like the scan (see "Truth vs. Delphes" above),
`run_svj_tsv.sh` can generate a Delphes-level TSV instead of the truth-level
default: edit `BINARY_NAME="svj_regression_delphes"` near the top of the
script (also drop `-c 16` to `-c 1` in the `#SBATCH` header — the Delphes
binary is always single-threaded, so extra CPUs sit idle). The output
filename tag is derived from the binary name so the two streams never
collide: `svj_regression` → `jets_default(_N).tsv`, `svj_regression_delphes`
→ `jets_delphes(_N).tsv`. Merge with the matching `--binary` flag:

```bash
bash merge_svj_tsv.sh 4 --binary svj_regression_delphes
# -> simulated/tsv/jets_delphes.tsv
```

`svj_regression_delphes` never writes a kinematics TSV (unlike
`svj_regression`, it has no `tsv_kin_file` key at all); `run_svj_tsv.sh`
skips that override for it, and `merge_svj_tsv.sh` skips the kinematics
merge step accordingly — no separate flag needed.

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

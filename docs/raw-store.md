# The raw event store

Saving every observable PYTHIA produces, once, so that no future question about
this parameter space requires re-simulating it.

---

## Why

A scan's expensive product is not the fitted NPZ — that is a few MB of
distribution parameters. The expensive product is the **events**: ~900 CPU-hours
for the production grid, and they are thrown away the moment the fits are
written. Every later change — a different observable, a different transform, a
different distribution family, a symbolic-regression target that wants a
quantity nobody selected at scan time — costs that ~900 hours again.

EOS makes the alternative cheap. Tens of GB is nothing there, and re-fitting
from stored events is minutes rather than hours. So a production scan now writes
**every observable the generator computes**, for **both** the truth and Delphes
streams, and `fit_raw.py` re-fits any subset from disk.

The two streams are written separately and never merged, so any observable can
be compared truth-against-detector at the same grid point, from the same physics.

---

## What is stored

All 28 base observables — not the 16 that `DEFAULT_SCAN` fits. The set that is
*fitted* and the set that is *stored* are deliberately decoupled
(`scan_svj.py --raw-obs`, default `all`), for two reasons:

- Two observables, `closeJetIsLead` and `nInvClose`, have **no fit pipeline** at
  all: they are binary and small-integer respectively, so they can never appear
  in `--obs`. Their raw values are still worth keeping.
- An observable left out today cannot be recovered later without re-running the
  whole scan, which is the exact cost this store exists to avoid.

The registry's four ratio observables (`mass2/mass1`, `e3c/e2c`, `tau2/tau1`,
`tau3/tau2`) are *not* stored: they are exactly recomputable from the base 28,
so storing them would only cost disk.

Under Delphes, `nInvClose`, `fInv` and `closeJetIsLead` are identically zero by
construction — there is no truth-level invisible-particle information once
reconstruction has run. They are stored anyway, so the two streams stay
column-aligned and a diff is a plain subtraction.

---

## Format, and why not NPZ

Two files per scan job, written by `src/run_regression/raw_store.py`:

```
<stem>_raw.f32        little-endian float32, C-order, (n_rows, n_obs), headerless
<stem>_raw.idx.npz    grid_idx  (n_points, K) int16   grid indices per POINT
                      row_start (n_points,)   int64   first row in the .f32
                      row_count (n_points,)   int32   rows for that point
                      obs_names, axis_names, n_obs
```

The original `--save-raw` wrote one `*_raw.npz`. Three things about it do not
survive contact with a production grid:

**Memory.** It accumulated every point's events in a Python list and `np.vstack`ed
at the very end, so peak RSS was twice a job's entire raw output — ~3.6 GB for a
32-job split, against a `request_memory` of 2 GB. Jobs would have died partway
through the night, after burning the compute. The store now writes each point as
it completes and drops it: raw memory is one point, whatever the grid size.

**Redundancy.** It stored the K grid indices as int64 **per event**, via
`np.tile`, though they are constant within a point — 48 of every 176 bytes were
a copy of something already known. They now live in a per-point index.

**Precision.** Observables are parsed from a `%e`-formatted TSV and so carry
seven significant digits; float64 stored fourteen of them. float32 holds ~7.2,
which round-trips the generator's output exactly (relative error 2.3e-8).

The `.f32` is headerless on purpose. It is appended to point by point, and event
yield per point varies with acceptance, so the final row count is not knowable
when the file is opened — which is precisely what a `.npy` header would demand.
The index is rewritten at every checkpoint, so a job killed mid-flight leaves a
readable store covering the points it finished, and a truncated `.f32` is
harmless because the reader trusts the index.

---

## Storage budget

Measured, not estimated — 112 bytes per event for 28 observables:

| | per stream | both streams |
|---|---|---|
| production grid, 16384 pts x 20 000 events | 36.7 GB | **73 GB** |
| the old format would have been (16 obs, f64, per-event indices) | 57.7 GB | 115 GB |

Per Condor job at `queue 32`: ~1.1 GB of raw events, written straight to EOS as
one long sequential stream — the workload EOS is good at, unlike the many-small-
files case §3 of [lxplus.md](lxplus.md) warns about.

---

## Running it

```bash
source setup_env.sh

condor_submit condor/scan_raw_truth.sub      # 512 jobs, 32 points each
condor_submit condor/scan_raw_delphes.sub    # 512 jobs, 32 points each
```

**One core per job, deliberately.** The first attempt at this run used
`request_cpus = 16` and `queue 32`, and it did not schedule: 10.4 hours to the
first slot, and only 2 of 32 jobs ever started. See [lxplus.md](lxplus.md) §9.
Measured worker throughput is also ~2.2x slower than the login-node benchmark
in §5.1, so size jobs from ~450 s per point, not ~205 s: 32 points is ~4 h for
truth and ~4.8 h for Delphes, both inside `workday`.

Every job writes its own raw shard, and `RawReader` takes a list of stems, so a
many-job split costs nothing at read time:

```python
import glob, raw_store
r = raw_store.RawReader(sorted(glob.glob('.../truth/svj/*_raw.f32')))
```

Both write outside the repository, to `/eos/user/$USER/svj/prod/{truth,delphes}`.
That is deliberate: the default `output_dir` is `simulated`, and a production run
there would overwrite the committed `simulated/svj/svj_scan.npz`.

The grid is in `src/run_regression/scan_prod_truth.cfg`; the Delphes config is
the same file with `binary = svj_regression_delphes` appended.

| axis | range | points |
|---|---|---|
| `mZ` | 500 -> 4000 GeV, log | 8 |
| `rinv_pion` | 0.05 -> 0.70 | 8 |
| `mPiOverLambda` | 0.1 -> 2.0 | 4 |
| `LambdaDQCD` | 2.0 -> 15.0 GeV | 4 |
| `alphaD` | 0.1 -> 0.6 | 4 |
| `jetR` | 0.4 -> 1.2 | 4 |

`mPiOverLambda` starts at 0.1 rather than 0.0 because 0.0 gives a massless dark
pion *and* a massless dark quark — a degenerate corner that would have been 25%
of the grid. 0.1 was checked to run cleanly at all four extreme
`(LambdaDQCD, mZ)` corners, at 95-100% event acceptance.

---

## Re-fitting without simulating

```bash
python src/run_regression/fit_raw.py \
    /eos/user/l/lbojorqu/svj/prod/truth/svj/svj_scan_raw.f32 \
    --scan-npz /eos/user/l/lbojorqu/svj/prod/truth/svj/svj_scan.npz \
    --obs "leadVisPt,MET,tau2,tau3,e2c" \
    --out-npz my_refit.npz --n-workers 16
```

`fit_raw.py` selects the requested columns **by name** from the store. That
matters: before the store recorded its own `obs_names`, the worker assumed its
input columns were already in `--obs` order, so re-fitting a 5-observable subset
of a 28-observable store silently read the first five stored columns instead of
the five named. Legacy `*_raw.npz` files, which record no names, still take the
old positional path and say so.

Points are contiguous slices of a memmap, so a re-fit reads only what it needs
rather than materialising the store.

---

## Caveats worth knowing before relying on this

- **`jetR` is the one axis the store cannot save you from.** Every other
  parameter enters only through the events; `jetR` changes the jet clustering
  itself, so a new value means new simulation. It is a scan axis here for
  exactly that reason.
- **Rows are events that passed `event_valid_mask` for the *fitted*
  selection.** A later re-fit therefore sees a sample already filtered by the 16
  observables fitted at scan time, not an unfiltered one. For observables
  outside that 16 this is a mild, shared selection effect — not a bias between
  grid points, but worth stating before it is used for anything quantitative.
- **Fitting is not deterministic.** Re-fitting identical data twice moves some
  parameters by up to ~1e-2 absolute (a few percent relative, on shape
  parameters that are weakly constrained). This is a property of the fit, not of
  the store: it is the same magnitude as the scan-versus-refit difference, so
  float32 storage contributes nothing measurable on top of it.
- **`--merge` still ignores `output_dir`** (see [lxplus.md](lxplus.md) §8). The
  raw stores are per-job and are not merged — `RawReader` accepts a list of
  shard stems and presents them as one store — but the *fitted* shard NPZs from
  a redirected `output_dir` must be merged by hand.

---

## The production run, as it actually went  (submitted 2026-09-02, finished 2026-09-04)

Recorded because almost none of it went as the estimates above predicted, and
the failures are the reusable part.

### What landed

| | truth | Delphes |
|---|---|---|
| grid points | **16 272 / 16 384 (99.32%)** | **16 353 / 16 384 (99.81%)** |
| events | 317 650 375 | 307 578 801 |
| mean events/point | 19 521 | 18 809 |
| observables | 28 | 28 |
| duplicated points | 0 | 0 |
| on disk | 46 GB | 45 GB |

**16 242 points are present in both streams**, which is what a truth-against-
detector comparison needs. Delphes' lower mean event count is the expected
acceptance loss under reconstruction; it has 76 points below 10k events where
truth has none.

### The scheduling lesson, which cost a day

The first submission used `request_cpus = 16` and `queue 32`. It sat **idle for
10.4 hours** before one slot matched, and only 2 of 32 jobs ever started;
`condor_q -better-analyze` reported *"1 slots match and are willing to run your
job, 3226 slots would match if drained"*. The farm is built of partly-occupied
machines — roughly 110 000 single-core slots against approximately one free
16-core slot — so a 16-core request is effectively a whole-node reservation.

Resubmitting as `request_cpus = 1`, `n_outer_workers = 1`, `queue 512` took the
matching-slot count from **1 to 1090**, and the run completed in under two days.
Neither `+JobFlavour` nor job count was the lever; both were tested. See
[lxplus.md](lxplus.md) §9.

Measured per-job cost: **95-157 minutes** for 32 points, against the ~1.8 h
predicted from login-node throughput — worker cores run about 2.2x slower than
the benchmark node, so size jobs from ~450 s per point, not ~205 s.

### Two data-integrity findings

**`checkpoint_every` must be smaller than points-per-job.** It was left at 50
while restructuring to 32 points per job, so the raw index never flushed
mid-job — and a headerless `.f32` with no index is unreadable. For the first
71 jobs an eviction would have discarded everything computed. Fixed to 4 while
the remaining 93% were still queued. One shard (`svj_scan_331`) did die early
and lost its raw entirely, confirming the exposure was real.

**The index must be written after the data is durable.** During the run, 22
shards appeared truncated — the index describing up to a point more data than
another process could read, with the index file's mtime roughly three minutes
ahead of the `.f32`. That is EOS write-back: the small index lands immediately
while appended bytes are still in flight. Harmless for a job that finishes,
because `close()` settles both, but a job killed in that window leaves an index
promising rows that never arrived. `RawWriter.flush` now `fsync`s the data
before writing the index, and `RawReader` drops any incomplete tail with a
warning rather than silently handing out unwritten rows.

### Shard NPZs are wasteful, and merging is worth doing

Each of the 512 per-job scan NPZs carries `param_flat` at the **full** grid
shape — 16384 x 184 x 8 bytes = 24 MB — of which only its own 32 points are
non-NaN. That is ~12 GB of almost entirely NaN across both streams, and a merge
collapses it to 24 MB. Note the caveat in [lxplus.md](lxplus.md) §8: `--merge`
ignores `output_dir`, so a redirected run like this one must be merged by hand.

### Backfilling the gaps

The missing points are concentrated: truth's 112 belong to just **5** of the 512
job slices, Delphes' 31 to 31 slices. Because `scan_svj.py` skips grid points
already finite in its own shard NPZ, resubmitting only those slices redoes the
gaps and nothing else — `condor/gapfill_truth.sub` and
`condor/gapfill_delphes.sub` enumerate them with `queue SLICE from (...)`.

One wrinkle worth knowing: a shard that recorded points in its NPZ but lost its
raw index (331, which had 4 points in the NPZ and no index) would be *skipped*
by resume. Its NPZ has to be removed for the raw to regenerate.

### Incident: the backfill destroyed data  (2026-09-06)

Recorded in full because the mechanism is subtle and the same shape of bug will
recur in anything that resumes.

`RawWriter.__init__` opened the store with `'wb'`, which truncates. Separately,
`scan_svj.py` resumes by skipping grid points already finite in its shard NPZ.
Those two behaviours are incompatible, and the incompatibility is invisible
until a shard is re-run *partly* finished: the job then has only its missing
points left to do, opens the store fresh, and writes a shard containing only
those — silently discarding everything already stored.

The `n_todo == 0` early return protects a *fully* complete shard, which is why
the first production pass was unaffected. But a backfill targets exactly the
partly-finished slices, so every job in it hit the bug.

| | before backfill | after | net |
|---|---|---|---|
| truth | 16 272 | 16 308 | **+36** (of an expected +112) |
| Delphes | 16 353 | **15 405** | **-948** |

18 Delphes shards were left zero-byte and 13 more held a single point; 4 truth
shards were cut to 8-20 points of 32.

**Fix.** `RawWriter` now continues an existing store by default: it reads the
prior index, keeps only entries whose rows are actually present in the file,
truncates to the end of the last such entry (dropping any partial tail from a
killed job), and appends from there. It refuses outright if the stored
`obs_names` disagree with the current run rather than mixing two column sets.
Verified: writing 3 points, reopening, and appending 2 more yields 5 points with
the original three intact — previously it yielded 2.

**Recovery.** The 35 damaged slices had their NPZ *and* raw moved to
`prod/damaged_backup/` — moved, not deleted, since the fitted parameters in
those NPZs are perfectly valid and only unusable for resume — and resubmitted
via `condor/recover_{truth,delphes}.sub` to regenerate all 32 points each.

**The general lesson.** A resumable pipeline needs every one of its outputs to
be resumable, not just the one the resume logic checks. Here the NPZ was
resumable and the raw store was not, and nothing connected the two. Worth
checking before adding any further per-job artefact.

## The Delphes SimpleCalorimeter segfault (2026-09-07)

**Symptom.** After the recovery and two gap-fill passes, the truth stream
reached 16384/16384 points but Delphes stalled at 16373, with 11 points that
failed every resubmission. One background shard (`diboson_4`) was stuck the
same way.

**Diagnosis.** Reproduced locally at
`mZ=500, rinv_pion=0.05, mPiOverLambda=0.733, LambdaDQCD=6.33, alphaD=0.267,
jetR=0.4`:

```
#5  SimpleCalorimeter::Process() from libDelphes.so
#4  <signal handler called>
exit=139                                   (SIGSEGV)
```

A segfault **inside Delphes**, not in our code. It explains why only the
Delphes stream had holes — truth never invokes Delphes — and why `diboson_4`
carried an identical stack trace.

**Why every retry failed.** The crash is deterministic in the PYTHIA seed, and
`write_point_cfg()` wrote no `seed_offset`, so every attempt generated the
identical event sequence and died on the identical event, after ~6 s, before
writing a single row. The resume logic was working perfectly — it skipped the 31
completed points in the slice and re-crashed on the 32nd, forever.

**Fix.** `scan_svj.py` now retries a crashed point with a fresh seed
(`_SEED_RETRIES = 3`, stride 7919). Attempt 0 still writes no `seed_offset`, so
every point computed before this change remains bit-reproducible.

Verified by running the offending point at four seeds:

| seed_offset | exit | kept events |
|---|---|---|
| 0 | **139 (SIGSEGV)** | crashed at 6 s, 0 rows |
| 1 | 0 | 19665 / 20000 |
| 2 | 0 | 19680 / 20000 |
| 3 | 0 | 19687 / 20000 |

Three for three, at a steady ~98.3% acceptance. **The point is healthy** — this
was never a physics corner where the generator cannot produce valid events, just
one indigestible event blocking an otherwise normal point.

That also sizes the retry budget. One crash in four seeds puts the per-seed
crash probability near 25% *at this point*, so four attempts leave ~0.4%
residual risk. Since only 11 of 16384 Delphes points ever crashed (0.07%), that
elevated rate has to be specific to these points rather than general —
plausibly the very light dark quark (`mq = 0.113 GeV`) combined with
`rinv_pion = 0.05`, which together drive unusually high hadron multiplicity
into the calorimeter.

Only **crash-shaped** failures are retried — a non-zero exit or a missing TSV. A
point that runs to completion but yields too few valid events has a physics
problem, and a new seed would only hide it.

For the background, the shard index *is* the seed, so `diboson_4` was
resubmitted as shard 20; the loader globs `diboson_*.tsv` and normalises by the
shards that actually landed.

**Caveat worth stating.** Reseeding does not fix the Delphes bug, it steps
around it, and it does so by discarding one event configuration that Delphes
cannot digest. At roughly 1 event in 20000 (5e-5) the induced bias is
negligible, but it is a selection on detector-simulation behaviour rather than
on physics. If these crashes ever become common rather than rare, the right
response is to find the offending kinematics and fix or guard Delphes, not to
keep reseeding.

**Why 11 missing points mattered at all.** See docs/normalisation.md M7: in 6-D
each point is a corner of up to 64 cells, so 11 missing points (0.07%) still
cost 2.9% of the interpolable volume. Point completeness is not cell
completeness.

## Re-fitting the whole grid with fit_raw.py

Two bugs made this impossible before 2026-09-07; both are worth knowing about
because they were silent.

**1. The CLI took one shard.** A production scan is sharded one stem per Condor
slice (512 for this grid), but `fit_raw.py` accepted a single positional path,
so it could only ever re-fit 32 points. `RawReader` already accepted a list;
only the CLI did not. It now takes a glob:

```bash
python3 src/run_regression/fit_raw.py \
    /eos/user/l/lbojorqu/svj/prod/truth/svj/svj_scan_*_raw.idx.npz \
    --scan-npz /eos/user/l/lbojorqu/svj/prod/truth/svj/svj_scan.npz \
    --obs <comma-separated names> \
    --out-npz <dir>/svj_scan.npz \
    --n-workers 4
```

512 shards stitch in 2.8 s. `RawReader` checks they agree on `obs_names`.

**2. It materialised the entire grid before fitting.** The task list was a list
comprehension over `reader.iter_points()`:

```python
tasks = [(gidx, np.asarray(evts[:, cols], dtype=np.float64), obs_selection)
         for gidx, evts in reader.iter_points()]
```

For 16384 points x 20000 events x 17 observables x 8 bytes that is **44.6 GB** —
and `float64` doubles the stored float32. Observed at **33.7 GB RSS with zero
workers spawned** on a 57 GB shared login node, before a single line of output.
The whole point of `raw_store` is to stream (see its module docstring on
`request_memory`), and this defeated it.

Now lazy, with a bounded in-flight window of `2 x n_workers`: peak parent RSS a
few MB, and progress prints every 25 points instead of after the 44 GB build.

**A merge must come first**, because `fit_raw` reads axis metadata and the
complete `scan_params` from a merged scan NPZ:

```bash
python3 src/run_regression/scan_svj.py src/run_regression/scan_prod_truth.cfg \
    --merge --n-jobs 512 --keep-shards
```

**`--keep-shards` is mandatory, not cosmetic.** Without it `_merge` deletes the
per-shard NPZs, and those are what the scan's resume logic reads to skip
completed points — deleting them silently breaks every future gap-fill.

`fit_raw` also now writes `svj_scan_meta.json` beside its output, copying
`fixed_params`/`derived_exprs` from the input scan's meta and overwriting the
observable selection and parameter layout. Without it the refit is not loadable
by the GUI, which would silently fall back to its built-in defaults (`mq=4`,
`jetR=1`) — wrong for any scan that derives them, i.e. every production scan.

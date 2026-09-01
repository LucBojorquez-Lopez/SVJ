# Running on CERN lxplus

A complete path from `git clone` to submitted batch jobs on lxplus, and an
explicit account of **what ships with the repository versus what you have to
rebuild**.

> **What is verified, and what is not.** The environment, the build and the
> analysis pipeline were exercised on this account on lxplus (RHEL 9.8, LCG_110,
> gcc 13.1, Python 3.13): `make svj_regression` against the LCG view and the
> full §5 verification ladder.
>
> The Condor path in §7 was arrived at by submitting real jobs, not by reading
> documentation — the AFS/EOS split in §7.2 exists because the alternatives were
> tried and observed to fail. `condor/smoke.sub` runs end to end: worker node
> reads the EOS clone, sources `setup_env.sh`, runs the scan, writes its NPZ
> back to EOS, and its `.out`/`.err` are readable.
>
> Still unexercised: the array `.sub` files at full size — their job flavours
> and larger resource requests. They share every other setting with
> `smoke.sub`. Run §7.3 first regardless; it costs a minute.

---

## 1. What you do and do not need to rebuild

| | Ships in the clone | Must be rebuilt on lxplus |
|---|---|---|
| Scan NPZs (`svj_scan.npz`, `working_example/`) | ✅ committed | — |
| Validation NPZs | ✅ committed | — |
| Python analysis code, tests, docs | ✅ | — |
| `Makefile` | ✅ tracked | — |
| **PYTHIA 8.317** | ❌ | **no** — comes from CVMFS (§2) |
| **FastJet 3.5.1** | ❌ | **no** — comes from CVMFS (§2) |
| Python environment | ❌ | **no** — comes from CVMFS (§2) |
| **`svj_regression` binary** | ❌ gitignored | yes — one 4-second `make` (§4) |
| Raw TSVs (`simulated/tsv/*.tsv`) | ❌ gitignored | regenerate as needed |

The important consequence: **interpolation, sampling, the GUI and every
validation plot work the moment the clone finishes.** You need the C++
toolchain only to generate *new* events — a new scan, a new TSV, or the GUI's
VALIDATE button.

A compiled binary is deliberately not committed: it bakes absolute library
paths from the machine that built it, so a committed one would be useless
anywhere else.

---

## 2. Environment

Everything comes from one LCG view on CVMFS. Source `setup_env.sh` at the
repository root and you are done:

```bash
source setup_env.sh
```

That is the whole setup. Put it in your `.bashrc` if you like; **every** shell
and **every** batch job needs it.

### Why there is nothing to build

`LCG_110/x86_64-el9-gcc13-opt` ships **PYTHIA 8.317** and **FastJet 3.5.1** —
precisely the versions this project targets — along with gcc 13.1, Python 3.13,
numpy, scipy, matplotlib, tqdm, pytest, ipywidgets, ipympl and jupyterlab. So
there is no PYTHIA build, no FastJet build, and no venv.

This is not merely convenient. Source-building PYTHIA and FastJet needs several
GB, and on a default CERN account there is nowhere to put them: AFS home is a
2 GB quota, and `/afs/cern.ch/work/<u>/<user>` **does not exist** until you
request it through the CERN Resources portal. Earlier revisions of this guide
told you to build into `$SVJ_WORK` on AFS work; on an account without that
volume, that instruction cannot be followed. The CVMFS route sidesteps the
question entirely and costs no quota at all.

`setup_env.sh` sets, on top of the view:

| Variable | Why |
|---|---|
| `PYTHIA_DIR`, `FASTJET_DIR` | both point at the view — it is a single prefix providing `lib/libpythia8.so` and `bin/fastjet-config`, which is all the `Makefile` asks for |
| `GZIP_LIB` | an extra `-Wl,-rpath` to the view's gcc `lib64` (see §4) |
| `SVJ_REPO` | the repo root; the `.sub` files read it via `$ENV(SVJ_REPO)` |
| `X509_USER_PROXY` | `~/private/x509up` by default; the `.sub` files read it via `$ENV(...)` |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | an autoloaded pytest plugin in the view otherwise emits CDash `<DartMeasurement>` XML for every test, burying the summary under thousands of lines of markup |

To pin a different view, `export LCG_VIEW=...` before sourcing. List what is
actually available rather than trusting a tag in a document — they come and go
with lxplus releases:

```bash
ls /cvmfs/sft.cern.ch/lcg/views/          # pick an LCG_* release
ls /cvmfs/sft.cern.ch/lcg/views/LCG_110/  # pick a platform
```

Any view you choose must provide both `lib/libpythia8.so` and
`bin/fastjet-config`; `make check-deps` tells you in one second if it does not.

> **One trap worth knowing about.** The view's own `setup.sh` reads `$COMPILER`
> unguarded, so it aborts with `COMPILER: unbound variable` under `set -u`.
> `condor/svj_job.sh` runs `set -euo pipefail`, which means a naive `source`
> kills every batch job on that line. `setup_env.sh` shields the source and
> restores the caller's flags afterwards, so callers need no special handling.

### If the LCG view ever fails you

Building from source still works and the `Makefile` still supports it — you just
need somewhere with a few GB. Request an AFS work volume, or use an EOS
directory, then:

```bash
export SVJ_WORK=/afs/cern.ch/work/${USER:0:1}/${USER}/svj    # once granted
mkdir -p "$SVJ_WORK" && cd "$SVJ_WORK"

wget https://pythia.org/download/pythia83/pythia8317.tgz
tar xzf pythia8317.tgz && cd pythia8317 && ./configure && make -j4 && cd ..

wget http://fastjet.fr/repo/fastjet-3.5.1.tar.gz
tar xzf fastjet-3.5.1.tar.gz && cd fastjet-3.5.1
./configure --prefix="$SVJ_WORK/fastjet3" && make -j4 install && cd ..

export PYTHIA_DIR="$SVJ_WORK/pythia8317" FASTJET_DIR="$SVJ_WORK/fastjet3"
```

Use `-j4`, not `-j$(nproc)` — lxplus login nodes are shared and heavily
parallel builds get throttled or killed. Expect 15–30 minutes for PYTHIA. Do
this on AFS or local disk, **not** EOS: it is many thousands of small
compile-and-link writes, which is the one workload EOS is genuinely bad at.

---

## 3. Where the repository lives

EOS (`/eos/user/${USER:0:1}/${USER}/`) has far more room than AFS home, and if
your other repositories already live there it is the natural home for this one.
It works, and on a warm FUSE mount it is not even slow — `git status` on this
73 MB clone measures ~0.1 s.

If you do hit small-file slowness, `git config feature.manyFiles true` plus
`core.untrackedCache`, `core.preloadIndex` and `core.commitGraph` are the
settings that help most. Treat them as a remedy, not a prerequisite.

One consequence of an EOS-resident clone is not about git at all: Condor
cannot put its executable or its logs there, so a small amount has to sit on
AFS. That is handled for you — see §7.2.

---

## 4. Clone and build the generator

```bash
cd /eos/user/${USER:0:1}/${USER}          # or wherever you decided in §3
git clone https://github.com/LucBojorquez-Lopez/SVJ.git
cd SVJ

source setup_env.sh
make check-deps        # confirms PYTHIA + FastJet are findable; builds nothing
make svj_regression    # ~4 seconds
```

`PYTHIA_DIR` / `FASTJET_DIR` come from `setup_env.sh`, so plain
`make svj_regression` is all you ever type. Both are resolved to **absolute**
paths and written into the binary's rpath, so the executable runs from any
working directory.

> **Why `GZIP_LIB` is in `setup_env.sh`.** The binary links against the view's
> gcc-13 libraries, but the loader resolves `libstdc++` against EL9's system
> copy, which is gcc-11-era — the binary then dies with
> `GLIBCXX_3.4.31 not found`. `setup_env.sh` adds an rpath to the view's gcc
> `lib64` to fix it. `GZIP_LIB` is otherwise only set by PYTHIA's own
> `examples/Makefile.inc`, which a CVMFS view does not ship, so the `Makefile`
> appends it to the link line untouched and needs no edit.
>
> With that rpath the binary runs on nothing but `PYTHIA8DATA`. Without it, it
> runs only while the view is sourced. Sourcing `setup_env.sh` is required for
> Python regardless, so this is belt-and-braces — but it turns a confusing
> runtime loader error into a non-issue.

---

## 5. Verify, in increasing order of cost

Run these in order. If step 1 or 2 fails, the problem is your environment. If 3
fails, it is the build. Keeping them separate saves a lot of time. Timings below
are measured on an lxplus login node.

```bash
source setup_env.sh

# 1. Analysis code only — no PYTHIA, no NPZ.               ~2 s
pytest tests/ -q                                  # expect 211 passed

# 2. The shipped scan interpolates and samples.            ~5 s
python -c "
import numpy as np, sys; sys.path.insert(0,'src')
import helpers
R,op,off,names = helpers.interpolate_svj_params(
    {'mZ':1500,'rinv_pion':0.3,'mRho':20.0,'alphaD':0.4})
X = helpers.sample_svj_new(R,op,off,names,n_samples=50_000)
X = X[np.isfinite(X).all(axis=1)]
print('sampled', X.shape, 'observables:', names)"
# expect ~(49_97x, 11) — a few draws are NaN by design, see api.md

# 3. The binary runs.  50 000 events, 16 threads.          ~35 s
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
wc -l simulated/tsv/jets_default.tsv               # expect 49999

# 4. A one-point scan, end to end.                         ~20 s
python src/run_regression/scan_svj.py src/run_regression/scan_smoke.cfg
# → simulated_smoke/svj/svj_scan.npz, param_flat (1,1,1,1,1,1,184)
```

Step 4 uses the committed `scan_smoke.cfg` — one grid point at 2000 events.
It writes to `simulated_smoke/`, deliberately not `simulated/`, because a
one-point scan with `n_jobs=1` is named `svj_scan.npz` and would otherwise
overwrite the committed 6-axis production scan.

---

## 6. Scratch space in batch jobs

The scan and validation workers write a per-point cfg and TSV to
`tempfile.gettempdir()`, which honours **`$TMPDIR`**. HTCondor sets `$TMPDIR`
to the job's private scratch directory, so this works with no configuration —
confirmed in the smoke test, which reported
`scratch: /pool/condor/dir_<n>/tmp`.

Do not override `$TMPDIR` to a shared or network path. Each worker's TSV holds
`nEvent` events, and `n_outer_workers` of them exist at once — with
`nEvent=20000` and 16 workers that is a few hundred MB of churn that belongs on
node-local disk.

Request scratch explicitly if your scan is large:

```
request_disk = 4 GB
```

---

## 7. HTCondor

Every workflow runs through one shared wrapper,
[`condor/svj_job.sh`](../condor/svj_job.sh), with the settings common to all
submit files in [`condor/_common.inc`](../condor/_common.inc).

| File | Workflow | Merge step |
|------|----------|-----------|
| `smoke.sub` | one grid point, `espresso` — run first | — |
| `scan.sub` | parameter scan → `svj_scan_*.npz` | `scan_svj.py --merge --n-jobs N` |
| `tsv.sub` | batch TSV shards | `bash merge_svj_tsv.sh N` |
| `validation.sub` | production validation | `python merge_svj_validation.py` |

Job flavours cap wall time: `espresso` 20 min, `microcentury` 1 h, `longlunch`
2 h, `workday` 8 h, `tomorrow` 1 day, `testmatch` 3 days, `nextweek` 1 week.

### 7.1 Every submission

```bash
source setup_env.sh
condor_submit condor/smoke.sub
condor_q
```

No `module load`, no schedd juggling. That is deliberate, and §7.2 explains
what it cost to get there.

### 7.2 Why a little of this lives on AFS

The repository, the configs, the NPZs and all bulk data are on EOS. Two small
things are not: the script Condor executes, and the job logs. Both live under
`$SVJ_AFS` (default `~/.svj`), which `setup_env.sh` creates and populates.

That split is forced, not chosen. CERN's standard schedds reject a submit file
naming any `/eos` path in `exec`, `log`, `output`, `error` or `TransferInput`:

```
Standard batch schedds cannot use /eos paths directly within the submit file.
```

`module load lxbatch/eossubmit` routes you to a schedd that accepts those — but
that schedd then rejects every path that is **not** on `/eos`:

```
Non-supported submit file for EosSubmit schedd: Absolute paths in exec,
stdin/out/err, TransferInput must be in /eos.
```

So it is all-EOS or all-AFS; there is no mixing. And all-EOS does not work:
a job's `.out`/`.err` transferred back to EOS land in the namespace but stat as
`-??????????` indefinitely — `ls` shows the entry, `cat` says "No such file or
directory", and it never resolves. The job itself succeeds and writes its NPZ
fine, but you cannot read its stdout or stderr, which makes debugging an array
impossible. Both were confirmed by test submissions.

Hence: launcher and logs on AFS, everything else on EOS. The pieces are

| Path | What |
|---|---|
| `$SVJ_AFS/launch.sh` | `exec`s into `$SVJ_REPO/condor/svj_job.sh` after checking `SVJ_REPO` is set and reachable; kept in sync from the tracked `condor/launch.sh` by `setup_env.sh` |
| `$SVJ_AFS/logs/` | `.out`, `.err`, `.log` |
| `initialdir` | `$SVJ_AFS` — `condor_submit` otherwise uses your cwd as the job's `iwd`, and an `/eos` `iwd` is itself a rejected path |
| `environment` | carries `SVJ_REPO`; `launch.sh` runs from a scratch copy and cannot infer the repo from its own location |

**The worker node never reads AFS.** Condor transfers `launch.sh` to scratch.
The worker reads EOS, using the forwarded x509 proxy — which is why
`use_x509userproxy` matters and why an expired proxy shows up as a "cannot read
the repository" error rather than anything mentioning credentials.

Edit `condor/launch.sh` in git, then re-source `setup_env.sh`; it reinstalls
the AFS copy whenever the two differ and says so.

AFS home is a 2 GB quota, so prune `$SVJ_AFS/logs` occasionally. A 16-job array
produces 33 small text files.

Refresh the proxy before a long array — one that expires mid-array takes the
remaining jobs with it. Note `-out`: without it `voms-proxy-init` writes to
`/tmp`, which is node-local and useless to Condor.

```bash
voms-proxy-init -voms atlas -out "$X509_USER_PROXY" -valid 168:00
voms-proxy-info -file "$X509_USER_PROXY" -timeleft
```

### 7.3 Smoke test first — this is mandatory

```bash
source setup_env.sh
condor_submit condor/smoke.sub
condor_q
cat "$SVJ_AFS"/logs/smoke_*.err "$SVJ_AFS"/logs/smoke_*.out
```

One `espresso` job, one grid point, 2000 events, under a minute of compute.
It is the only test that exercises what no local run can: that a worker node
can read the EOS-resident repository, source `setup_env.sh` off EOS, find the
binary, and write results back. Read its `.err` for permission or
"no such file" errors on repository paths before you submit anything larger.

### 7.4 Parameter scan

```bash
condor_submit condor/scan.sub

# After every job finishes, merge the per-job NPZs (login node is fine):
python src/run_regression/scan_svj.py --merge --n-jobs 16
```

`N_JOBS` must agree in three places: `queue N` in the `.sub`, the `arguments`
line, and the `--merge --n-jobs` call. Set `n_outer_workers` in
`src/run_regression/scan_regression.cfg` to match `request_cpus`.

To run a different grid without editing `scan_regression.cfg`, point
`SVJ_SCAN_CFG` at another file — that is how `smoke.sub` selects
`scan_smoke.cfg`:

```
environment = "SVJ_REPO=$(SVJ_REPO_DIR) SVJ_SCAN_CFG=path/to/my.cfg"
```

### 7.5 Batch TSV generation

```bash
condor_submit condor/tsv.sub
bash merge_svj_tsv.sh 4          # after all jobs finish
```

Each job gets a distinct `seed_offset`, so PYTHIA seeds never overlap — see the
seed table in [running-a-scan.md](running-a-scan.md).

### 7.6 Production validation

```bash
condor_submit condor/validation.sub
python merge_svj_validation.py   # edit N_JOBS in it if you changed queue N
```

Each validation point costs 3 PYTHIA runs, so this is by far the most expensive
workflow. Start with `N3=2` and one job to measure a point's cost, then scale.

---

## 8. Where the data should live

Scan NPZs are small enough to stay in git. Everything else is bulk and belongs
on EOS:

```bash
export EOS_SVJ=/eos/user/${USER:0:1}/${USER}/svj
mkdir -p "$EOS_SVJ"/{tsv,scans,raw}
```

Pass EOS paths to `tsv_file` / `tsv_kin_file` in `svj_regression.cfg`.
`--save-raw` output in particular (hundreds of MB to several GB) should never
touch AFS.

To load a scan from EOS:

```python
helpers.set_svj_scan_path('/eos/user/l/lbojorqu/svj/scans/svj_scan.npz')
```

> **Before you redirect `output_dir`.** `scan_svj.py` resolves `output_dir`
> relative to the current working directory, so batch jobs must `cd` to the
> repository root first — `condor/svj_job.sh` does this.
>
> More awkwardly: **`--merge` ignores `output_dir` entirely** and always reads
> and writes `simulated/svj/`. Point `output_dir` somewhere else and the scan
> shards land there while the merge looks in `simulated/svj/` and finds
> nothing. Either leave `output_dir` at its default `simulated`, or merge by
> hand. The scan NPZs are a few MB, so the default costs little.

---

## 9. Known gaps

- The array `.sub` files (`scan`, `tsv`, `validation`) have not been submitted
  at full size. They share all their plumbing with `smoke.sub` via
  `_common.inc`, so §7.3 covers the risky part, but job flavours and the larger
  `request_cpus` / `request_memory` / `request_disk` values are unexercised.
- `merge_svj_validation.py` hardcodes `N_JOBS = 16` and the shard path pattern.
- The `validation` workflow's `--N3 125` and the `tsv` workflow's overrides are
  hardcoded in `condor/svj_job.sh`; edit them there, not in the `.sub`.
- There is no DAGMan file, so the merge step after each array is manual.
- `sample_svj_new` returns NaN for a small fraction of draws; see the
  known-limitation section of [api.md](api.md).
- `setup_env.sh` pins `LCG_110/x86_64-el9-gcc13-opt`. LCG views are eventually
  removed from CVMFS; when that one goes, pick a newer view that still carries
  pythia8 and fastjet and `export LCG_VIEW=...` (§2).
- Writing job `.out`/`.err` to EOS is broken, not merely discouraged (§7.2). If
  a future lxplus release fixes it, the AFS split in `_common.inc` could be
  dropped — retest before assuming so.

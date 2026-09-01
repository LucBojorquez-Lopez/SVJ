# Running on CERN lxplus

A complete path from `git clone` to submitted batch jobs on lxplus, and an
explicit account of **what ships with the repository versus what you have to
rebuild**.

> **What is verified, and what is not.** The build, the Python environment,
> the analysis pipeline and the `condor/svj_job.sh` wrapper (env sourcing, repo
> resolution, `$TMPDIR` handling, and the `scan` workflow end to end producing
> a valid NPZ) were all exercised on Rocky Linux 8 / GCC 8.5 / Python 3.12.
>
> **Not** verified: the HTCondor submission layer itself — the `.sub` files,
> `$(ProcId)` expansion, job flavours, and resource requests — because no
> Condor pool was available. Nor the LCG view tags in §2, which change with
> lxplus releases. Smoke-test with a single short job (§7.1) before submitting
> an array.

---

## 1. What you do and do not need to rebuild

| | Ships in the clone | Must be rebuilt on lxplus |
|---|---|---|
| Scan NPZs (`svj_scan.npz`, `working_example/`) | ✅ committed | — |
| Validation NPZs | ✅ committed | — |
| Python analysis code, tests, docs | ✅ | — |
| `Makefile` | ✅ tracked | — |
| **PYTHIA 8.317** | ❌ | yes (§3) |
| **FastJet 3.5.1** | ❌ | yes (§3) |
| **`svj_regression` binary** | ❌ gitignored | yes (§4) |
| **Python venv** | ❌ | yes (§2) |
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

### Choose your filesystem first

AFS home is quota-limited (~10 GB) and PYTHIA + FastJet together run to several
GB. Put the dependencies in your work area or on EOS, and keep only the repo
and venv in AFS:

```bash
export SVJ_WORK=/afs/cern.ch/work/${USER:0:1}/${USER}/svj
mkdir -p "$SVJ_WORK"
```

`$EOS_MGM_URL`-backed EOS paths (`/eos/user/${USER:0:1}/${USER}/`) work too and
have far more room, but note that EOS is **not** a good place for the build
itself — compilation does many small writes. Build in `work`, store data on EOS.

### Compiler and Python via LCG

The system GCC on lxplus may predate the C++17 support this project needs.
Source an LCG view first — list what is actually available rather than copying
the tag below verbatim:

```bash
ls /cvmfs/sft.cern.ch/lcg/views/            # pick a current LCG_* release
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
g++ --version         # need >= 9
python3 --version     # need >= 3.9
```

Put that `source` line in a small `setup_env.sh` of your own — **every** shell
and **every** batch job needs it before activating the venv.

### Create the venv

Create it *after* sourcing the LCG view so it inherits that Python:

```bash
cd "$SVJ_WORK"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy "scipy>=1.6" matplotlib tqdm pytest ipywidgets ipympl jupyterlab
```

Drop `ipywidgets ipympl jupyterlab` if you will not run the GUI, and
`matplotlib` too if you only need the scan and fits.

---

## 3. Build PYTHIA and FastJet

```bash
cd "$SVJ_WORK"

# PYTHIA 8.317
wget https://pythia.org/download/pythia83/pythia8317.tgz
tar xzf pythia8317.tgz
cd pythia8317 && ./configure && make -j4 && cd ..

# FastJet 3.5.1
wget http://fastjet.fr/repo/fastjet-3.5.1.tar.gz
tar xzf fastjet-3.5.1.tar.gz
cd fastjet-3.5.1
./configure --prefix="$SVJ_WORK/fastjet3"
make -j4 install && cd ..
```

Use `-j4`, not `-j$(nproc)` — lxplus login nodes are shared and heavily
parallel builds get throttled or killed. Expect 15–30 minutes for PYTHIA.

---

## 4. Clone and build the generator

```bash
cd "$SVJ_WORK"
git clone https://github.com/LucBojorquez-Lopez/SVJ.git
cd SVJ

export PYTHIA_DIR="$SVJ_WORK/pythia8317"
export FASTJET_DIR="$SVJ_WORK/fastjet3"

make check-deps        # confirms both are findable; builds nothing
make svj_regression
```

Add those two `export` lines to your `setup_env.sh` so plain
`make svj_regression` keeps working, and because batch jobs that rebuild need
them.

Both paths are resolved to **absolute** paths and written into the binary's
rpath, so the executable runs from any working directory.

---

## 5. Verify, in increasing order of cost

```bash
# 1. Analysis code only — no PYTHIA, no NPZ.  ~10 s
pytest tests/ -q                                  # expect 211 passed

# 2. The shipped scan interpolates and samples.  ~5 s
python -c "
import numpy as np, sys; sys.path.insert(0,'src')
import helpers
R,op,off,names = helpers.interpolate_svj_params(
    {'mZ':1500,'rinv_pion':0.3,'mRho':20.0,'alphaD':0.4})
X = helpers.sample_svj_new(R,op,off,names,n_samples=50_000)
X = X[np.isfinite(X).all(axis=1)]
print('sampled', X.shape, 'observables:', names)"

# 3. The binary runs.  ~1 min
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
wc -l simulated/tsv/jets_default.tsv

# 4. A one-point scan, end to end.  ~2 min
#    Copy scan_regression.cfg, set every axis to n=1 and nEvent=2000 first.
python src/run_regression/scan_svj.py my_tiny_scan.cfg
```

If step 1 or 2 fails, the problem is your venv. If 3 fails, it is the C++
build. Keeping them separate saves a lot of time.

---

## 6. Scratch space in batch jobs

The scan and validation workers write a per-point cfg and TSV to
`tempfile.gettempdir()`, which honours **`$TMPDIR`**. HTCondor sets `$TMPDIR`
to the job's private scratch directory, so this works with no configuration.

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

The `run_svj_*.sh` scripts at the repository root are **SLURM** array jobs for
a specific Harvard partition. They do not run under HTCondor. Below is the
equivalent for each; the shared wrapper is
[`condor/svj_job.sh`](../condor/svj_job.sh).

The mapping is mechanical:

| SLURM | HTCondor |
|-------|----------|
| `--array=0-15` | `queue 16` |
| `$SLURM_ARRAY_TASK_ID` | `$(ProcId)` |
| `-c 16` | `request_cpus = 16` |
| `--mem=2G` | `request_memory = 2 GB` |
| `-t 0-10:00` | `+JobFlavour = "tomorrow"` |
| `-p <partition>` | (no equivalent; flavour sets the wall-clock limit) |
| `--output=logs/x_%A_%a.out` | `output = logs/x_$(ClusterId)_$(ProcId).out` |
| `--dependency=afterok:<id>` | a DAGMan `.dag` file, or submit the merge by hand |

Job flavours cap wall time: `espresso` 20 min, `microcentury` 1 h, `longlunch`
2 h, `workday` 8 h, `tomorrow` 1 day, `testmatch` 3 days, `nextweek` 1 week.

### 7.1 Smoke test first

Before any array, submit **one** short job and read its log:

```bash
condor_submit condor/scan.sub -append 'queue 1' \
                              -append '+JobFlavour = "espresso"'
condor_q
# when done:
cat logs/scan_*.err
```

### 7.2 Parameter scan

```bash
condor_submit condor/scan.sub

# After every job finishes, merge the per-job NPZs (login node is fine):
python src/run_regression/scan_svj.py --merge --n-jobs 16
```

`N_JOBS` must agree in three places: `queue N` in the `.sub`, the `--n-jobs`
argument, and the `--merge --n-jobs` call. Set `n_outer_workers` in
`src/run_regression/scan_regression.cfg` to match `request_cpus`.

### 7.3 Batch TSV generation

```bash
condor_submit condor/tsv.sub
bash merge_svj_tsv.sh 4          # after all jobs finish
```

Each job gets a distinct `seed_offset`, so PYTHIA seeds never overlap — see the
seed table in [running-a-scan.md](running-a-scan.md).

### 7.4 Production validation

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

Point the scan output there with `output_dir` in the `[simulation]` section of
`scan_regression.cfg`, and pass EOS paths to `tsv_file` / `tsv_kin_file` in
`svj_regression.cfg`. `--save-raw` output in particular (hundreds of MB to
several GB) should never touch AFS.

To load a scan from EOS:

```python
helpers.set_svj_scan_path('/eos/user/l/lbojorqu/svj/scans/svj_scan.npz')
```

Note that `scan_svj.py` resolves `output_dir` **relative to the current working
directory**, so batch jobs must `cd` to the repository root first — the wrapper
in `condor/svj_job.sh` does this.

---

## 9. Known gaps

- The `.sub` files in `condor/` have never been submitted to a real Condor
  pool. The wrapper they call is tested; the submission layer is not. Treat
  §7.1 as mandatory.
- The `validation` workflow's `--N3 125` and the `tsv` workflow's overrides are
  hardcoded in `condor/svj_job.sh`; edit them there, not in the `.sub`.
- `merge_svj_validation.py` hardcodes `N_JOBS = 16` and the shard path pattern.
- There is no DAGMan file, so the merge step after each array is manual.
- `sample_svj_new` returns NaN for a small fraction of draws; see the
  known-limitation section of [api.md](api.md).

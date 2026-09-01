# Running on CERN lxplus

A complete path from `git clone` to submitted batch jobs on lxplus, and an
explicit account of **what ships with the repository versus what you have to
rebuild**.

> **Status.** Verified on lxplus (RHEL 9.8, EL9 gcc 11.5, LCG_110, Python 3.13):
> the environment, `tools/build_deps.sh`, `make svj_regression`, the §5 ladder,
> `condor/smoke.sub`, a 64-point scan run both locally and as a 4-job Condor
> array, the `--merge` step, and `fit_raw.py`.
>
> Not verified: an array at full production size. `queue 16` with
> `request_cpus = 16` and the longer job flavours are untried; everything those
> files do apart from the scale is shared with `smoke.sub` via `_common.inc`.
> Run §7.3 before any array — it costs a minute.

---

## 0. Setting up a clone on EOS, start to finish

The whole procedure, for when you are doing this again on a fresh or re-merged
checkout. Each step links to the section that explains it.

```bash
# 1. Clone onto EOS.
cd /eos/user/${USER:0:1}/${USER}
git clone https://github.com/LucBojorquez-Lopez/SVJ.git && cd SVJ   # §3

# 2. Environment.  Creates $SVJ_AFS (~/.svj) and installs the Condor launcher.
source setup_env.sh                                                 # §2

# 3. Dependencies.  ~20 min.  Skip to run against the LCG view instead.
bash tools/build_deps.sh                                            # §2
source setup_env.sh          # re-source; $SVJ_DEPS should now say `local`

# 4. Generator.
make svj_regression                                                 # §4

# 5. Verify locally, cheapest first.
pytest tests/ -q                                                    # §5

# 6. Grid proxy, for EOS access from worker nodes.  Note -out.
voms-proxy-init -voms atlas -out "$X509_USER_PROXY" -valid 168:00    # §7.2

# 7. One Condor job before any array.  Mandatory.
condor_submit condor/smoke.sub && condor_q                           # §7.3
cat "$SVJ_AFS"/logs/smoke_*.err
```

### What has to be re-done, and what does not

`setup_env.sh` and `tools/build_deps.sh` are idempotent — re-running them on an
existing setup is safe and cheap.

| | Survives a re-clone? |
|---|---|
| `$SVJ_WORK` (PYTHIA, FastJet — a sibling of the repo, not inside it) | yes, reused |
| `$SVJ_AFS` (`~/.svj`: launcher, logs) | yes, launcher re-synced from the repo |
| x509 proxy | yes, until it expires |
| `svj_regression` binary | **no** — gitignored, rebuild with `make` |
| Downloaded PYTHIA/FastJet tarballs | cached in `$SVJ_WORK`, not re-fetched |

Rebuild the generator (`make clean && make svj_regression`) whenever you switch
between local and view dependencies: the library paths are baked into the
binary's rpath at link time.

---

## 1. What you do and do not need to rebuild

| | Ships in the clone | Must be rebuilt on lxplus |
|---|---|---|
| Scan NPZs (`svj_scan.npz`, `working_example/`) | ✅ committed | — |
| Validation NPZs | ✅ committed | — |
| Python analysis code, tests, docs | ✅ | — |
| `Makefile` | ✅ tracked | — |
| **PYTHIA 8.317** | ❌ | yes — `tools/build_deps.sh`, ~20 min (§2) |
| **FastJet 3.5.1** | ❌ | yes — same script, ~2 min (§2) |
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

```bash
source setup_env.sh
```

That is the whole setup for an existing checkout. It reports which PYTHIA and
FastJet it selected via `$SVJ_DEPS`.

### Where each piece comes from

| Piece | Source |
|---|---|
| Python 3.13, numpy, scipy, matplotlib, tqdm, pytest, jupyter | LCG view on CVMFS |
| PYTHIA 8.317, FastJet 3.5.1 | **our own builds** under `$SVJ_WORK`, with the LCG view as fallback |
| Compiler for `svj_regression` | the **system** gcc when using our builds; the view's gcc 13 when falling back |

Python comes from CVMFS because a self-built stack is not worth it here: the
system Python is 3.9, which is end-of-life and caps you at numpy 2.0.x /
scipy 1.12.x (numpy 2.1+ and scipy 1.13+ both require 3.10), whereas the view
gives 3.13 with numpy 2.4 and scipy 1.17. There is nowhere to build a newer
Python either — this account has no `/afs/cern.ch/work` volume and ~1 GB free
in its 2 GB AFS home.

PYTHIA and FastJet are ours because they are the pieces you may actually want
to modify — a Hidden Valley study is a plausible reason to patch PYTHIA — and
because building them ourselves means the generator binary keeps working when
an LCG view is retired.

### Building PYTHIA and FastJet

```bash
source setup_env.sh
bash tools/build_deps.sh      # ~20 min, mostly PYTHIA
source setup_env.sh           # re-source: it now selects the local builds
echo "$SVJ_DEPS"              # -> local
```

The script downloads both tarballs into `$SVJ_WORK` (cached, so a rebuild does
not re-fetch), builds on local disk, prunes PYTHIA's object files, and copies
the finished trees to EOS. Installed size is ~205 MB, not the several GB an
unpruned in-place build suggests.

Two choices inside it are deliberate:

- **System gcc, not the view's.** A binary linked against these then needs
  nothing from CVMFS. The repo `Makefile` follows automatically: it
  `-include`s `$PYTHIA_DIR/examples/Makefile.inc`, which a source build writes
  (`CXX=/usr/bin/g++`) and a CVMFS view does not ship at all.
- **Build local, copy to EOS** — but configure with the *final* EOS prefix.
  A direct build on EOS is many thousands of small compile-and-link writes,
  the one workload EOS is bad at; copying a finished tree is what it is good
  at. The trap is that both packages bake the configured prefix into their
  output: `fastjet-config` reports it, the libtool `.la` files record it, and
  `libpythia8.so` gets it as a compiled-in ELF `RPATH`. Configure with the
  build directory and copy afterwards and you get a library whose `RPATH`
  points into `/tmp` — fine on the machine that built it, broken on every
  worker node, and invisible until a batch job fails. `build_deps.sh`
  configures with the destination, stages FastJet through `DESTDIR`, and
  asserts afterwards that no build-directory path survived.

`-j4`, not `-j$(nproc)`: lxplus login nodes are shared and routinely sit above
a load average of their core count.

### Adding another dependency

If the generator grows a dependency beyond PYTHIA and FastJet — a detector
simulation, say — it has to be added in two places, and both matter:

1. **`tools/build_deps.sh`** — fetch, configure with its **final** `$SVJ_WORK`
   prefix, build on local disk, copy in. Extend the relocatability check at the
   end to cover its libraries.
2. **The `$SVJ_DEPS=local` branch of `setup_env.sh`** — export whatever the
   build needs, and add it to the guard that decides local-versus-view. The
   fallback branch has no equivalent unless the LCG view happens to ship it, so
   decide explicitly what should happen when the local build is missing.

The `Makefile` needs a corresponding `-I` / `-L` / `-Wl,-rpath` triple. Use
absolute paths, as the existing two do, so the binary runs from any working
directory and from a Condor scratch dir.

Then re-run §5 and §7.3: a new shared library is exactly the kind of thing that
works on the login node and fails on a worker, because the login node has paths
a worker does not.

### The LCG view, and the fallback

`setup_env.sh` falls back to the view when `$SVJ_WORK` has no usable build, so
a fresh clone works before you build anything and a half-finished build
degrades rather than blocks. The view happens to ship exactly PYTHIA 8.317 and
FastJet 3.5.1 — the versions this project targets — so the fallback is a real
one, not a token.

To pin a different view, `export LCG_VIEW=...` before sourcing. List what
exists rather than trusting a tag in a document:

```bash
ls /cvmfs/sft.cern.ch/lcg/views/          # pick an LCG_* release
ls /cvmfs/sft.cern.ch/lcg/views/LCG_110/  # pick a platform
```

A view used as fallback must provide `lib/libpythia8.so` and
`bin/fastjet-config`; `make check-deps` says so in a second if it does not.

> **One trap worth knowing about.** The view's own `setup.sh` reads `$COMPILER`
> unguarded, so it aborts with `COMPILER: unbound variable` under `set -u`.
> `condor/svj_job.sh` runs `set -euo pipefail`, which means a naive `source`
> kills every batch job on that line. `setup_env.sh` shields the source and
> restores the caller's flags afterwards, so callers need no special handling.

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
bash tools/build_deps.sh   # PYTHIA + FastJet, ~20 min; skip to use the view
source setup_env.sh        # re-source so it picks up the new builds

make check-deps            # confirms both are findable; builds nothing
make svj_regression        # ~4 seconds
```

`PYTHIA_DIR` / `FASTJET_DIR` come from `setup_env.sh`, so plain
`make svj_regression` is all you ever type. Both are resolved to **absolute**
paths and written into the binary's rpath, so the executable runs from any
working directory.

Run `make clean && make svj_regression` after switching between local and view
dependencies — the rpath is baked in at link time, so a stale binary keeps
pointing at whichever install it was linked against.

> **Where the compiler comes from, and why it differs by branch.**
>
> With **local builds**, the `Makefile` picks up `CXX=/usr/bin/g++` from
> `$PYTHIA_DIR/examples/Makefile.inc`, so the binary links the system
> `libstdc++` that exists on every worker node. Nothing from CVMFS is needed at
> runtime.
>
> With the **view**, there is no `Makefile.inc`, so `CXX` comes from `PATH` —
> the view's gcc 13 — and the binary needs that gcc's `libstdc++`. EL9's system
> copy is gcc-11-era and the binary dies with `GLIBCXX_3.4.31 not found`.
> `setup_env.sh` handles this by passing an rpath through `GZIP_LIB`, a
> variable only a source build's `Makefile.inc` would otherwise set — so the
> `Makefile` needs no edit, and the local branch correctly leaves it unset.

> **`ldd` will still show CVMFS, and that is fine.** Sourcing `setup_env.sh`
> puts the view on `LD_LIBRARY_PATH` for Python's sake, so `libstdc++` and
> `libgcc_s` resolve from CVMFS in that shell. That is a property of the shell,
> not of the binary: `readelf -d` shows an RPATH containing only `$SVJ_WORK`
> paths, and in a clean environment the same two libraries resolve from
> `/lib64`. Both work — gcc 13's `libstdc++` is backward compatible with a
> gcc-11-built binary. To check what the binary itself requires:
>
> ```bash
> readelf -d src/generate_events/svj_regression | grep -E 'RPATH|NEEDED'
> env -i PATH=/usr/bin:/bin ldd src/generate_events/svj_regression
> ```

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

- An array has not been run at full production size. The 64-point / 4-job run
  that validated `scan.sub`'s mechanics used `request_cpus = 8` and
  `microcentury`; `queue 16` with `request_cpus = 16` and `tomorrow` is untried.
- `tools/build_deps.sh` builds PYTHIA and FastJet only. Anything else the
  generator comes to depend on needs adding there, and to the `$SVJ_DEPS=local`
  branch of `setup_env.sh`.
- `merge_svj_validation.py` hardcodes `N_JOBS = 16` and the shard path pattern.
- The `validation` workflow's `--N3 125` and the `tsv` workflow's overrides are
  hardcoded in `condor/svj_job.sh`; edit them there, not in the `.sub`.
- There is no DAGMan file, so the merge step after each array is manual.
- `sample_svj_new` returns NaN for a small fraction of draws; see the
  known-limitation section of [api.md](api.md).
- `setup_env.sh` pins `LCG_110/x86_64-el9-gcc13-opt` for Python. LCG views are
  eventually removed from CVMFS; when that one goes, `export LCG_VIEW=...` to a
  newer one (§2). With local PYTHIA/FastJet builds the C++ side is unaffected,
  but the Python stack follows the view.
- `tools/build_deps.sh` pins PYTHIA 8317 and FastJet 3.5.1 in two variables at
  the top. Bumping either is a one-line change plus a rebuild, but nothing
  verifies that the repo's observable code still matches a new PYTHIA.
- Job `.out`/`.err` cannot be written to EOS (§7.2). If a future lxplus release
  changes that, the AFS split in `_common.inc` could be dropped — retest before
  assuming so.

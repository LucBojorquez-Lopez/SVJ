# Setup

This walks through a **fresh clone to a working build**, on a generic Linux
machine and on CERN lxplus. Nothing here assumes a previous checkout.

## Prerequisites

| Requirement | Minimum version | Notes |
|-------------|-----------------|-------|
| Linux x86-64 | — | Binary is ELF 64-bit; macOS not tested |
| g++ | 9 | C++17 `<filesystem>` required |
| Python | 3.9 | 3.12 recommended; CI runs 3.11 |
| PYTHIA | 8.317 | built in-place; see §2 |
| FastJet | 3.5.1 | installed with `--prefix`; see §3 |

Everything Python-side (tests, fitting, interpolation, GUI) works without
PYTHIA or FastJet. You only need the C++ toolchain to **generate new events**.
If you just want to sample from the shipped scan, skip to §4.

---

## 1. Clone

```bash
git clone https://github.com/LucBojorquez-Lopez/SVJ.git
cd SVJ
```

The working tree is ~35 MB, but **the clone is ~305 MB**: early history
contains several revisions of multi-tens-of-MB raw TSVs (~570 MB of blob
content, since compacted into the pack). Those paths are gitignored now, so
the repository will not keep growing this way, but the existing history still
carries them. Budget accordingly on quota-limited filesystems, or use
`git clone --depth 1` when you only need a working copy.

The checkout includes the default scan
(`simulated/svj/working_example/`, 7.5 MB) and a larger 6-axis scan
(`simulated/svj/svj_scan.npz`, 25 MB), so the API and GUI work immediately —
no simulation required.

### Directory layout for the C++ build

The Makefile defaults to finding PYTHIA and FastJet as **siblings** of this
repository:

```
<parent>/
├── pythia8317/    # PYTHIA 8.317, built in-place
├── fastjet3/      # FastJet >= 3, installed with --prefix
└── SVJ/           # this repo  <- you are here
```

This is only the default. Both locations are overridable — see §4.

---

## 2. Install PYTHIA 8.317

```bash
# From <parent>/ (one level above SVJ/)
wget https://pythia.org/download/pythia83/pythia8317.tgz
tar xzf pythia8317.tgz        # extracts to pythia8317/
cd pythia8317
./configure
make -j$(nproc)
cd ..
```

Builds in place; no `make install`. The shared library lands at
`pythia8317/lib/libpythia8.so`.

---

## 3. Install FastJet 3.5.1

```bash
# From <parent>/
wget http://fastjet.fr/repo/fastjet-3.5.1.tar.gz
tar xzf fastjet-3.5.1.tar.gz
cd fastjet-3.5.1
./configure --prefix="$(pwd)/../fastjet3"
make -j$(nproc) install
cd ..
```

Installs to `fastjet3/`, with the config binary at
`fastjet3/bin/fastjet-config` — the Makefile queries it for compiler and
linker flags.

---

## 4. Python environment

```bash
python3 -m venv ~/venvs/svj
source ~/venvs/svj/bin/activate
pip install numpy "scipy>=1.6" matplotlib tqdm pytest ipywidgets ipympl jupyterlab
```

| Package | Needed for |
|---------|-----------|
| `numpy`, `scipy>=1.6` | everything |
| `matplotlib` | `diagnostics.py`, the GUI |
| `tqdm` | progress bars in the KLD grid helpers (optional) |
| `pytest` | the test suite |
| `ipywidgets`, `ipympl`, `jupyterlab` | the interactive GUI only |

Confirm the analysis side works before touching the C++ build:

```bash
pytest tests/ -v          # 211 tests, no PYTHIA or NPZ required
```

---

## 5. Build the event generator

The `Makefile` is tracked in git and needs no editing:

```bash
make svj_regression
```

To point at installations elsewhere, override either variable — on the command
line or from the environment:

```bash
make svj_regression PYTHIA_DIR=/path/to/pythia8317 \
                    FASTJET_DIR=/path/to/fastjet3
```

Both paths are resolved to absolute paths before being written into the
binary's rpath, so the resulting executable runs from **any** working
directory.

Other targets:

```bash
make check-deps    # verify PYTHIA/FastJet are findable; build nothing
make clean         # remove the binary
```

`check-deps` runs automatically before every build and prints an actionable
error naming the variable to set if either dependency is missing.

> **No binary is committed.** `src/generate_events/svj_regression` is
> gitignored — a compiled binary carries absolute library paths from the
> machine that built it and would be useless anywhere else. Always run
> `make svj_regression` after a fresh clone.

---

## 6. Verify the build

```bash
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
```

Writes `simulated/tsv/jets_default.tsv` (~50 000 events; the output directory
is created automatically). Generated TSVs are gitignored — they run to tens of
MB and are reproducible from the binary plus a cfg.

---

## Running on CERN lxplus

The build works on lxplus, with three differences from the layout above.

**Build PYTHIA and FastJet somewhere with space.** AFS home is quota-limited
(~10 GB) and the two builds together are several GB. Use your work area or
EOS:

```bash
export SVJ_DEPS=/afs/cern.ch/work/${USER:0:1}/${USER}/svj-deps
mkdir -p "$SVJ_DEPS" && cd "$SVJ_DEPS"
# ... then follow §2 and §3 inside $SVJ_DEPS ...
```

**Point the Makefile at them.** Since they are no longer siblings of the repo:

```bash
cd /path/to/SVJ
make svj_regression PYTHIA_DIR="$SVJ_DEPS/pythia8317" \
                    FASTJET_DIR="$SVJ_DEPS/fastjet3"
```

Persist it in your shell profile so plain `make svj_regression` keeps working:

```bash
export PYTHIA_DIR="$SVJ_DEPS/pythia8317"
export FASTJET_DIR="$SVJ_DEPS/fastjet3"
```

**Use a recent compiler and Python via LCG.** The system g++ on lxplus may
predate the C++17 support this project needs. Source an LCG view first — pick
one matching the current lxplus architecture rather than copying the tag
below verbatim (`ls /cvmfs/sft.cern.ch/lcg/views/` to see what is available):

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
g++ --version        # expect >= 9
```

Create the venv *after* sourcing the LCG view so it inherits that Python, and
re-source the view in every batch job that activates the venv.

> **Batch jobs.** The `run_svj_*.sh` scripts at the repository root are SLURM
> array jobs written for a specific Harvard partition
> (`-p arguelles_delgado`). They will not run under HTCondor as-is. See
> [running-a-scan.md](running-a-scan.md) for what each one does and which
> parameters need translating.

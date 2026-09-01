# Setup

This walks through a **fresh clone to a working build** on a generic Linux
machine. Nothing here assumes a previous checkout. For CERN lxplus
specifically — AFS quota, LCG views, HTCondor — see [lxplus.md](lxplus.md).

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

A full clone is **~97 MB**: a 35 MB working tree plus ~62 MB of history.
(Raw TSVs were purged from history in Sept 2026 and are now gitignored, so the
repository will not grow that way again.) Use `git clone --depth 1` if you only
want a working copy and no history.

**The scan NPZs are committed**, so the API, GUI and validation plots work
immediately after cloning — no simulation and no C++ build required:

| Path | Size | |
|------|------|--|
| `simulated/svj/working_example/svj_scan.npz` | 6.7 MB | the default scan |
| `simulated/svj/working_example/validation_production.npz` | 755 KB | its validation |
| `simulated/svj/svj_scan.npz` | 25 MB | larger 6-axis scan |
| `simulated/svj/validation_production.npz` | 1.1 MB | its validation |
| `simulated/v1/regression_scan.npz` | 515 KB | archived v1 grid (KLD utilities) |

You only need §2, §3 and §5 below to **generate new events**. To work with the
shipped scans, go straight to §4.

### Directory layout for the C++ build

The Makefile defaults to finding PYTHIA and FastJet as **siblings** of this
repository:

```
<parent>/
├── pythia8317/    # PYTHIA 8.317, built in-place
├── fastjet3/      # FastJet >= 3, installed with --prefix
└── SVJ/           # this repo  <- you are here
```

This is only the default. Both locations are overridable — see §5.

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

lxplus needs a different filesystem layout (AFS quota), a compiler and Python
from an LCG view, and HTCondor rather than SLURM. That is covered in its own
guide, including exactly what ships in the clone versus what must be rebuilt:

**→ [lxplus.md](lxplus.md)**

The short version: the scan NPZs are committed, so the API, GUI and validation
plots work as soon as the clone finishes. You need PYTHIA, FastJet and the
`svj_regression` build only to generate *new* events. Point the Makefile at
your own installs with `PYTHIA_DIR` / `FASTJET_DIR` (§5 above), and use the
submit files in [`condor/`](../condor/) instead of the SLURM scripts.

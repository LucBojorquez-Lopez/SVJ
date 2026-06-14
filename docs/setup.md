# Setup

## Prerequisites

| Requirement | Minimum version | Notes |
|-------------|-----------------|-------|
| Linux x86-64 | — | Binary is ELF 64-bit; macOS not tested |
| g++ | 9 | C++17 `<filesystem>` required |
| Python | 3.8 | |
| PYTHIA | 8.317 | built in-place; see §3 below |
| FastJet | 3.5.1 | installed with `--prefix`; see §4 below |

## Required directory layout

The Makefile hardcodes `PYTHIA_DIR = ../pythia8317` and `FASTJET_DIR = ../fastjet3`,
so both libraries must live **as siblings** of this repository:

```
<parent>/
├── pythia8317/    # PYTHIA 8.317 built in-place
├── fastjet3/      # FastJet ≥ 3 installed with --prefix
└── mySVJ/         # this repo  ← you are here
```

## Install PYTHIA 8.317

```bash
# Run from <parent>/ (one level above mySVJ/)
wget https://pythia.org/download/pythia83/pythia8317.tgz
tar xzf pythia8317.tgz        # extracts to pythia8317/
cd pythia8317
./configure
make -j$(nproc)
cd ..
```

This builds PYTHIA in-place; no `make install` step is needed.
The shared library ends up at `pythia8317/lib/libpythia8.so`.

## Install FastJet 3.5.1

```bash
# Run from <parent>/
wget http://fastjet.fr/repo/fastjet-3.5.1.tar.gz
tar xzf fastjet-3.5.1.tar.gz
cd fastjet-3.5.1
./configure --prefix="$(pwd)/../fastjet3"
make -j$(nproc) install
cd ..
```

This installs the library to `fastjet3/` and the config binary to
`fastjet3/bin/fastjet-config`, which the Makefile queries for compiler/linker flags.

## Install Python packages

```bash
pip install numpy "scipy>=1.6" matplotlib ipywidgets ipympl jupyterlab
```

`scipy ≥ 1.6` is required for `scipy.stats.multivariate_t`.
`ipympl` enables the `%matplotlib widget` backend used by the Jupyter GUI.

## Create the Makefile

The `Makefile` is **not tracked in git** (it is listed in `.gitignore`) because it
contains paths that are local to each machine. Create it at the repository root:

```bash
# From mySVJ/ root:
cat > Makefile << 'EOF'
PYTHIA_DIR  = ../pythia8317
FASTJET_DIR = ../fastjet3
-include $(PYTHIA_DIR)/examples/Makefile.inc

CXX_COMMON += -std=c++17 -pthread

CXX_COMMON := $(OBJ_COMMON) -I$(PYTHIA_DIR)/include $(CXX_COMMON) $(GZIP_LIB)
CXX_COMMON += -L$(PYTHIA_DIR)/lib -Wl,-rpath,$(PYTHIA_DIR)/lib -lpythia8 -ldl
CXX_COMMON += $(shell $(FASTJET_DIR)/bin/fastjet-config --cxxflags --libs)
CXX_COMMON += -lstdc++fs

src/generate_events/svj_regression: src/generate_events/svj_regression.cc
	$(CXX) $< -o $@ $(CXX_COMMON)

.PHONY: svj_regression
svj_regression: src/generate_events/svj_regression
EOF
```

Adjust `PYTHIA_DIR` and `FASTJET_DIR` if your installations live elsewhere.
The indented recipe line uses a **tab** character — `cat << 'EOF'` preserves it. If it gets lost, open the Makefile and add the tab manually.

## Build the event generator

```bash
# From the mySVJ/ root:
make svj_regression
```

This compiles `src/generate_events/svj_regression.cc` against PYTHIA8 and FastJet
and writes the binary to `src/generate_events/svj_regression`.

> **Note**: A pre-compiled binary is tracked in git for convenience, but it has
> absolute library paths baked in from the original build machine. Always
> re-run `make svj_regression` after a fresh clone before trying to run or scan.

## Verify the build

```bash
# From the mySVJ/ root (the binary resolves ../pythia8317 relative to here):
src/generate_events/svj_regression src/generate_events/svj_regression.cfg
```

A successful run writes `simulated/tsv/jets_default.tsv` (~50 000 events).
The output directory is created automatically.

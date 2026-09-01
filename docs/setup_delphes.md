# DELPHES setup (detector-level stream)

This is a **separate, additive companion** to [docs/setup.md](setup.md) --
it does not change anything about the existing truth-level pipeline
(PYTHIA8 + FastJet, `svj_regression.cc` / `scan_svj.py`). It covers two
things built on top of the same ROOT/Delphes dependencies:

- **`svj_regression_delphes`** -- a production binary that plugs into
  `scan_svj.py` / `validate_grid.py` / `validate_production.py` as an
  alternative to `svj_regression`, for full-scale scan and TSV generation
  with Delphes-level reconstruction instead of truth-level clustering. See
  `docs/running-a-scan.md`'s "Truth vs. Delphes" section for day-to-day
  usage; this doc covers the build and architecture.
- **`svj_delphes_test`** -- a diagnostic driver for investigating *how* and
  *why* Delphes reconstruction changes a given event (paired truth/Delphes
  output, outlier finding, constituent-level lego plots).

## New dependencies

Two new sibling directories, alongside the existing `pythia8317/` and
`fastjet3/`:

```
<parent>/
├── pythia8317/        # existing
├── fastjet3/          # existing
├── root/               # new
├── delphes3.5.1/       # new
└── mySVJ/              # this repo
```

Neither PYTHIA8 nor FastJet need ROOT; it's a new dependency introduced
only for the Delphes stream.

### 1. Install ROOT

Download a precompiled binary release matching your OS/compiler rather than
building from source (a from-source build needs `cmake` and can take
30-90+ minutes). Get the exact filename for your platform from
`https://root.cern/download/` (not `https://root.cern/install/all_releases/`,
which is a changelog page with no file links).

```bash
cd <parent-of-mySVJ>

wget https://root.cern/download/root_v6.34.08.Linux-ubuntu20.04-x86_64-gcc9.4.tar.gz \
     -O root.tar.gz
tar xzf root.tar.gz          # extracts to ./root/
rm root.tar.gz

source root/bin/thisroot.sh  # every new shell, before building/running anything Delphes-related
root-config --version        # sanity check
```

Substitute the filename above for your own OS/compiler if you're not on
Ubuntu 20.04 / gcc 9.4. ROOT ships patch releases fairly often, so check the
download page for the current version.

### 2. `tclsh`

Delphes' `./configure` runs `tclsh ./doc/genMakefile.tcl`, and Delphes
detector cards are themselves Tcl scripts parsed at runtime. Check
`which tclsh`; if missing, install with `apt install tcl` (Ubuntu/Debian) or
`conda install tcl`.

### 3. Build Delphes 3.5.1

```bash
cd <parent-of-mySVJ>
wget https://github.com/delphes/delphes/archive/refs/tags/3.5.1.tar.gz
tar xzf 3.5.1.tar.gz && mv delphes-3.5.1 delphes3.5.1
cd delphes3.5.1
./configure
make -j$(nproc) HAS_PYTHIA8=true PYTHIA8=$(pwd)/../pythia8317
cd ..
```

**Troubleshooting: `undefined reference to tbb::...` while linking
`libImt.so`.** ROOT's precompiled binary's implicit-multithreading module
(`libImt.so`) is built against classic TBB (pre-"oneTBB", SONAME
`libtbb.so.2`), which may not be installed on your system at all
(`ldd root/lib/libImt.so | grep tbb` shows `not found`). Fix:

```bash
sudo apt install libtbb2 libtbb-dev
```

Ubuntu 20.04's `focal/universe` archive ships the matching classic-TBB
version (`2020.1-2`). A newer TBB from conda-forge (`oneTBB`, 2021+) does
**not** work here -- it dropped the `tbb::task`-based API this ROOT build
expects, so don't install TBB via conda instead of `apt`.

## Makefile addition

The project's `Makefile` is not tracked in git (see `docs/setup.md`) --
add these targets to your own local copy, alongside the existing
`svj_regression` target:

```makefile
ROOT_DIR    = ../root
DELPHES_DIR = ../delphes3.5.1

src/generate_events/svj_regression_delphes: src/generate_events/svj_regression_delphes.cc
	$(CXX) $< -o $@ $(CXX_COMMON) \
	  -I$(DELPHES_DIR) -I$(DELPHES_DIR)/external \
	  $(shell $(ROOT_DIR)/bin/root-config --cflags --libs) -lEG \
	  -L$(DELPHES_DIR) -Wl,-rpath,$(DELPHES_DIR) -lDelphes

.PHONY: svj_regression_delphes
svj_regression_delphes: src/generate_events/svj_regression_delphes

src/generate_events/svj_delphes_test: src/generate_events/svj_delphes_test.cc
	$(CXX) $< -o $@ $(CXX_COMMON) \
	  -I$(DELPHES_DIR) -I$(DELPHES_DIR)/external \
	  $(shell $(ROOT_DIR)/bin/root-config --cflags --libs) -lEG \
	  -L$(DELPHES_DIR) -Wl,-rpath,$(DELPHES_DIR) -lDelphes

.PHONY: svj_delphes_test
svj_delphes_test: src/generate_events/svj_delphes_test
```

This reuses `$(CXX_COMMON)` (already has the Pythia8/FastJet flags from the
existing template) and layers Delphes/ROOT flags on top.

**Target naming**: the file-producing rule's target must be the *full
path* (`src/generate_events/svj_regression_delphes`), not the bare name --
`make` runs from the project root, so a bare target name writes the binary
to the repo root instead of `src/generate_events/`. Use the two-rule
pattern above (a real file-path rule plus a `.PHONY` alias for the short
name), matching the existing `svj_regression` target.

**`-lEG`**: `root-config --libs` alone does not include ROOT's `libEG`
(Event Generator interface), where `TDatabasePDG`/`TParticlePDG` live --
both binaries use them directly, and several Delphes modules use them
internally. Delphes' own top-level `Makefile` appends `-lEG` on top of
`root-config --libs` for the same reason
(`DELPHES_LIBS = $(shell $(RC) --libs) -lEG`).

## Production binary: `svj_regression_delphes`

A drop-in Delphes-level counterpart to `svj_regression`: it writes a
**plain, single-stream TSV** with the same `# name\tname\t...` header
convention, so it works unmodified with `observables.py` / `scan_svj.py` /
`validate_fit.py` / `validate_grid.py` / `validate_production.py`.

**Shared with `svj_regression`.** Both binaries compute the same
28-observable tuple via the shared `svj_observables_common.h` header (see
`docs/extending-observables.md`) -- clustering, thrust, hemisphere mass,
E2C/E3C, τ-N, HT/RT/Meff, sphericity, dPhi\*, etc. come from identical code,
so a scan's `--obs`/`DEFAULT_SCAN` selection means the same thing for
either binary. Physics setup (`setupPythia()` and friends) is an
intentional line-for-line copy rather than a shared `#include`, so each
binary still builds from a single translation unit.

**What's different.** Truth particles are converted to Delphes candidates
via `fillDelphesInput()` (see "Implementation notes" below for the
particle-conversion details), run through the particle-level-only Delphes
card (`src/generate_events/svj_delphes_particles.tcl`), then re-adapted
from `EFlowMerger` candidates back into the shared `SvjJetInputs` form via
`buildDelphesInputs()`. Since there's no truth-level invisible-particle
info once Delphes reconstruction runs, `invis_ptcls` is always empty for
this binary -- `nInvClose`, `fInv`, and `closeJetIsLead` (which need
geometric matching to invisible particles) always evaluate to `0`; every
other observable, including `dPhiMETclose`/`dPhiMETfar` (computed from
jets + MET only, not invisibles), is a genuine Delphes-level value.

**Single-threaded by design.** Delphes' `MomentumSmearing.cc` and
`SimpleCalorimeter.cc` call the process-wide `gRandom` singleton directly
per-event, with no thread-safety wrapper -- this rules out in-process
multithreading for any Delphes-enabled binary. All scan-level parallelism
comes from `scan_svj.py`'s outer `ProcessPoolExecutor` (`nWorkers` is not
read by this binary; `n_outer_workers` does all the work), matching the
existing recommendation for `svj_regression` (`nWorkers=1`, "outer loop
handles parallelism").

**Cfg keys** (`src/generate_events/svj_regression_delphes.cfg`): the same
physics keys as `svj_regression.cfg` (`mZ`, `LambdaDQCD`, `mPi`, `mRho`,
`mq`, `rinv_pion`, `rinv_rho`, `Brmu`, `alphaD`, `nEvent`, `jetR`), plus
`delphes_card` (path to the Tcl card, default
`src/generate_events/svj_delphes_particles.tcl`), `jets_vis_only`,
`dijet_only`, `vis_jet_pt_min`, and `seed_offset` (no `nWorkers` key, since
the binary is always single-threaded). `scan_svj.py`'s per-point temp cfg
doesn't set these last few keys, so scans always use this binary's compiled
defaults for them -- the same defaults `svj_regression` uses, so truth and
Delphes scans stay directly comparable.

```bash
source ../root/bin/thisroot.sh   # each new shell
make svj_regression_delphes

src/generate_events/svj_regression_delphes \
    src/generate_events/svj_regression_delphes.cfg
# -> simulated/tsv/jets_delphes.tsv (plain OBS_NAMES-header TSV)
```

To use it in a scan or batch TSV generation instead of one-off runs, see
`docs/running-a-scan.md`'s "Truth vs. Delphes" and "Batch TSV generation"
sections.

## Diagnostic driver: `svj_delphes_test`

A single-threaded driver for investigating *how* Delphes reconstruction
changes a given event, one physics point at a time. Unlike
`svj_regression_delphes`, it computes truth-level **and** Delphes-level
observables from the same Pythia event in the same loop iteration, and
writes both to one combined TSV row (`<obs>_truth` / `<obs>_delphes`
columns) -- it does not produce a separate truth-only TSV to diff against
`svj_regression`'s output.

**Why pairing, not two separate TSVs.** `svj_regression` and a hypothetical
separate Delphes-only run can't be compared row-by-row: a multi-worker
truth run interleaves many different Pythia seeds, so only a small
fraction of its rows would even share a seed with a single-threaded
Delphes run -- and independently of that, both binaries skip events with
zero passing jets, at different rates, so even a matched-seed pair would
drift out of row alignment as soon as detector smearing pushes one event's
jet count across the threshold differently than the other. Computing both
from the same event, in the same loop, sidesteps this by construction: no
seed-matching, no discard-desync risk.

One consequence: this binary's own truth-side numbers are not a substitute
for a full, high-statistics `svj_regression` run (it shares
`svj_delphes_test.cfg`'s single `nEvent`, runs single-threaded, at whatever
sample size Delphes' per-event cost makes practical). Use `svj_regression`
directly for statistics; use the paired TSV specifically to study
per-event Delphes effects.

### Build & run

```bash
source ../root/bin/thisroot.sh   # each new shell
make svj_delphes_test

src/generate_events/svj_delphes_test src/generate_events/svj_delphes_test.cfg
# -> simulated/tsv/jets_delphes_paired.tsv

python src/generate_events/compare_delphes_truth.py
```

`compare_delphes_truth.py` produces two figures (marginal truth-vs-Delphes
overlay, and per-event `delphes - truth` differences) plus, per
observable, a printed "zero-jet contingency" breakdown (`both-ok` /
`truth-only-zero` / `delphes-only-zero` / `both-zero`) -- a direct,
per-event answer to whether jets are migrating across `vis_jet_pt_min`
under Delphes reconstruction.

### Investigating specific events (lego plots)

To see *why* a particular event's jet observable changed under Delphes --
e.g. the largest leadVisPt outliers -- render its leading jet's
constituents in eta-phi, truth vs Delphes side by side:

```bash
# 1. Find the 10 events where Delphes adds the most leadVisPt:
python src/generate_events/compare_delphes_truth.py --top-diff 10
# prints a ranked table plus a ready-to-paste "dump_events = ..." line

# 2. Paste that line into svj_delphes_test.cfg's dump_events key, then
#    re-run svj_delphes_test. The fixed Pythia seed (see file header) means
#    the same event indices regenerate the exact same events -- this time
#    their leading-jet constituents are additionally written to
#    jets_delphes_constituents_truth.tsv / _delphes.tsv (one row per
#    constituent: eventIndex, eta, phi, pt). A normal run with dump_events
#    left empty never touches these two files.
make svj_delphes_test   # only if svj_delphes_test.cc changed since the last build
src/generate_events/svj_delphes_test src/generate_events/svj_delphes_test.cfg

# 3. Render the lego plots (one figure per event, truth left / Delphes right,
#    bar height = constituent pT; phi is recentered on each event's own
#    highest-pT constituent so a jet straddling +-pi isn't split across the
#    plot edges):
python src/generate_events/plot_lego.py
# or, to save PNGs instead of showing interactively:
python src/generate_events/plot_lego.py --out-dir simulated/tsv/lego
```

`--top-diff` also accepts `--top-diff-obs` (any of `leadVisPt`, `MET`,
`leadJetMass`, `nConst`, `nJets`; default `leadVisPt`) and `--top-diff-dir`
(`high` = Delphes adds the most, `low` = Delphes subtracts the most, `abs`
= largest magnitude either way; default `high`).

## Verification

1. Both binaries build cleanly against the ROOT/Delphes dependencies above.
2. `svj_delphes_test` runs to completion and produces a non-empty
   `jets_delphes_paired.tsv` with header
   `# leadVisPt_truth\tMET_truth\tleadJetMass_truth\tnConst_truth\tnJets_truth\t`
   `leadVisPt_delphes\tMET_delphes\tleadJetMass_delphes\tnConst_delphes\tnJets_delphes`.
3. `compare_delphes_truth.py` shows the expected sanity signature:
   `leadVisPt` and `leadJetMass` broadened/shifted relative to truth
   (resolution), `MET` showing a resolution tail truth doesn't have, and
   `nConst` systematically *lower* under Delphes (particle-flow
   granularity loses constituents relative to infinite-resolution truth).
   No observable should be empty or wildly discontinuous -- that would
   flag a bug in the particle-conversion step. The per-event difference
   panels and the zero-jet contingency counts should be internally
   consistent with the marginal-overlay panels (e.g. a `leadVisPt` diff
   distribution centered above zero should accompany a marginal
   `leadVisPt` overlay where Delphes sits to the right of truth).
4. `svj_regression_delphes` runs to completion and produces a non-empty,
   28-column TSV that `observables.py`'s `load_tsv()` reads without
   modification.
5. `pytest tests/ -v` (from the project root) passes -- neither binary
   touches `observables.py`, `diagnostics.py`, or `svj_explorer.py`.

## Implementation notes

Non-obvious details worth knowing if you're extending this further:

- **Particle conversion.** `fillDelphesInput()` (in both binaries) is
  ported from Delphes 3.5.1's own `readers/DelphesPythia8.cpp`
  (`ConvertInput()`) rather than the class-based `DelphesPythia8Reader`
  API that exists on Delphes' `master` branch today (a later refactor) --
  the two APIs differ, so match against the installed 3.5.1 source, not
  upstream docs. Dark pions/rhos (PID 51/53) are explicitly excluded
  before Delphes ever sees them: ROOT's `TDatabasePDG::GetParticle()`
  returns a spurious match for these codes (a collision with built-in
  "technicolor pion" placeholders), so relying on a null return to filter
  them out is incorrect.

- **`InitTask()`/`ProcessTask()`/`FinishTask()`, not
  `Init()`/`Process()`/`Finish()`.** The real per-module reconstruction
  logic (tracking efficiency, smearing, calorimeter simulation, EFlow
  merging) only executes via the Task-suffixed trio. `Delphes::Process()`
  is a literal no-op, and `Delphes::Init()` only builds the module list
  from the `.tcl` card's `ExecutionPath` -- it never runs a module itself.
  The Task-suffixed trio (via ROOT's `TTask::ExecuteTask()`) is what
  recurses into every configured module and runs its real `Process()`.
  Calling the non-Task methods compiles and runs with no error, but
  silently skips all reconstruction -- a much worse failure mode than a
  build error.

- **TBB ABI mismatch** (see the build troubleshooting note above) --
  ROOT's binary distribution needs classic TBB (`libtbb.so.2`), available
  via `apt install libtbb2 libtbb-dev` on Ubuntu 20.04, not a newer
  conda-forge `tbb` package (oneTBB, which removed the API this ROOT build
  expects).

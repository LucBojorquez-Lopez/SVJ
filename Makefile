# SVJ event-generator build
# =========================
# Builds src/generate_events/svj_regression from svj_regression.cc against
# PYTHIA8 and FastJet.
#
#   make svj_regression           build the truth-level generator
#   make svj_regression_delphes   build the Delphes-level generator
#   make svj_delphes_test         build the truth-vs-Delphes comparison tool
#   make delphes                  both of the above
#   make clean                    remove the built binaries
#   make check-deps               verify PYTHIA/FastJet can be found, build nothing
#   make check-delphes-deps       verify ROOT/Delphes can be found, build nothing
#
# The *_delphes targets additionally need ROOT and a built Delphes tree; see
# docs/setup_delphes.md (local) or docs/lxplus.md §2 (lxplus/EOS).  They are
# entirely optional — `make svj_regression` works with no ROOT installed.
#
# PYTHIA_DIR and FASTJET_DIR default to siblings of this repository, matching
# the layout in docs/setup.md.  Override either from the command line or the
# environment when your installs live elsewhere (e.g. on lxplus):
#
#   make svj_regression PYTHIA_DIR=/afs/cern.ch/work/u/user/pythia8317 \
#                       FASTJET_DIR=/afs/cern.ch/work/u/user/fastjet3
#
# Both are resolved to ABSOLUTE paths before being baked into the binary's
# rpath, so the resulting executable runs from any working directory.  (A
# relative rpath resolves against the current directory, not the binary, and
# silently breaks the moment you cd somewhere else.)

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

PYTHIA_DIR  ?= $(REPO_ROOT)/../pythia8317
FASTJET_DIR ?= $(REPO_ROOT)/../fastjet3

PYTHIA_ABS     := $(abspath $(PYTHIA_DIR))
FASTJET_ABS    := $(abspath $(FASTJET_DIR))
FASTJET_CONFIG := $(FASTJET_ABS)/bin/fastjet-config

# Supplies CXX, CXX_COMMON, OBJ_COMMON and GZIP_LIB when present.  Optional:
# the explicit flags below are sufficient on their own.
-include $(PYTHIA_ABS)/examples/Makefile.inc

CXX_COMMON += -std=c++17 -pthread
CXX_COMMON := $(OBJ_COMMON) -I$(PYTHIA_ABS)/include $(CXX_COMMON) $(GZIP_LIB)
CXX_COMMON += -L$(PYTHIA_ABS)/lib -Wl,-rpath,$(PYTHIA_ABS)/lib -lpythia8 -ldl

# Deferred (=, not :=) so fastjet-config is only invoked when a build actually
# runs — after check-deps has had a chance to print a readable error.
FASTJET_FLAGS = $(shell $(FASTJET_CONFIG) --cxxflags --libs)

GEN    := $(REPO_ROOT)/src/generate_events
TARGET := $(GEN)/svj_regression
SOURCE := $(GEN)/svj_regression.cc

# Shared observable code, #included by svj_regression.cc and by the Delphes
# generator.  Listed as a prerequisite so editing it triggers a rebuild --
# without this, a change to an observable silently leaves a stale binary.
COMMON_H := $(GEN)/svj_observables_common.h

.PHONY: svj_regression clean check-deps
svj_regression: $(TARGET)

$(TARGET): $(SOURCE) $(COMMON_H) | check-deps
	$(CXX) $< -o $@ $(CXX_COMMON) $(FASTJET_FLAGS) -lstdc++fs

check-deps:
	@test -f "$(PYTHIA_ABS)/lib/libpythia8.so" || { \
	    echo "ERROR: libpythia8.so not found under PYTHIA_DIR=$(PYTHIA_ABS)"; \
	    echo "       Build PYTHIA 8.317 there (see docs/setup.md), or pass"; \
	    echo "       PYTHIA_DIR=/path/to/pythia8317"; \
	    exit 1; }
	@test -x "$(FASTJET_CONFIG)" || { \
	    echo "ERROR: fastjet-config not found at $(FASTJET_CONFIG)"; \
	    echo "       Install FastJet there (see docs/setup.md), or pass"; \
	    echo "       FASTJET_DIR=/path/to/fastjet3"; \
	    exit 1; }

# ── Delphes / ROOT ───────────────────────────────────────────────────────────
#
# Only the *_delphes targets need these, so everything here is deferred and no
# ROOT lookup happens during a plain `make svj_regression`.
#
# root-config is taken from PATH first -- that is how the LCG view provides it
# on lxplus -- and falls back to a sibling source checkout on a local box.
#
# Delphes has NO `make install`: it builds libDelphes.so inside its own source
# tree.  DELPHES_DIR therefore points at that tree, not at an install prefix.
DELPHES_DIR ?= $(REPO_ROOT)/../delphes3.5.1
DELPHES_ABS := $(abspath $(DELPHES_DIR))
ROOT_CONFIG_FALLBACK := $(REPO_ROOT)/../root/bin/root-config

# Tested for EMPTINESS, not definedness, and deliberately not with `?=`.
# PYTHIA's examples/Makefile.inc -- -included above whenever PYTHIA came from a
# source build, which is the normal case on lxplus -- sets `ROOT_CONFIG=` when
# PYTHIA was configured without ROOT.  That counts as "already defined", so a
# `?=` here silently never fires and the delphes targets die on an empty
# root-config path.  A command-line ROOT_CONFIG=... still wins: command-line
# variables override file assignments, and a non-empty one skips this block.
#
# Kept on ONE line: a backslash-newline inside $(shell ...) reaches the shell as
# backslash-space, which escapes the space and breaks the `||` fallback.
ifeq ($(strip $(ROOT_CONFIG)),)
ROOT_CONFIG := $(shell command -v root-config 2>/dev/null || echo $(ROOT_CONFIG_FALLBACK))
endif

# -lEG is NOT included in `root-config --libs`, but Delphes needs it for
# TDatabasePDG.  Delphes' own makefile adds it for the same reason.
ROOT_FLAGS    = $(shell $(ROOT_CONFIG) --cflags --libs) -lEG
DELPHES_FLAGS = -I$(DELPHES_ABS) -I$(DELPHES_ABS)/external \
                -L$(DELPHES_ABS) -Wl,-rpath,$(DELPHES_ABS) -lDelphes

DELPHES_GEN  := $(GEN)/svj_regression_delphes
DELPHES_TEST := $(GEN)/svj_delphes_test

.PHONY: svj_regression_delphes svj_delphes_test delphes check-delphes-deps
svj_regression_delphes: $(DELPHES_GEN)
svj_delphes_test:       $(DELPHES_TEST)
delphes: $(DELPHES_GEN) $(DELPHES_TEST)

# svj_regression_delphes.cc shares the observable code; svj_delphes_test.cc
# does not include it, so it is not a prerequisite there.
$(DELPHES_GEN): $(GEN)/svj_regression_delphes.cc $(COMMON_H) | check-deps check-delphes-deps
	$(CXX) $< -o $@ $(CXX_COMMON) $(FASTJET_FLAGS) $(DELPHES_FLAGS) $(ROOT_FLAGS) -lstdc++fs

$(DELPHES_TEST): $(GEN)/svj_delphes_test.cc | check-deps check-delphes-deps
	$(CXX) $< -o $@ $(CXX_COMMON) $(FASTJET_FLAGS) $(DELPHES_FLAGS) $(ROOT_FLAGS) -lstdc++fs

check-delphes-deps:
	@test -f "$(DELPHES_ABS)/libDelphes.so" || { \
	    echo "ERROR: libDelphes.so not found under DELPHES_DIR=$(DELPHES_ABS)"; \
	    echo "       Build Delphes there, or pass DELPHES_DIR=/path/to/delphes."; \
	    echo "       lxplus/EOS: bash tools/build_deps.sh  (docs/lxplus.md 2)"; \
	    echo "       local:      docs/setup_delphes.md"; \
	    exit 1; }
	@command -v "$(ROOT_CONFIG)" >/dev/null 2>&1 || test -x "$(ROOT_CONFIG)" || { \
	    echo "ERROR: root-config not found (tried: $(ROOT_CONFIG))"; \
	    echo "       lxplus/EOS: source setup_env.sh -- the LCG view supplies ROOT."; \
	    echo "       local:      install ROOT, or pass ROOT_CONFIG=/path/to/root-config"; \
	    exit 1; }

clean:
	rm -f $(TARGET) $(DELPHES_GEN) $(DELPHES_TEST)

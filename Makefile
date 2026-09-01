# SVJ event-generator build
# =========================
# Builds src/generate_events/svj_regression from svj_regression.cc against
# PYTHIA8 and FastJet.
#
#   make svj_regression      build the generator
#   make clean               remove the built binary
#   make check-deps          verify PYTHIA/FastJet can be found, build nothing
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

TARGET := $(REPO_ROOT)/src/generate_events/svj_regression
SOURCE := $(REPO_ROOT)/src/generate_events/svj_regression.cc

.PHONY: svj_regression clean check-deps
svj_regression: $(TARGET)

$(TARGET): $(SOURCE) | check-deps
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

clean:
	rm -f $(TARGET)

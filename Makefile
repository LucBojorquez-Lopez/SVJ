PYTHIA_DIR  = ../pythia8317
FASTJET_DIR = ../fastjet3
-include $(PYTHIA_DIR)/examples/Makefile.inc

CXX_COMMON += -std=c++17 -pthread

CXX_COMMON := $(OBJ_COMMON) -I$(PYTHIA_DIR)/include $(CXX_COMMON) $(GZIP_LIB)
CXX_COMMON += -L$(PYTHIA_DIR)/lib -Wl,-rpath,$(PYTHIA_DIR)/lib -lpythia8 -ldl
CXX_COMMON += $(shell $(FASTJET_DIR)/bin/fastjet-config --cxxflags --libs)

%: %.cc
	$(CXX) $< -o $@ $(CXX_COMMON)

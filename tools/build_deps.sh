#!/bin/bash
# tools/build_deps.sh — build PYTHIA 8.317 and FastJet 3.5.1 into $SVJ_WORK.
#
#   source setup_env.sh
#   bash tools/build_deps.sh
#
# Once these exist, setup_env.sh prefers them over the LCG view automatically
# (it reports which via $SVJ_DEPS).  Re-source it after this finishes.
#
# ── Two deliberate choices ──────────────────────────────────────────────────
#
# 1. Compiled with the SYSTEM gcc, not the LCG view's.  A binary linked against
#    these then depends on nothing in CVMFS, so it survives an LCG view being
#    retired.  The repo Makefile picks the same compiler up automatically: it
#    `-include`s $PYTHIA_DIR/examples/Makefile.inc, which a source build writes
#    and a CVMFS view does not ship.
#
# 2. Built on local disk, then copied to EOS.  A direct build on EOS means many
#    thousands of small compile-and-link writes, which is the one workload EOS
#    is genuinely bad at.  Copying a finished tree is what it is good at.
#
#    Both are nevertheless configured with their FINAL EOS prefix, never the
#    temporary build path.  Autotools and PYTHIA both bake the configured
#    prefix into their output — fastjet-config reports it, the .la files record
#    it, and libpythia8.so gets it as a compiled-in ELF RPATH.  Configure with
#    the build directory and copy afterwards and you get a library whose RPATH
#    points into /tmp: fine on the machine that built it, broken on every
#    worker node.  FastJet is staged via DESTDIR so the EOS write stays a
#    single bulk copy; PYTHIA builds in place, so its tree is copied directly.

set -euo pipefail

PYTHIA_VER=8317
FASTJET_VER=3.5.1
JOBS="${JOBS:-4}"

DEST="${SVJ_WORK:?source setup_env.sh first}"
BUILD="${SVJ_BUILD_TMP:-${TMPDIR:-/tmp}/svj-deps-build.$$}"

# The system compiler, explicitly — an LCG view is normally on PATH by now.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
command -v "$CXX" >/dev/null || { echo "no $CXX on this machine" >&2; exit 1; }

mkdir -p "$DEST"
echo "### compiler : $($CXX --version | head -1)"
echo "### build dir: $BUILD"
echo "### install  : $DEST"

# Tarballs are cached in $DEST so a rebuild does not re-download 34 MB.
# Note the path is /releases/, not the /download/ that older docs give — that
# one now 404s.
fetch() {
    local url="$1" out="$2"
    [[ -s "$DEST/$out" ]] && { echo "### cached: $out"; return; }
    echo "### fetching $out"
    wget -q -O "$DEST/$out.part" "$url"
    mv "$DEST/$out.part" "$DEST/$out"
}
fetch "https://pythia.org/releases/pythia83/pythia${PYTHIA_VER}.tgz" "pythia${PYTHIA_VER}.tgz"
fetch "http://fastjet.fr/repo/fastjet-${FASTJET_VER}.tar.gz"          "fastjet-${FASTJET_VER}.tar.gz"

trap 'rm -rf "$BUILD"' EXIT
rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"

echo "### extracting"
tar xzf "$DEST/fastjet-${FASTJET_VER}.tar.gz"
tar xzf "$DEST/pythia${PYTHIA_VER}.tgz"

# Logs go beside the build, not inside the source trees, so they are not
# copied into the install.
echo "### building FastJet ${FASTJET_VER}"
cd "$BUILD/fastjet-${FASTJET_VER}"
./configure --prefix="$DEST/fastjet3" >"$BUILD/fastjet-configure.log" 2>&1
make -j"$JOBS"                        >"$BUILD/fastjet-make.log"      2>&1
make install DESTDIR="$BUILD/stage"   >"$BUILD/fastjet-install.log"   2>&1

echo "### building PYTHIA ${PYTHIA_VER}  (the long one, 15-30 min)"
cd "$BUILD/pythia${PYTHIA_VER}"
./configure --prefix="$DEST/pythia${PYTHIA_VER}" >"$BUILD/pythia-configure.log" 2>&1
make -j"$JOBS"                                   >"$BUILD/pythia-make.log"      2>&1

# Object files: ~1 GB, needed neither to link against nor to run.
rm -rf "$BUILD/pythia${PYTHIA_VER}/tmp"

echo "### installing to $DEST"
rm -rf "$DEST/pythia${PYTHIA_VER}" "$DEST/fastjet3"
cp -a "$BUILD/stage$DEST/fastjet3" "$DEST/fastjet3"
cp -a "$BUILD/pythia${PYTHIA_VER}" "$DEST/pythia${PYTHIA_VER}"

# The whole point of configuring with the final prefix — verify it took.
#
# Checked precisely rather than with a recursive grep: compiled objects legitimately
# record the build directory in DWARF (.debug_line_str) and that is cosmetic, so a
# blanket grep reports failures that do not matter and trains you to ignore it.
# What actually breaks a worker node is an RPATH or a wrong path in the config
# scripts the Makefile consults.
echo "### verifying no build-directory paths survived where they matter"
_bad=0

for _so in "$DEST/fastjet3"/lib/*.so "$DEST/pythia${PYTHIA_VER}"/lib/*.so; do
    [[ -e "$_so" ]] || continue
    if readelf -d "$_so" 2>/dev/null | grep -E "RPATH|RUNPATH" | grep -qF "$BUILD"; then
        echo "ERROR: $(basename "$_so") has an RPATH into the build dir" >&2
        _bad=1
    fi
done

for _txt in "$DEST/fastjet3/bin/fastjet-config" \
            "$DEST/pythia${PYTHIA_VER}/bin/pythia8-config" \
            "$DEST/pythia${PYTHIA_VER}/Makefile.inc" \
            "$DEST/pythia${PYTHIA_VER}/examples/Makefile.inc" \
            "$DEST/fastjet3"/lib/*.la; do
    [[ -f "$_txt" ]] || continue
    if grep -qF "$BUILD" "$_txt"; then
        echo "ERROR: $_txt still points at the build dir" >&2
        _bad=1
    fi
done

[[ "$_bad" -eq 0 ]] || { echo "install is not relocatable; not usable on a worker node" >&2; exit 1; }
echo "### clean: RPATHs and config scripts all point at $DEST"

echo "### done:"
"$DEST/fastjet3/bin/fastjet-config" --version
grep -m1 'define PYTHIA_VERSION ' "$DEST/pythia${PYTHIA_VER}/include/Pythia8/Pythia.h"
echo
echo "Now re-source setup_env.sh and rebuild:  make clean && make svj_regression"

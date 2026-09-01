# shellcheck shell=bash
# setup_env.sh — SVJ environment on CERN lxplus (EL9).
#
# This file is SOURCED, never executed — it has no shebang and is not +x.
#
# Source this in every interactive shell and every batch job:
#
#     source /eos/user/l/lbojorqu/SVJ/setup_env.sh
#
# condor/svj_job.sh sources it automatically via $SVJ_ENV.
#
# Python, and the compiler used for anything not covered below, come from a
# single LCG view on CVMFS — this account has no /afs/cern.ch/work volume and
# ~1 GB free in its 2 GB AFS home, so a self-built Python stack is not on.
#
# PYTHIA and FastJet are our own builds under $SVJ_WORK on EOS, compiled with
# the system gcc, with the view as a fallback.  See docs/lxplus.md §2.

LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt}"

if [[ ! -f "$LCG_VIEW/setup.sh" ]]; then
    echo "setup_env.sh: no LCG view at $LCG_VIEW" >&2
    echo "              pick another from /cvmfs/sft.cern.ch/lcg/views/ and" >&2
    echo "              re-source with LCG_VIEW=... (needs pythia8 + fastjet)" >&2
    return 1 2>/dev/null || exit 1
fi

# The view's setup.sh reads $COMPILER unguarded, so it aborts under `set -u`.
# condor/svj_job.sh runs `set -euo pipefail`, so shield the source and restore
# the caller's flags afterwards — otherwise every batch job dies right here.
_svj_flags="$-"
set +u
# shellcheck source=/dev/null
source "$LCG_VIEW/setup.sh"
case "$_svj_flags" in *u*) set -u ;; esac
unset _svj_flags

# This file lives at the repository root, so its own directory is the repo.
SVJ_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SVJ_REPO

# ── PYTHIA and FastJet ───────────────────────────────────────────────────────
#
# The Makefile wants a PYTHIA prefix with lib/libpythia8.so and a FastJet
# prefix with bin/fastjet-config.  Two sources can provide those:
#
#   1. Our own builds under $SVJ_WORK, compiled with the SYSTEM gcc.  Preferred.
#      A binary built against these depends on nothing from CVMFS, so it keeps
#      working when an LCG view is retired — and PYTHIA is ours to patch.
#   2. The LCG view, which happens to ship exactly 8.317 and 3.5.1.  Fallback,
#      so a missing or half-finished local build degrades instead of blocking.
#
# Build the local copies with tools/build_deps.sh.
export SVJ_WORK="${SVJ_WORK:-$(dirname "$SVJ_REPO")/svj-work}"

# Globbed rather than pinned to pythia8317, so bumping PYTHIA_VER in
# tools/build_deps.sh does not silently drop you back to the view.  The glob
# sorts, so the last match is the highest version present.
_svj_pythia=""
for _svj_d in "$SVJ_WORK"/pythia8*/; do
    [[ -f "$_svj_d/lib/libpythia8.so" ]] && _svj_pythia="${_svj_d%/}"
done

if [[ -n "$_svj_pythia" && -x "$SVJ_WORK/fastjet3/bin/fastjet-config" ]]; then
    export PYTHIA_DIR="$_svj_pythia"
    export FASTJET_DIR="$SVJ_WORK/fastjet3"
    # Must override the view's value: PYTHIA reads its settings from whichever
    # xmldoc this points at, and silently using another install's is a subtle
    # way to run a different physics configuration than you think.
    export PYTHIA8DATA="$PYTHIA_DIR/share/Pythia8/xmldoc"
    export SVJ_DEPS=local
    # No rpath needed: the Makefile picks up CXX=/usr/bin/g++ from this
    # PYTHIA's examples/Makefile.inc, so the binary links the system libstdc++
    # that is already present on every worker node.
    unset GZIP_LIB
else
    export PYTHIA_DIR="$LCG_VIEW"
    export FASTJET_DIR="$LCG_VIEW"
    export SVJ_DEPS=lcg
    # A view install ships no examples/Makefile.inc, so the Makefile takes CXX
    # from PATH — the view's gcc 13 — and GZIP_LIB stays free for us to use as
    # an extra rpath.  Without it the binary resolves libstdc++ against EL9's
    # gcc-11-era system copy and dies with `GLIBCXX_3.4.31 not found`.
    SVJ_GCC_LIB="$(readlink -f "$(dirname "$(g++ -print-file-name=libstdc++.so)")" 2>/dev/null)"
    if [[ -n "$SVJ_GCC_LIB" && -d "$SVJ_GCC_LIB" ]]; then
        export GZIP_LIB="-Wl,-rpath,$SVJ_GCC_LIB"
    fi
    unset SVJ_GCC_LIB
fi
unset _svj_pythia _svj_d

# ── Delphes ──────────────────────────────────────────────────────────────────
#
# Optional.  Only src/generate_events/svj_regression_delphes and
# svj_delphes_test need it; nothing on the Python side imports it.  A missing
# Delphes therefore degrades to "those two make targets refuse to build"
# instead of breaking the environment, which is why this block never returns
# non-zero.
#
# Delphes has NO install prefix -- `make` leaves libDelphes.so inside its own
# source tree and ships no `make install` -- so DELPHES_DIR points at a source
# tree, not at a prefix like PYTHIA_DIR and FASTJET_DIR do.
#
# Globbed rather than pinned, for the same reason as PYTHIA above: bumping
# DELPHES_VER in tools/build_deps.sh must not silently leave this unset.
#
# ROOT is a hard requirement of Delphes and comes from the LCG view sourced
# above -- there is no local-build fallback for it, and none is wanted: ROOT is
# far too big to build here against a 2 GB AFS quota.  The Makefile finds it via
# `root-config` on PATH, which the view provides.
_svj_delphes=""
for _svj_d in "$SVJ_WORK"/delphes*/; do
    [[ -f "$_svj_d/libDelphes.so" ]] && _svj_delphes="${_svj_d%/}"
done

if [[ -n "$_svj_delphes" ]] && command -v root-config >/dev/null 2>&1; then
    export DELPHES_DIR="$_svj_delphes"
    export SVJ_DELPHES=local
elif [[ -n "$_svj_delphes" ]]; then
    export DELPHES_DIR="$_svj_delphes"
    export SVJ_DELPHES=no-root      # tree present, but the view has no ROOT
else
    unset DELPHES_DIR
    export SVJ_DELPHES=none         # run tools/build_deps.sh --delphes
fi
unset _svj_delphes _svj_d

# An autoloaded pytest plugin in the view emits CDash <DartMeasurement> XML for
# every test, burying the summary line in thousands of lines of markup.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# ── The AFS side of Condor ───────────────────────────────────────────────────
#
# CERN's batch schedds reject a submit file whose executable/log/output/error
# sits on /eos, and the EosSubmit schedd that accepts those then rejects
# everything NOT on /eos — plus its transferred-back .out/.err are unreadable.
# So a small amount has to live on AFS: the launcher Condor executes, and the
# job logs.  Everything substantial stays on EOS.  See condor/launch.sh.
#
# AFS home is a 2 GB quota, so keep only logs here and prune them periodically.
export SVJ_AFS="${SVJ_AFS:-$HOME/.svj}"
mkdir -p "$SVJ_AFS/logs"

# Keep the AFS launcher in sync with the tracked copy, so condor/launch.sh is
# edited in git rather than in an untracked file nobody remembers exists.
if [[ -f "$SVJ_REPO/condor/launch.sh" ]] &&
   ! cmp -s "$SVJ_REPO/condor/launch.sh" "$SVJ_AFS/launch.sh"; then
    install -m 755 "$SVJ_REPO/condor/launch.sh" "$SVJ_AFS/launch.sh"
    echo "setup_env.sh: refreshed $SVJ_AFS/launch.sh from the repo"
fi

# Grid proxy for EOS/XRootD access from Condor worker nodes.  The .sub files
# reference this via $ENV(X509_USER_PROXY), so it must be exported before
# condor_submit.  Refresh it with:
#
#     voms-proxy-init -voms atlas -out "$X509_USER_PROXY" -valid 168:00
#
# The proxy may live on AFS even when everything else is on EOS — Condor treats
# it as a credential reference, not a transferred job file.
export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/private/x509up}"

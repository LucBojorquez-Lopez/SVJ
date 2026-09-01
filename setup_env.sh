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
# Everything the project needs — g++ 13, Python 3.13, numpy/scipy/matplotlib,
# PYTHIA 8.317 and FastJet 3.5.1 — comes from a single LCG view on CVMFS.
# Nothing is built into AFS or EOS, which matters here: this account has no
# /afs/cern.ch/work volume and only ~1 GB free in its 2 GB AFS home, whereas
# source-building PYTHIA + FastJet needs several GB.  See docs/lxplus.md.

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

# The Makefile wants a PYTHIA prefix with lib/libpythia8.so and a FastJet
# prefix with bin/fastjet-config.  The view is a single prefix providing both.
export PYTHIA_DIR="$LCG_VIEW"
export FASTJET_DIR="$LCG_VIEW"

# Extra rpath baked into the binary at link time.  GZIP_LIB is otherwise only
# set by PYTHIA's own examples/Makefile.inc, which a view install does not
# ship, so the Makefile appends this to the link line untouched — no Makefile
# edit needed.  Without it the binary resolves libstdc++ against EL9's system
# copy, which is gcc-11-era and fails with `GLIBCXX_3.4.31 not found`.
SVJ_GCC_LIB="$(readlink -f "$(dirname "$(g++ -print-file-name=libstdc++.so)")" 2>/dev/null)"
if [[ -n "$SVJ_GCC_LIB" && -d "$SVJ_GCC_LIB" ]]; then
    export GZIP_LIB="-Wl,-rpath,$SVJ_GCC_LIB"
fi
unset SVJ_GCC_LIB

# This file lives at the repository root, so its own directory is the repo.
SVJ_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SVJ_REPO

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

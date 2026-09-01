#!/bin/bash
# condor/launch.sh — the AFS-side entry point for every SVJ Condor job.
#
# This exists for one reason: CERN's batch schedds will not accept a submit
# file whose executable, log, output or error path is on /eos.  The EosSubmit
# schedd (`module load lxbatch/eossubmit`) does accept those, but then refuses
# any path that is NOT on /eos — the two cannot be mixed — and job stdout/stderr
# transferred back to EOS proved unreadable: the files land in the namespace but
# stat as `-??????????` indefinitely, which makes an array impossible to debug.
#
# So the executable and the logs live on AFS, and everything else — the
# repository, the configs, the NPZs, the bulk data — stays on EOS.  Condor
# transfers this script to the worker's scratch directory, so the worker never
# needs to read AFS at all; it only needs EOS, which it reaches with the
# forwarded x509 proxy.
#
# `setup_env.sh` installs a copy of this file at $SVJ_AFS/launch.sh and keeps it
# in sync, so edit it HERE, in git, and re-source setup_env.sh.
#
# $SVJ_REPO is supplied by the .sub file's `environment` line; without it this
# script cannot find the repository, because its own location is a scratch copy.

set -euo pipefail

if [[ -z "${SVJ_REPO:-}" ]]; then
    echo "ERROR: SVJ_REPO is not set." >&2
    echo "       The .sub file must pass it, e.g." >&2
    echo '       environment = "SVJ_REPO=/eos/user/l/lbojorqu/SVJ"' >&2
    exit 1
fi

if [[ ! -x "$SVJ_REPO/condor/svj_job.sh" ]]; then
    echo "ERROR: no executable wrapper at $SVJ_REPO/condor/svj_job.sh" >&2
    echo "       Is SVJ_REPO right, and can this worker read EOS?" >&2
    echo "       A missing x509 proxy usually shows up exactly here." >&2
    exit 1
fi

exec "$SVJ_REPO/condor/svj_job.sh" "$@"

#!/bin/bash
# condor/svj_job.sh
# Shared HTCondor wrapper for every SVJ workflow.
#
#   Usage (from a .sub file):
#       executable = condor/svj_job.sh
#       arguments  = "<workflow> <n_jobs> $(ProcId)"
#
#   <workflow> is one of: scan | tsv | validation
#
# Edit SVJ_ENV below to point at your own environment script (see docs/lxplus.md
# §2).  That script must source an LCG view, activate the venv, and export
# PYTHIA_DIR / FASTJET_DIR.
#
# NOTE: untested on lxplus.  Smoke-test with a single job first —
# docs/lxplus.md §7.1.

set -euo pipefail

SVJ_ENV="${SVJ_ENV:-$HOME/setup_env.sh}"

WORKFLOW="${1:?usage: svj_job.sh <scan|tsv|validation> <n_jobs> <task_id>}"
N_JOBS="${2:?missing n_jobs}"
TASK="${3:?missing task id}"

# Condor starts the job in its scratch dir; every path below is repo-relative,
# and scan_svj.py resolves output_dir against the CWD, so move to the repo root.
#
# Derived from this script's own location, which is correct under
# `should_transfer_files = NO` (the executable runs from its real path on a
# shared filesystem).  If you enable file transfer, Condor copies this script
# to scratch and that inference breaks — set SVJ_REPO explicitly instead.
REPO_ROOT="${SVJ_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ ! -f "$REPO_ROOT/src/run_regression/scan_svj.py" ]]; then
    echo "ERROR: '$REPO_ROOT' does not look like the SVJ repository." >&2
    echo "       Set SVJ_REPO=/path/to/SVJ (needed if your .sub uses" >&2
    echo "       should_transfer_files = YES)." >&2
    exit 1
fi
cd "$REPO_ROOT"

echo "=== SVJ $WORKFLOW  task $TASK / $((N_JOBS-1))  started $(date) ==="
echo "    host:    $(hostname)"
echo "    repo:    $REPO_ROOT"
echo "    scratch: ${TMPDIR:-/tmp}   (workers write per-point cfg/TSV here)"

# shellcheck source=/dev/null
[[ -f "$SVJ_ENV" ]] && source "$SVJ_ENV" || {
    echo "ERROR: environment script not found: $SVJ_ENV" >&2
    echo "       Set SVJ_ENV, or create it (docs/lxplus.md §2)." >&2
    exit 1; }

echo "    python:  $(python --version 2>&1)"
mkdir -p logs

case "$WORKFLOW" in

  scan)
    python src/run_regression/scan_svj.py \
        src/run_regression/scan_regression.cfg \
        --job-index "$TASK" --n-jobs "$N_JOBS"
    ;;

  tsv)
    BINARY="src/generate_events/svj_regression"
    [[ -x "$BINARY" ]] || { echo "ERROR: run 'make svj_regression' first" >&2; exit 1; }
    # readConfig() takes the LAST occurrence of a duplicated key, so appended
    # lines override the base cfg.
    TMP_CFG="${TMPDIR:-/tmp}/svj_tsv_${TASK}.cfg"
    cat src/generate_events/svj_regression.cfg > "$TMP_CFG"
    cat >> "$TMP_CFG" <<EOCFG

# per-job overrides — seed = workerID + 1 + seed_offset * nWorkers
seed_offset  = ${TASK}
tsv_file     = simulated/tsv/jets_default_${TASK}.tsv
tsv_kin_file = simulated/tsv/jets_kinematics_${TASK}.tsv
save_tsv     = 1
EOCFG
    "$BINARY" "$TMP_CFG"
    rm -f "$TMP_CFG"
    ;;

  validation)
    python src/run_regression/validate_production.py \
        simulated/svj/svj_scan.npz \
        --N1 50000 --N2 20000 --N3 125 \
        --seed "$TASK" --n-workers 16 --n-mmd 2000 \
        --out "simulated/svj/validation_production_${TASK}.npz"
    ;;

  *)
    echo "ERROR: unknown workflow '$WORKFLOW' (want scan|tsv|validation)" >&2
    exit 1
    ;;
esac

echo "=== SVJ $WORKFLOW  task $TASK  finished $(date) ==="

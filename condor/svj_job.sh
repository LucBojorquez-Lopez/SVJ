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
# Environment, via SVJ_ENV — default: the setup_env.sh at the repository root.
# It must source an LCG view and export PYTHIA_DIR / FASTJET_DIR (docs/lxplus.md
# §2).  The default points inside the repo rather than at $HOME because a worker
# node reads an EOS-resident clone reliably, whereas reaching your AFS home from
# a worker is a separate credential question.
#
# SVJ_SCAN_CFG overrides the scan config, so one .sub can run either a one-point
# smoke test (src/run_regression/scan_smoke.cfg) or a production grid.

set -euo pipefail

SVJ_SCAN_CFG="${SVJ_SCAN_CFG:-src/run_regression/scan_regression.cfg}"

WORKFLOW="${1:?usage: svj_job.sh <scan|tsv|validation> <n_jobs> <task_id>}"
N_JOBS="${2:?missing n_jobs}"
TASK="${3:?missing task id}"

# Condor starts the job in its scratch dir; every path below is repo-relative,
# and scan_svj.py resolves output_dir against the CWD, so move to the repo root.
#
# $SVJ_REPO is the reliable source and every .sub sets it.  The $BASH_SOURCE
# fallback only works when the executable really runs from its own path, which
# on lxplus it does NOT: CERN's schedds transfer the executable to the job's
# scratch directory even with `should_transfer_files = NO`, so $BASH_SOURCE
# resolves to /pool/condor/dir_<n>/condor_exec.exe and every path derived from
# it is wrong.
REPO_ROOT="${SVJ_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ ! -f "$REPO_ROOT/src/run_regression/scan_svj.py" ]]; then
    echo "ERROR: '$REPO_ROOT' does not look like the SVJ repository." >&2
    echo "       Set SVJ_REPO=/path/to/SVJ in your .sub's environment line." >&2
    exit 1
fi
cd "$REPO_ROOT"

# Resolved only now, so it follows $SVJ_REPO rather than $BASH_SOURCE.
SVJ_ENV="${SVJ_ENV:-$REPO_ROOT/setup_env.sh}"

echo "=== SVJ $WORKFLOW  task $TASK / $((N_JOBS-1))  started $(date) ==="
echo "    host:    $(hostname)"
echo "    repo:    $REPO_ROOT"
echo "    scratch: ${TMPDIR:-/tmp}   (workers write per-point cfg/TSV here)"

if [[ ! -f "$SVJ_ENV" ]]; then
    echo "ERROR: environment script not found: $SVJ_ENV" >&2
    echo "       Set SVJ_ENV, or create it (docs/lxplus.md §2)." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$SVJ_ENV"

echo "    env:     $SVJ_ENV"
echo "    python:  $(command -v python)  $(python --version 2>&1)"

case "$WORKFLOW" in

  scan)
    echo "    cfg:     $SVJ_SCAN_CFG"
    python src/run_regression/scan_svj.py \
        "$SVJ_SCAN_CFG" \
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

#!/bin/bash
#SBATCH --job-name=svj_scan
#SBATCH --array=0-15                   # 4 parallel slices; must match N_JOBS below
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH -p arguelles_delgado
#SBATCH --mem=5G
#SBATCH -t 0-10:00
#SBATCH --output=logs/svj_%A_%a.out
#SBATCH --error=logs/svj_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luc_bojorquezlopez@college.harvard.edu

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs simulated/svj

source ~/venvs/svj/bin/activate

# ── Resource-usage report helper ──────────────────────────────────────────────
# Parses the output of /usr/bin/time -v and prints a calibration summary.
# Note: for the scan script this covers the Python process; per-point C++ RSS
# is reported by scan_svj.py itself (via resource.getrusage(RUSAGE_CHILDREN)).
# Usage: _print_time_report <time_file> <label>
_print_time_report() {
    local time_file="$1"  label="$2"
    [[ -f "$time_file" ]] || { echo "(no resource file: $time_file)"; return; }

    local rss_kb user_s sys_s wall_raw
    rss_kb=$(  grep 'Maximum resident set size'  "$time_file" | awk '{print $NF}')
    user_s=$(  grep 'User time (seconds)'        "$time_file" | awk '{print $NF}')
    sys_s=$(   grep 'System time (seconds)'      "$time_file" | awk '{print $NF}')
    wall_raw=$(grep 'Elapsed (wall clock) time'  "$time_file" | awk '{print $NF}')

    local wall_s
    if [[ "$wall_raw" =~ ^([0-9]+):([0-9]{2}):([0-9.]+)$ ]]; then
        wall_s=$(echo "${BASH_REMATCH[1]}*3600 + ${BASH_REMATCH[2]}*60 + ${BASH_REMATCH[3]}" | bc)
    else
        wall_s=$(echo "$wall_raw" | awk -F: '{printf "%.2f", $1*60+$2}')
    fi

    local rss_mb suggest_gb cpu_eff
    rss_mb=$(    echo "scale=1; ${rss_kb:-0} / 1024"                                        | bc)
    suggest_gb=$(echo "scale=0; (${rss_kb:-0} * 3 / 2 / 1048576) + 1"                      | bc)
    cpu_eff=$(   echo "scale=1; (${user_s:-0} + ${sys_s:-0}) / \
                       (${wall_s:-1} * ${SLURM_CPUS_ON_NODE:-1}) * 100"                     | bc 2>/dev/null || echo "N/A")

    echo ""
    echo "=== Resource usage: $label ==="
    printf "  Peak RSS (Python main)  : %s MB\n"  "$rss_mb"
    printf "  Wall time               : %s\n"      "$wall_raw"
    printf "  CPU cores alloc         : %s\n"      "${SLURM_CPUS_ON_NODE:-?}"
    printf "  CPU efficiency          : %s%%\n"    "$cpu_eff"
    printf "  → suggest --mem         : %sG\n"     "$suggest_gb"
    echo "  (C++ worker RSS is in the scan_svj.py output above)"
    echo "================================="
    rm -f "$time_file"
}

echo "=== job $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID  started $(date) ==="
echo "    node:   $SLURMD_NODENAME"
echo "    cpus:   $SLURM_CPUS_ON_NODE"
echo "    python: $(python --version)"
echo ""

# Number of array tasks — must match --array=0-(N_JOBS-1) above.
N_JOBS=16

TIME_FILE="/tmp/svj_scan_time_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

/usr/bin/time -v -o "$TIME_FILE" \
    python src/run_regression/scan_svj.py \
        src/run_regression/scan_regression.cfg \
        --job-index "$SLURM_ARRAY_TASK_ID" \
        --n-jobs "$N_JOBS"

_print_time_report "$TIME_FILE" "svj_scan task $SLURM_ARRAY_TASK_ID"

echo ""
echo "=== job $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID  finished $(date) ==="

# ── Merge ─────────────────────────────────────────────────────────────────────
# After ALL array jobs complete, run the merge step once from the login node:
#
#   python src/run_regression/scan_svj.py --merge --n-jobs 4
#
# Or submit it as a dependent job:
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> merge_svj.sh

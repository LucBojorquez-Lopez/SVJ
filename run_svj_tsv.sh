#!/bin/bash
# run_svj_tsv.sh
# SLURM array job: generate N independent TSV shards and merge into
# simulated/tsv/jets_default.tsv.
#
# Each array task runs the full svj_regression binary with the same physics
# parameters but a unique seed_offset, producing statistically independent
# events.  The total event count is N_JOBS * nEvent (from svj_regression.cfg).
#
# Usage
# -----
# 1. Adjust N_JOBS and --array below to match the desired number of shards.
#    Also set -c to match nWorkers in svj_regression.cfg (one CPU per thread).
# 2. Submit the array:
#       sbatch run_svj_tsv.sh
# 3. After all tasks finish, merge the shards:
#       bash merge_svj_tsv.sh <N_JOBS>
#    Or submit a dependent merge job automatically (see bottom of this file).

#SBATCH --job-name=svj_tsv
#SBATCH --array=0-3                   # N_JOBS shards; must match N_JOBS below
#SBATCH --nodes=1
#SBATCH -c 48                         # CPUs per task; should match nWorkers in svj_regression.cfg
#SBATCH -p arguelles_delgado
#SBATCH --mem=16G
#SBATCH -t 0-02:00                    # wall time; adjust to ~2× single-run time
#SBATCH --output=logs/svj_tsv_%A_%a.out
#SBATCH --error=logs/svj_tsv_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luc_bojorquezlopez@college.harvard.edu

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs simulated/tsv

BINARY="src/generate_events/svj_regression"
BASE_CFG="src/generate_events/svj_regression.cfg"

# ── Resource-usage report helper ──────────────────────────────────────────────
# Parses the output of /usr/bin/time -v and prints a calibration summary.
# Usage: _print_time_report <time_file> <label>
_print_time_report() {
    local time_file="$1"  label="$2"
    [[ -f "$time_file" ]] || { echo "(no resource file: $time_file)"; return; }

    local rss_kb user_s sys_s wall_raw
    rss_kb=$(  grep 'Maximum resident set size'  "$time_file" | awk '{print $NF}')
    user_s=$(  grep 'User time (seconds)'        "$time_file" | awk '{print $NF}')
    sys_s=$(   grep 'System time (seconds)'      "$time_file" | awk '{print $NF}')
    wall_raw=$(grep 'Elapsed (wall clock) time'  "$time_file" | awk '{print $NF}')

    # Convert h:mm:ss or m:ss.cs to total seconds for CPU-efficiency calc.
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
    printf "  Peak RSS        : %s MB\n"   "$rss_mb"
    printf "  Wall time       : %s\n"      "$wall_raw"
    printf "  CPU cores alloc : %s\n"      "${SLURM_CPUS_ON_NODE:-?}"
    printf "  CPU efficiency  : %s%%\n"    "$cpu_eff"
    printf "  → suggest --mem : %sG   (peak RSS × 1.5 + headroom)\n"  "$suggest_gb"
    echo "================================="
    rm -f "$time_file"
}

# Must match --array=0-(N_JOBS-1) above.
N_JOBS=4
TASK="$SLURM_ARRAY_TASK_ID"

echo "=== job $SLURM_JOB_ID  array task $TASK / $((N_JOBS-1))  started $(date) ==="
echo "    node:   $SLURMD_NODENAME"
echo "    cpus:   $SLURM_CPUS_ON_NODE"
echo ""

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary '$BINARY' not found or not executable." >&2
    echo "Run 'make svj_regression' from the project root first." >&2
    exit 1
fi

# Build a per-job config by appending overrides to the base cfg.
# readConfig() in the binary uses std::map, so a duplicated key takes the
# value from the LAST occurrence in the file — appended lines win.
TMP_CFG="/tmp/svj_tsv_${SLURM_JOB_ID}_${TASK}.cfg"

cat "$BASE_CFG" > "$TMP_CFG"
cat >> "$TMP_CFG" << EOCFG

# ── per-job overrides appended by run_svj_tsv.sh ─────────────────────────────
# seed_offset shifts every worker's PYTHIA seed:
#   seed = workerID + 1 + seed_offset * nWorkers
# Each array task therefore draws from a disjoint seed range.
seed_offset  = ${TASK}
tsv_file     = simulated/tsv/jets_default_${TASK}.tsv
tsv_kin_file = simulated/tsv/jets_kinematics_${TASK}.tsv
save_tsv     = 1
EOCFG

TIME_FILE="/tmp/svj_tsv_time_${SLURM_JOB_ID}_${TASK}.txt"

echo "Running: $BINARY $TMP_CFG"
/usr/bin/time -v -o "$TIME_FILE" "$BINARY" "$TMP_CFG"
rm -f "$TMP_CFG"

_print_time_report "$TIME_FILE" "svj_tsv task $TASK"

echo ""
echo "=== job $SLURM_JOB_ID  array task $TASK  finished $(date) ==="

# ── Merge ─────────────────────────────────────────────────────────────────────
# After ALL array tasks complete, merge the shards from the login node:
#
#   bash merge_svj_tsv.sh 4          # where 4 = N_JOBS
#
# Or submit the merge as a dependent job at sbatch time:
#
#   ARRAY_JID=$(sbatch --parsable run_svj_tsv.sh)
#   sbatch --dependency=afterok:${ARRAY_JID} \
#          --wrap "bash merge_svj_tsv.sh 4"

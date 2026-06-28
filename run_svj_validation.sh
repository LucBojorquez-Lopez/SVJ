#!/bin/bash
# run_svj_validation.sh
# SLURM array job: full production validation of svj_scan.npz.
#
# Splits N3=2000 validation points across 16 array tasks (125 each).
# Each task uses a unique --seed so it samples a statistically independent
# set of interior grid points.  After all tasks finish, merge the shards:
#
#   python merge_svj_validation.py
#
# or submit as a dependent job (see bottom of this file).
#
# Parameters per task
# -------------------
#   N1  = 50000   model samples drawn from the interpolation
#   N2  = 5000   PYTHIA events per run per validation point (3 runs/point)
#   N3  = 125     validation points per task  (16 × 125 = 2000 total)
#   n-workers = 16    parallel workers within each task
#   n-mmd     = 2000  MMD subsample size

#SBATCH --job-name=svj_val
#SBATCH --array=0-15                  # 16 tasks; must match N_JOBS below
#SBATCH --nodes=1
#SBATCH -c 16                          # CPUs per task; matches --n-workers
#SBATCH -p arguelles_delgado
#SBATCH --mem=2G
#SBATCH -t 0-2:00                     # 24 h ceiling; expected ~18 h per task
#SBATCH --output=logs/svj_val_%A_%a.out
#SBATCH --error=logs/svj_val_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luc_bojorquezlopez@college.harvard.edu

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs simulated/svj

source ~/venvs/svj/bin/activate

# ── Resource-usage report helper ──────────────────────────────────────────────
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
    printf "  Peak RSS        : %s MB\n"   "$rss_mb"
    printf "  Wall time       : %s\n"      "$wall_raw"
    printf "  CPU cores alloc : %s\n"      "${SLURM_CPUS_ON_NODE:-?}"
    printf "  CPU efficiency  : %s%%\n"    "$cpu_eff"
    printf "  → suggest --mem : %sG   (peak RSS × 1.5 + headroom)\n"  "$suggest_gb"
    echo "================================="
    rm -f "$time_file"
}

TASK="$SLURM_ARRAY_TASK_ID"
N_JOBS=16
N3_PER_JOB=125     # N_JOBS × N3_PER_JOB = 2000 total validation points

echo "=== job $SLURM_JOB_ID  array task $TASK / $((N_JOBS-1))  started $(date) ==="
echo "    node:   $SLURMD_NODENAME"
echo "    cpus:   $SLURM_CPUS_ON_NODE"
echo "    python: $(python --version)"
echo "    N3 this task: $N3_PER_JOB   seed: $TASK"
echo ""

TIME_FILE="/tmp/svj_val_time_${SLURM_JOB_ID}_${TASK}.txt"

/usr/bin/time -v -o "$TIME_FILE" \
    python src/run_regression/validate_production.py \
        simulated/svj/svj_scan.npz \
        --N1 50000 \
        --N2 20000 \
        --N3 ${N3_PER_JOB} \
        --seed ${TASK} \
        --n-workers 16 \
        --n-mmd 2000 \
        --out "simulated/svj/validation_production_${TASK}.npz"

_print_time_report "$TIME_FILE" "svj_val task $TASK"

echo ""
echo "=== job $SLURM_JOB_ID  array task $TASK  finished $(date) ==="

# ── Merge ─────────────────────────────────────────────────────────────────────
# After ALL array tasks complete, merge the 16 shards from the login node:
#
#   python merge_svj_validation.py
#
# Or submit the merge as a dependent job at sbatch time:
#
#   ARRAY_JID=$(sbatch --parsable run_svj_validation.sh)
#   sbatch --dependency=afterok:${ARRAY_JID} \
#          --wrap "source ~/venvs/svj/bin/activate && python merge_svj_validation.py"

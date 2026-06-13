#!/bin/bash
#SBATCH --job-name=svj_scan
#SBATCH --array=0-3                   # 4 parallel slices; must match N_JOBS below
#SBATCH --nodes=1
#SBATCH -c 48
#SBATCH -p arguelles_delgado
#SBATCH --mem=16G
#SBATCH -t 0-10:00
#SBATCH --output=logs/svj_%A_%a.out
#SBATCH --error=logs/svj_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luc_bojorquezlopez@college.harvard.edu

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs simulated/svj

source ~/venvs/svj/bin/activate

echo "=== job $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID  started $(date) ==="
echo "    node:   $SLURMD_NODENAME"
echo "    cpus:   $SLURM_CPUS_ON_NODE"
echo "    python: $(python --version)"
echo ""

# Number of array tasks — must match --array=0-(N_JOBS-1) above.
N_JOBS=4

python src/run_regression/scan_svj.py \
    src/run_regression/scan_regression.cfg \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$N_JOBS"

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

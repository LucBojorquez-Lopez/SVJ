#!/bin/bash
#SBATCH --job-name=svj_gennorm
#SBATCH --array=0-3                   # 4 jobs, one per node
#SBATCH --nodes=1                     # each job on its own node (shared memory required)
#SBATCH -c 48
#SBATCH -p arguelles_delgado
#SBATCH --mem=16G
#SBATCH -t 0-10:00
#SBATCH --output=logs/gennorm_%A_%a.out
#SBATCH --error=logs/gennorm_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luc_bojorquezlopez@college.harvard.edu

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs simulated/gennorm

source ~/venvs/svj/bin/activate

echo "=== job $SLURM_JOB_ID started $(date) ==="
echo "    node:  $SLURMD_NODENAME"
echo "    cpus:  $SLURM_CPUS_ON_NODE"
echo "    python: $(python --version)"
echo ""

python scan_regression_gennorm.py --job-index $SLURM_ARRAY_TASK_ID --n-jobs 4

echo ""
echo "=== job $SLURM_JOB_ID finished $(date) ==="

#!/bin/bash
#SBATCH --job-name=svj_gennorm
#SBATCH --nodes=1                     # all 64 cores must be on one node (shared memory)
#SBATCH -c 64
#SBATCH -p arguelles_delgado
#SBATCH --mem=32G
#SBATCH -t 0-20:00
#SBATCH --output=logs/gennorm_%j.out
#SBATCH --error=logs/gennorm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=luc_bojorquezlopez@college.harvard.edu

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p logs simulated/gennorm

source ~/miniconda3/bin/activate dark

echo "=== job $SLURM_JOB_ID started $(date) ==="
echo "    node:  $SLURMD_NODENAME"
echo "    cpus:  $SLURM_CPUS_ON_NODE"
echo ""

python scan_regression_gennorm.py

echo ""
echo "=== job $SLURM_JOB_ID finished $(date) ==="

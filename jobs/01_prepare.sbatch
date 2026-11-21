#!/bin/bash
#SBATCH --job-name=drug_prepare
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

set -euo pipefail
module load mamba/latest
source activate drugresponse
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
/home/kgadamse/.conda/envs/drugresponse/bin/python -u src/prepare_data.py

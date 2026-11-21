#!/bin/bash
#SBATCH --job-name=drug_mlp
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mlp_%j.out
#SBATCH --error=logs/mlp_%j.err

set -euo pipefail
module load mamba/latest
source activate drugresponse
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
/home/kgadamse/.conda/envs/drugresponse/bin/python -u src/mlp.py --seed 0

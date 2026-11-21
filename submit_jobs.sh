#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for name in OmicsExpressionProteinCodingGenesTPMLogp1.csv Model.csv GDSC2_fitted_dose_response_27Oct23.xlsx; do
  if [[ ! -s "data/raw/$name" ]]; then
    echo "Missing data/raw/$name. Download all 3 files before submitting."
    exit 1
  fi
done
prep=$(sbatch --parsable jobs/01_prepare.sbatch)
echo "Prepare job: $prep"
base=$(sbatch --parsable --dependency="afterok:$prep" jobs/02_baseline.sbatch)
echo "Baseline job: $base (after prepare)"
mlp=$(sbatch --parsable --dependency="afterok:$prep" jobs/03_mlp.sbatch)
echo "MLP job: $mlp (after prepare)"
echo "Check progress with: squeue -u \"$USER\""

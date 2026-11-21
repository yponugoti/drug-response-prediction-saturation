# Drug response prediction: first working experiment

This is a **pilot**, not the complete experimental design. It evaluates one GDSC2 drug using DepMap protein-coding gene expression with five fixed folds. It deliberately postpones the final LINCS L1000 + target gene panel, drug-target mapping, GNN, SHAP, shuffled nulls, and multi-drug aggregation.

## Setup (Mac or Linux)

```bash
cd drug-response-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place these files in `data/raw/`:

- `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (DepMap 24Q4)
- `Model.csv` (DepMap 24Q4 metadata)
- `GDSC2_fitted_dose_response_27Oct23.xlsx` (GDSC2 release 8.5)

## Run

```bash
python src/prepare_data.py
python src/baseline.py
python src/mlp.py
```

`prepare_data.py` automatically picks the drug with the most matched cell lines (minimum 300) and writes `data/processed/pilot.npz`, `folds.npy`, and metadata. To choose a drug, run `python src/prepare_data.py --drug trametinib`. To lower the pilot cutoff, add `--min-cells 100`. To use more features, add `--max-genes 2000` (or `--max-genes 0` for all genes, which is more expensive).

Outputs appear in `results/`. `baseline_folds.csv` and `mlp_folds_seed0.csv` can be compared; their folds are identical. Data standardization, Ridge alpha fitting, and MLP validation are confined to training data. **Important:** the MLP uses 15% of the training fold for early stopping, while Ridge uses three-fold CV. Before final paper-grade comparisons, align hyperparameter budgets, define the official gene panel, test five model seeds, and implement remaining proposal requirements.

## Data assumptions

The scripts expect DepMap gene expression rows indexed by ModelID/DepMap_ID, DepMap `Model.csv` with a COSMIC ID column, and GDSC2 with COSMIC_ID, DRUG_ID, DRUG_NAME and LN_IC50. GDSC duplicate drug/cell-line entries are averaged, and ambiguous COSMIC-to-ModelID mappings are excluded. If column names differ, the scripts show the available names so they can be adjusted.

Raw data is **not included** in this ZIP. The code was smoke-tested on synthetic data, but has not been run on your actual DepMap/GDSC downloads.

## ASU Sol: batch jobs

The `jobs/` folder contains 3 CPU-only SLURM jobs (no A100 required for this pilot). Submit using `submit_jobs.sh`. No training is run until the datasets are downloaded.

1. Transfer/extract this project to Sol (for example, `~/drug-response-project`).
2. Install the environment once on Sol in an interactive/lightwork allocation, not inside every training job:

```bash
module load mamba/latest
mamba create -y -n drugresponse -c conda-forge python=3.11 numpy pandas scipy scikit-learn openpyxl pytorch-cpu
source activate drugresponse
python -c 'import numpy,pandas,sklearn,torch,openpyxl; print("Environment ready")'
```

If `pytorch-cpu` is unavailable from your configured channels, try `pytorch` instead. Confirm `torch` imports before submission.

3. Download the raw datasets on a machine/session that has outbound Internet access with `bash download_data.sh`, or download on your laptop and copy all three files into `data/raw/` on Sol. The script contains externally hosted download links; validate the files after downloading (e.g. CSV header and XLSX file type). Some Sol compute nodes may not have web access.
4. From the project root: `bash submit_jobs.sh`. It schedules preparation first, then baseline and MLP after preparation succeeds. Logs go in `logs/` and model metrics go in `results/`.
5. Inspect with `squeue -u "$USER"`, `tail -f logs/prepare_*.out`, or `cat logs/prepare_*.err`.

The CPU and time allocations are initial estimates, not guarantees. These jobs run just one pilot drug and one MLP seed; they are not the 170-drug/five-seed/GNN final experiments. If `general/public` is unavailable to your account, adjust the partition/QoS to a permitted combination shown by `sinfo`/ASU RC docs.

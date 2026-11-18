"""Prepare one-drug DepMap/GDSC2 dataset for a first prediction experiment.

Pilot mode: uses a deterministic subset of protein-coding genes, NOT the final
LINCS L1000 + drug-target panel specified in the research proposal.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]


def find_column(frame, candidates, description):
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Missing {description}. Available columns: {list(frame.columns)[:35]}")


def cosmic_ids(series):
    # Handles integer-like values stored as strings, numbers, or '12345.0'.
    values = pd.to_numeric(series, errors="coerce")
    return values.astype("Int64")


def read_expression(path):
    print("Loading DepMap gene expression (this may take a moment)...", flush=True)
    expr = pd.read_csv(path, low_memory=False)
    model_col = find_column(expr, ["ModelID", "DepMap_ID", "DepMapID", "Unnamed: 0"], "expression model ID")
    expr = expr.rename(columns={model_col: "ModelID"})
    expr["ModelID"] = expr["ModelID"].astype(str).str.strip()
    expr = expr.drop_duplicates(subset=["ModelID"])
    return expr


def read_model(path):
    model = pd.read_csv(path, low_memory=False)
    id_col = find_column(model, ["ModelID", "DepMap_ID", "DepMapID"], "metadata model ID")
    cosmic_col = find_column(model, ["COSMICID", "COSMIC_ID", "COSMIC ID", "COSMIC_IDs"], "metadata COSMIC ID")
    model = model[[id_col, cosmic_col]].rename(columns={id_col: "ModelID", cosmic_col: "COSMIC_ID"})
    model["ModelID"] = model["ModelID"].astype(str).str.strip()
    model["COSMIC_ID"] = cosmic_ids(model["COSMIC_ID"])
    model = model.dropna(subset=["COSMIC_ID", "ModelID"]).drop_duplicates()
    # Don't guess if one COSMIC ID corresponds to multiple DepMap models.
    counts = model.groupby("COSMIC_ID")["ModelID"].nunique()
    ambiguous = set(counts[counts > 1].index)
    if ambiguous:
        print(f"Dropping {len(ambiguous)} COSMIC IDs with multiple DepMap models.")
    return model[~model["COSMIC_ID"].isin(ambiguous)].drop_duplicates("COSMIC_ID")


def read_gdsc(path):
    if path.suffix.lower() in (".xlsx", ".xls"):
        gdsc = pd.read_excel(path)
    else:
        gdsc = pd.read_csv(path, low_memory=False)
    names = {
        "COSMIC_ID": ["COSMIC_ID", "COSMICID"],
        "DRUG_ID": ["DRUG_ID", "DRUGID"],
        "DRUG_NAME": ["DRUG_NAME", "DRUGNAME"],
        "LN_IC50": ["LN_IC50", "LNIC50"],
    }
    renames = {find_column(gdsc, candidates, label): label for label, candidates in names.items()}
    gdsc = gdsc.rename(columns=renames)
    gdsc["COSMIC_ID"] = cosmic_ids(gdsc["COSMIC_ID"])
    gdsc["LN_IC50"] = pd.to_numeric(gdsc["LN_IC50"], errors="coerce")
    gdsc = gdsc.dropna(subset=["COSMIC_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50"])
    gdsc["DRUG_NAME"] = gdsc["DRUG_NAME"].astype(str).str.strip()
    return gdsc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, default=ROOT / "data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "data/raw/Model.csv")
    parser.add_argument("--gdsc", type=Path, default=ROOT / "data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx")
    parser.add_argument("--drug", help="Exact/partial drug name (default: drug with most matched cell lines)")
    parser.add_argument("--min-cells", type=int, default=300)
    parser.add_argument("--max-genes", type=int, default=1000, help="Pilot only: fixed alphabetic gene subset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for path in (args.expression, args.model, args.gdsc):
        if not path.exists():
            parser.error(f"File not found: {path}. Place downloads in data/raw/ or provide its path.")

    expr = read_expression(args.expression)
    model = read_model(args.model)
    gdsc = read_gdsc(args.gdsc)
    matched_ids = model[model["ModelID"].isin(expr["ModelID"])][["COSMIC_ID", "ModelID"]]
    gdsc = gdsc.merge(matched_ids, on="COSMIC_ID", how="inner", validate="many_to_one")
    if gdsc.empty:
        raise ValueError("No COSMIC IDs matched. Check that Model.csv and expression data are compatible.")

    # Average replicate screens for a drug/cell line, then pick one DRUG_ID per name.
    gdsc = gdsc.groupby(["DRUG_NAME", "DRUG_ID", "ModelID"], as_index=False)["LN_IC50"].mean()
    counts = gdsc.groupby(["DRUG_NAME", "DRUG_ID"])["ModelID"].nunique().reset_index(name="n_cells")
    best_ids = counts.sort_values(["DRUG_NAME", "n_cells"], ascending=[True, False]).drop_duplicates("DRUG_NAME")
    eligible = best_ids[best_ids["n_cells"] >= args.min_cells].sort_values("n_cells", ascending=False)
    if args.drug:
        eligible = eligible[eligible["DRUG_NAME"].str.contains(re.escape(args.drug), case=False, na=False)]
    if eligible.empty:
        print("Largest matched drugs:")
        print(best_ids.nlargest(12, "n_cells").to_string(index=False))
        raise ValueError("No drug met the selection criteria. Try --min-cells 100 for a pilot.")

    chosen = eligible.iloc[0]
    data = gdsc[(gdsc["DRUG_NAME"] == chosen["DRUG_NAME"]) & (gdsc["DRUG_ID"] == chosen["DRUG_ID"])]
    all_gene_cols = [c for c in expr.columns if c != "ModelID"]
    # Sort by gene name, not response or expression magnitude; same rule for every drug.
    gene_cols = sorted(all_gene_cols, key=lambda c: str(c).lower())[:args.max_genes] if args.max_genes > 0 else sorted(all_gene_cols)
    selected = expr[["ModelID"] + gene_cols].merge(data[["ModelID", "LN_IC50"]], on="ModelID", validate="one_to_one")
    selected = selected.sort_values("ModelID").reset_index(drop=True)
    numeric = selected[gene_cols].apply(pd.to_numeric, errors="coerce")
    # Only retain genes measured in every selected cell. No value imputation.
    usable_cols = numeric.columns[numeric.notna().all(axis=0)].tolist()
    if not usable_cols:
        raise ValueError("No fully observed expression features in the chosen panel.")
    X = numeric[usable_cols].to_numpy(dtype=np.float32)
    y = selected["LN_IC50"].to_numpy(dtype=np.float32)
    ids = selected["ModelID"].to_numpy(dtype=str)

    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "pilot.npz", X=X, y=y, model_ids=ids, genes=np.array(usable_cols, dtype=str))
    folds = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_assignments = np.empty(len(y), dtype=np.int64)
    for fold_number, (_, test_idx) in enumerate(folds.split(X)):
        fold_assignments[test_idx] = fold_number
    np.save(out / "folds.npy", fold_assignments)
    metadata = {
        "drug_name": str(chosen["DRUG_NAME"]), "drug_id": str(chosen["DRUG_ID"]),
        "cell_lines": int(len(y)), "genes": int(len(usable_cols)),
        "seed": args.seed, "min_cells": args.min_cells,
        "panel": "PILOT: alphabetically selected protein-coding genes; NOT the planned LINCS+targets panel",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Saved {out / 'pilot.npz'} and {out / 'folds.npy'}")


if __name__ == "__main__":
    main()

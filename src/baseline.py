import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"


def metrics(y_true, y_pred, include_spearman=True):
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic) if include_spearman else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    with np.load(DATA / "pilot.npz") as loaded:
        X, y = loaded["X"], loaded["y"]
    folds = np.load(DATA / "folds.npy")
    assert len(y) == len(folds) and set(folds) == set(range(5))
    OUT.mkdir(exist_ok=True, parents=True)
    rows, predictions = [], []

    for fold in range(5):
        train = folds != fold
        test = folds == fold
        dummy = DummyRegressor(strategy="mean").fit(X[train], y[train])
        ridge = GridSearchCV(
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            cv=KFold(n_splits=3, shuffle=True, random_state=0),
            scoring="neg_mean_squared_error", n_jobs=-1,
        ).fit(X[train], y[train])

        for name, model in [("dummy", dummy), ("ridge", ridge)]:
            prediction = model.predict(X[test])
            rows.append({"model": name, "fold": fold,
                         **metrics(y[test], prediction, include_spearman=(name != "dummy"))})
            for idx, predicted in zip(np.flatnonzero(test), prediction):
                predictions.append({"model": name, "fold": fold, "row_idx": int(idx),
                                    "actual": float(y[idx]), "predicted": float(predicted)})
        print(f"Fold {fold + 1}/5 complete (Ridge alpha={ridge.best_params_['ridge__alpha']})", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "baseline_folds.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "baseline_predictions.csv", index=False)
    summary = pd.DataFrame(rows).groupby("model")[["spearman", "rmse", "r2"]].agg(["mean", "std"])
    print("\nMean and SD across folds:\n", summary.to_string())
    print("\nSaved results/baseline_folds.csv and results/baseline_predictions.csv")


if __name__ == "__main__":
    main()

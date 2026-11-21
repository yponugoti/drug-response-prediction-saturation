import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d = np.load(repo / "data/processed/pilot.npz")
X, y, genes = d["X"], d["y"], np.array([g.split(" (")[0] for g in d["genes"]])
folds = np.load(repo / "data/processed/folds.npy")
res = {}
seed_files = sorted((repo / "results").glob("mlp_folds_seed*.csv"))
pd.concat([pd.read_csv(f) for f in seed_files]).to_csv(out / "mlp_5seeds.csv", index=False)

alphas = np.logspace(-1, 6, 15)
rows, ridge_coefs = [], []
for f in range(5):
    tr, te = folds != f, folds == f
    gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": alphas},
                      cv=KFold(3, shuffle=True, random_state=0), scoring="neg_mean_squared_error").fit(X[tr], y[tr])
    p = gs.predict(X[te])
    rows.append(dict(fold=f, alpha=gs.best_params_["ridge__alpha"], spearman=spearmanr(y[te], p).statistic,
                     rmse=np.sqrt(mean_squared_error(y[te], p)), r2=r2_score(y[te], p)))
    ridge_coefs.append(gs.best_estimator_[-1].coef_)
ridge_ext = pd.DataFrame(rows)
ridge_ext.to_csv(out / "ridge_extended_grid.csv", index=False)
res["ridge_ext"] = ridge_ext.drop(columns="fold").agg(["mean", "std"]).to_dict()
res["ridge_ext_alphas"] = ridge_ext["alpha"].tolist()

bp = pd.read_csv(repo / "results/baseline_predictions.csv")
mp = pd.read_csv(repo / "results/mlp_predictions_seed0.csv")
preds = pd.concat([bp, mp.drop(columns="seed")])
cal = []
for (m, f), g in preds.groupby(["model", "fold"]):
    cal.append(dict(model=m, fold=f, pred_sd=g.predicted.std(), true_sd=g.actual.std(),
                    bias=(g.predicted - g.actual).mean()))
cal = pd.DataFrame(cal)
cal.to_csv(out / "calibration.csv", index=False)
res["calibration"] = cal.groupby("model")[["pred_sd", "true_sd", "bias"]].mean().to_dict()
res["cal_slope"] = {m: float(np.polyfit(g.predicted, g.actual, 1)[0]) for m, g in preds.groupby("model") if m != "dummy"}
q = pd.qcut(bp[bp.model == "ridge"].sort_values("row_idx").actual, 5, labels=["Q1 (most sensitive)", "Q2", "Q3", "Q4", "Q5 (most resistant)"])
qmap = dict(zip(bp[bp.model == "ridge"].sort_values("row_idx").row_idx, q))
preds["quintile"] = preds.row_idx.map(qmap)
preds["abs_err"] = (preds.predicted - preds.actual).abs()
preds["signed_err"] = preds.predicted - preds.actual
eq = preds.groupby(["quintile", "model"], observed=True)[["abs_err", "signed_err"]].mean().unstack("model")
eq.to_csv(out / "error_by_quintile.csv")
res["err_quintile"] = {f"{a}|{m}|{qq}": float(v) for (a, m), col in eq.items() for qq, v in col.items()}
preds.to_csv(out / "pooled_predictions.csv", index=False)

bf = pd.read_csv(repo / "results/baseline_folds.csv")
mf = pd.read_csv(repo / "results/mlp_folds_seed0.csv")
r_sp = bf[bf.model == "ridge"].sort_values("fold").spearman.values
m_sp = mf.sort_values("fold").spearman.values
res["paired_diff"] = (r_sp - m_sp).tolist()
res["wilcoxon_p"] = float(wilcoxon(r_sp, m_sp).pvalue)
res["pooled_spearman"] = {m: float(spearmanr(g.actual, g.predicted).statistic) for m, g in preds.groupby("model") if m != "dummy"}

def spearman_all(Xr, yr):
    yc = yr - yr.mean(); Xc = Xr - Xr.mean(0)
    return (Xc * yc[:, None]).sum(0) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)

rng = np.random.default_rng(0)
dm_top, ridge_top, null_max, obs_max, n_sig = [], [], [], [], []
K = 50
for f in range(5):
    tr = folds != f
    Xr = np.apply_along_axis(rankdata, 0, X[tr]); yr = rankdata(y[tr])
    rho = spearman_all(Xr, yr)
    dm_top.append(set(np.argsort(-np.abs(rho))[:K]))
    ridge_top.append(set(np.argsort(-np.abs(ridge_coefs[f]))[:K]))
    obs_max.append(np.abs(rho).max())
    nm = [np.abs(spearman_all(Xr, rng.permutation(yr))).max() for _ in range(1000)]
    null_max.append(nm)
    thr = np.quantile(nm, 0.95)
    n_sig.append(int((np.abs(rho) > thr).sum()))
    if f == 0:
        rho0 = rho
jac = lambda a, b: len(a & b) / len(a | b)
res["dm_obs_max"] = [float(v) for v in obs_max]
res["dm_null_max95"] = [float(np.quantile(n, 0.95)) for n in null_max]
res["dm_n_sig"] = n_sig
res["jaccard_dm_vs_ridge_top50"] = [jac(a, b) for a, b in zip(dm_top, ridge_top)]
res["jaccard_dm_across_folds"] = float(np.mean([jac(dm_top[i], dm_top[j]) for i in range(5) for j in range(i + 1, 5)]))
res["jaccard_ridge_across_folds"] = float(np.mean([jac(ridge_top[i], ridge_top[j]) for i in range(5) for j in range(i + 1, 5)]))
order = np.argsort(-np.abs(rho0))[:10]
res["dm_top10_fold0"] = [(genes[i], float(rho0[i])) for i in order]
ro = np.argsort(-np.abs(ridge_coefs[0]))[:10]
res["ridge_top10_fold0"] = [(genes[i], float(ridge_coefs[0][i])) for i in ro]
np.save(out / "dm_null_fold0.npy", np.array(null_max[0])); np.save(out / "dm_rho_fold0.npy", rho0)

(out / "diagnostics.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=1, default=float))

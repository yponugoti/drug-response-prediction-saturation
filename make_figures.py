import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

diag, seeds_csv, repo = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
RIDGE, MLP, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#b9b8b3",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.3))

bf = pd.read_csv(repo / "results/baseline_folds.csv"); r = bf[bf.model == "ridge"].sort_values("fold").spearman.values
ms = pd.read_csv(seeds_csv); m0 = ms[ms.seed == 0].sort_values("fold").spearman.values
g = ms.groupby("fold").spearman; mm, sd = g.mean().values, g.std().values
x = np.arange(1, 6); a = ax[0, 0]
for i in range(5):
    a.plot([x[i] - .12, x[i] + .12], [r[i], m0[i]], color="#d6d5d0", lw=1, zorder=1)
a.scatter(x - .12, r, s=34, color=RIDGE, label="Ridge", zorder=3, edgecolor="white", lw=1)
a.scatter(x + .12, m0, s=34, color=MLP, label="MLP (seed 0)", zorder=3, edgecolor="white", lw=1)
a.errorbar(x + .32, mm, yerr=sd, fmt="D", ms=4, color=MLP, mfc="white", lw=1, capsize=2, label="MLP (5-seed mean ± SD)")
a.set_xticks(x); a.set_xticklabels([f"Fold {i}" for i in x]); a.set_ylim(0.2, 0.72)
a.set_ylabel("Held-out Spearman ρ"); a.set_title("(a) Ridge beats the MLP on every fold")
a.legend(loc="upper left", fontsize=7, ncol=1, handletextpad=0.3)
a.grid(axis="y", color="#ecebe7", lw=0.6); a.set_axisbelow(True)

p = pd.read_csv(diag / "pooled_predictions.csv"); a = ax[0, 1]
for name, col, lab in [("ridge", RIDGE, "Ridge"), ("mlp", MLP, "MLP (seed 0)")]:
    q = p[p.model == name]
    a.scatter(q.predicted, q.actual, s=6, alpha=0.45, color=col, lw=0, label=lab)
    k, b0 = np.polyfit(q.predicted, q.actual, 1); xs = np.linspace(-3, 1, 10)
    a.plot(xs, k * xs + b0, color=col, lw=2)
lim = (-3.3, 3.1); a.plot(lim, lim, color=GRAY, lw=1, ls="--", label="y = x"); a.set_xlim(-3.3, 1.2); a.set_ylim(*lim)
a.text(-3.15, 2.6, "Calibration slope\nRidge 0.94 · MLP 0.52", fontsize=7, color=INK2, va="top")
a.set_xlabel("Predicted LN_IC50"); a.set_ylabel("Observed LN_IC50"); a.set_title("(b) Out-of-fold predictions (n = 697)")
a.legend(loc="lower right", fontsize=7, markerscale=2.5)

e = p.groupby(["quintile", "model"])["signed_err"].mean().unstack()
order = ["Q1 (most sensitive)", "Q2", "Q3", "Q4", "Q5 (most resistant)"]; e = e.loc[order]; a = ax[1, 0]
xq = np.arange(5)
for name, col, lab, mk in [("dummy", GRAY, "Dummy mean", "s"), ("ridge", RIDGE, "Ridge", "o"), ("mlp", MLP, "MLP (seed 0)", "o")]:
    a.plot(xq, e[name], color=col, lw=2, marker=mk, ms=5, mec="white", label=lab)
a.axhline(0, color=GRAY, lw=0.8)
a.set_xticks(xq); a.set_xticklabels(["Q1\nsensitive", "Q2", "Q3", "Q4", "Q5\nresistant"])
a.set_ylabel("Mean error (pred − obs)"); a.set_xlabel("Observed LN_IC50 quintile")
a.set_title("(c) Both models shrink toward the mean"); a.legend(loc="upper right", fontsize=7)
a.grid(axis="y", color="#ecebe7", lw=0.6); a.set_axisbelow(True)

rho = np.abs(np.load(diag / "dm_rho_fold0.npy")); null = np.load(diag / "dm_null_fold0.npy"); a = ax[1, 1]
thr = np.quantile(null, 0.95)
a.hist(rho, bins=40, color=RIDGE, alpha=0.85, edgecolor="white", lw=0.5, label="Genes: observed |ρ|")
a.hist(null, bins=20, color=GRAY, alpha=0.6, edgecolor="white", lw=0.5, label="Shuffles: max |ρ|")
a.axvline(thr, color=INK, lw=1, ls="--")
a.text(thr + 0.005, a.get_ylim()[1] * 0.45, f"95% null max = {thr:.2f}\n{(rho > thr).sum()} of 1,000 genes above", fontsize=7, color=INK)
a.set_xlabel("|Spearman ρ| with LN_IC50 (training cells, fold 1)"); a.set_ylabel("Genes / shuffles")
a.set_title("(d) Direct measurement vs shuffle null"); a.legend(loc="upper right", fontsize=7, borderaxespad=0.1)

fig.tight_layout(h_pad=1.2, w_pad=1.5)
fig.savefig(diag / "fig1.png", dpi=220)
print("ok")

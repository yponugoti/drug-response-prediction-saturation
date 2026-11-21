import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUT = ROOT / "results"


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(DATA / "pilot.npz") as loaded:
        X, y = loaded["X"], loaded["y"]
    folds = np.load(DATA / "folds.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for fold in range(5):
        train_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)
        subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=args.seed)
        scaler = StandardScaler().fit(X[subtrain_idx])
        x_train = torch.tensor(scaler.transform(X[subtrain_idx]), dtype=torch.float32, device=device)
        x_val = torch.tensor(scaler.transform(X[val_idx]), dtype=torch.float32, device=device)
        x_test = torch.tensor(scaler.transform(X[test_idx]), dtype=torch.float32, device=device)
        y_train = torch.tensor(y[subtrain_idx], dtype=torch.float32, device=device)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

        # Reproducible initialization across reruns.
        torch.manual_seed(args.seed + fold)
        model = MLP(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = nn.MSELoss()
        best_val, best_state, stale = float("inf"), None, 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction = model(x_test).cpu().numpy()
        correlation = spearmanr(y[test_idx], prediction).statistic
        rows.append({"model": "mlp", "fold": fold, "seed": args.seed,
                     "spearman": float(correlation),
                     "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
                     "r2": float(r2_score(y[test_idx], prediction)), "epochs": epoch + 1})
        for idx, predicted in zip(test_idx, prediction):
            predictions.append({"model": "mlp", "fold": fold, "seed": args.seed,
                                "row_idx": int(idx), "actual": float(y[idx]), "predicted": float(predicted)})
        print(f"Fold {fold + 1}/5: Spearman={correlation:.3f}; RMSE={rows[-1]['rmse']:.3f}; epochs={epoch + 1}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"mlp_folds_seed{args.seed}.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / f"mlp_predictions_seed{args.seed}.csv", index=False)
    summary = pd.DataFrame(rows)[["spearman", "rmse", "r2"]].agg(["mean", "std"])
    print("\nMLP mean and SD across folds:\n", summary.to_string())


if __name__ == "__main__":
    main()

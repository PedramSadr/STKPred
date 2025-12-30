import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import optuna

# -----------------------
# Config & Paths
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = r"C:\My Documents\Mics\Logs"
INPUT_CSV = os.path.join(BASE_DIR, "tsla_daily.csv")

os.makedirs(BASE_DIR, exist_ok=True)


# ============================================================
# 1. Data Preparation (Leakage Free)
# ============================================================
def create_sequences(X_data, y_data, seq_len):
    """Helper to convert flat arrays into LSTM sequences."""
    X, y = [], []
    for i in range(len(X_data) - seq_len):
        X.append(X_data[i: i + seq_len])
        y.append(y_data[i + seq_len])
    return torch.tensor(np.array(X), dtype=torch.float32), \
        torch.tensor(np.array(y), dtype=torch.float32)


def prepare_data(csv_path, seq_len=30, horizon=10, val_split=0.2):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}\nPlease move 'tsla_daily.csv' to {BASE_DIR}")

    df = pd.read_csv(csv_path).ffill().bfill()

    # Feature Engineering
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["target_mu"] = np.log(df["Close"].shift(-horizon) / df["Close"])

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    df["target_sigma"] = df["log_ret"].rolling(window=indexer).std()
    df["target_log_sigma"] = np.log(df["target_sigma"] + 1e-6)
    df.dropna(inplace=True)

    feature_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "VWAP",
                    "SMA_25", "SMA_50", "SMA_200", "MACD_12_26_9", "VIX", "SPY", "DXY"]

    # 1. Split Raw Data FIRST (Chronological split)
    train_size = int(len(df) * (1 - val_split))

    X_raw = df[feature_cols].values
    y_raw = df[["target_mu", "target_log_sigma"]].values

    X_train_raw = X_raw[:train_size]
    X_val_raw = X_raw[train_size:]
    y_train_raw = y_raw[:train_size]
    y_val_raw = y_raw[train_size:]

    # 2. Fit Scalers ONLY on Training Data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    # Transform Validation using Train statistics
    X_val_scaled = scaler_x.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)

    # 3. Create Sequences
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, seq_len)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, seq_len)

    return X_train, y_train, X_val, y_val, scaler_y, len(feature_cols)


# ============================================================
# 2. Model (Fixed Dropout Warning)
# ============================================================
class StockBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        # FIX: PyTorch LSTM ignores dropout if num_layers=1 and throws a warning.
        # We force dropout=0 in that specific case.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)

        self.proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        embed = torch.cat([out[:, -1, :], out.mean(dim=1)], dim=1)
        z = torch.relu(self.proj(embed))
        return self.mu_head(z), self.sigma_head(z)


# ============================================================
# 3. Optuna Objective
# ============================================================
def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = StockBiLSTM(INPUT_DIM, hidden_dim, num_layers, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            mu, sigma = model(xb)
            loss = criterion(mu, yb[:, 0:1]) + 0.5 * criterion(sigma, yb[:, 1:2])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mu, sigma = model(xb)
                loss = criterion(mu, yb[:, 0:1]) + 0.5 * criterion(sigma, yb[:, 1:2])
                val_loss += loss.item()

        val_loss /= len(val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


# ============================================================
# 4. Main Execution
# ============================================================
if __name__ == "__main__":
    try:
        print(f"Checking for data in: {BASE_DIR}")
        X_train, y_train, X_val, y_val, scaler_y, INPUT_DIM = prepare_data(INPUT_CSV)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

        print("Starting Optuna Hyperparameter Tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)

        print("\n------------------------------------------------")
        print("Study Completed.")
        print("Best Hyperparameters:", study.best_params)
        print("Best Validation Loss:", study.best_value)
        print("------------------------------------------------")

        with open(os.path.join(BASE_DIR, "optuna_results.txt"), "w") as f:
            f.write(str(study.best_params))

    except Exception as e:
        print(f"\nERROR: {e}")
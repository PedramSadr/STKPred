import os
import math
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna

# ---------------------------------------------------------
# Environment / device setup (one process per GPU)
# To run on Runpod (e.g., 4 GPUs):
#   torchrun --standalone --nnodes=1 --nproc_per_node=4 RunpodLSTMOptionTrainer.py
# ---------------------------------------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
GLOBAL_RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

if torch.cuda.is_available():
    device = torch.device(f"cuda:{LOCAL_RANK}")
else:
    device = torch.device("cpu")

# ---------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------
DATA_PATH = "/workspace/lstm-project/data/TSLA_Options_Chain_With_Indicators.csv"

BASE_DIR = "/workspace/lstm-project"
DB_PATH = os.path.join(BASE_DIR, "optuna_study.db")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "optuna_checkpoints")
OUTPUT_DIR = "/workspace/output"

STUDY_NAME = "lstm_hyperparameter_study"
TOTAL_TRIALS = 50  # approximate global budget across all ranks
PREDICTION_HORIZON = 14  # predict strike 14 steps (~2 weeks) ahead

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    if GLOBAL_RANK == 0:
        raise SystemExit(f"Failed to read data at {DATA_PATH}: {e}")
    else:
        raise

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Basic sanity check: we need 'strike' as target
if "strike" not in df.columns:
    if GLOBAL_RANK == 0:
        raise SystemExit("Input CSV must contain a 'strike' column (case-insensitive).")
    else:
        raise SystemExit()

# Parse dates to build useful numeric features
if "date" not in df.columns or "expiration" not in df.columns:
    if GLOBAL_RANK == 0:
        raise SystemExit("CSV must contain 'date' and 'expiration' columns.")
    else:
        raise SystemExit()

df["date"] = pd.to_datetime(df["date"])
df["expiration"] = pd.to_datetime(df["expiration"])

# Days to expiration as a numeric feature
df["days_to_expiration"] = (df["expiration"] - df["date"]).dt.days.astype(np.float32)

# Encode option type (call/put) into a numeric feature
# Adjust mapping if your 'type' values differ (e.g. 'C'/'P', 'CALL'/'PUT', etc.)
if "type" in df.columns:
    type_map = {
        "C": 1.0,
        "CALL": 1.0,
        "P": 0.0,
        "PUT": 0.0,
    }
    df["type_encoded"] = (
        df["type"].astype(str).str.upper().map(type_map).fillna(0.0).astype(np.float32)
    )
else:
    df["type_encoded"] = 0.0  # fallback if missing

# Sort by date (and expiration as a secondary key) to define a time order
df = df.sort_values(["date", "expiration", "strike"]).reset_index(drop=True)

# Target: we predict future strike
prices = df["strike"].values.astype(np.float32)

# ---------------------------------------------------------
# Feature selection
# You said the CSV columns are:
# contractid, expiration, strike, type, last, mark, bid, bid_size, ask, ask_size,
# volume, open_interest, date, implied_volatility, delta, gamma, theta, vega, rho,
# rsi, vwap, sma_25, sma_50, sma_100, sma_200
#
# We drop non-numeric IDs / raw text, and keep numeric versions.
# ---------------------------------------------------------
numeric_feature_cols = [
    # NOTE: we deliberately SKIP 'contractid', 'date', 'expiration', and raw 'type'
    "strike",
    "last",
    "mark",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "rsi",
    "vwap",
    "sma_25",
    "sma_50",
    "sma_100",
    "sma_200",
    "days_to_expiration",
    "type_encoded",
]

# Keep only those that are actually in the dataframe
feature_cols = [c for c in numeric_feature_cols if c in df.columns]

if len(feature_cols) == 0:
    if GLOBAL_RANK == 0:
        raise SystemExit("No recognized numeric feature columns found in CSV.")
    else:
        raise SystemExit()

features_df = df[feature_cols].copy()

# ---- FIX: force all feature columns to numeric to avoid string values ----
for col in features_df.columns:
    features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

# Drop zero-variance features (constant columns)
stds = features_df.std()  # now safe: all columns are numeric (or NaN)
zero_std_cols = stds[stds < 1e-6].index
if not zero_std_cols.empty and GLOBAL_RANK == 0:
    print(f"Warning: Removing zero-variance features: {list(zero_std_cols)}")
if not zero_std_cols.empty:
    features_df = features_df.drop(columns=zero_std_cols)
    feature_cols = features_df.columns.tolist()

features = features_df.values.astype(np.float32)
input_size = len(feature_cols)
output_size = 1  # predict a single value (future strike)

# Train/validation split (no leakage)
train_features, val_features, train_prices, val_prices = train_test_split(
    features, prices, test_size=0.2, shuffle=False
)

# Normalization based on training slice only
train_feature_mean = train_features.mean(axis=0)
train_feature_std = train_features.std(axis=0) + 1e-9
train_price_mean = train_prices.mean()
train_price_std = train_prices.std() + 1e-9

train_features = (train_features - train_feature_mean) / train_feature_std
val_features = (val_features - train_feature_mean) / train_feature_std

normalized_train_prices = (train_prices - train_price_mean) / train_price_std
normalized_val_prices = (val_prices - train_price_mean) / train_price_std

# ---------------------------------------------------------
# Sequence creation (fixed prediction horizon)
# ---------------------------------------------------------
def create_sequences(features, prices, seq_length, horizon=PREDICTION_HORIZON):
    """
    Build sequences of length `seq_length` and targets `horizon` steps ahead.

    X[t]  = features[t : t+seq_length]
    y[t]  = prices[t + seq_length + horizon - 1]
    """
    xs, ys = [], []
    max_start = len(features) - seq_length - horizon + 1
    if max_start <= 0:
        return np.array([]), np.array([])
    for i in range(max_start):
        xs.append(features[i : i + seq_length])
        ys.append(prices[i + seq_length + horizon - 1])
    return np.array(xs), np.array(ys)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_model(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    val_loader,
    device,
    epochs,
    trial: optuna.trial.Trial = None,
    patience=20,
    min_delta=1e-5,
):
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item()

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)

        # Scheduler on validation loss
        try:
            scheduler.step(avg_val_loss)
        except Exception:
            pass

        # Optuna pruning (only on rank 0 to avoid conflicts)
        if trial is not None and GLOBAL_RANK == 0:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Track best model
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_loss, best_state

# ---------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------
def objective(trial: optuna.trial.Trial):
    if GLOBAL_RANK == 0:
        print(f"Starting Optuna trial {trial.number} on rank {GLOBAL_RANK}")

    hidden_size = trial.suggest_int("hidden_size", 30, 90)
    sequence_length = trial.suggest_int("sequence_length", 14, 50)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 50, 200)
    batch_size = trial.suggest_categorical("batch_size", [32, 48, 64, 96, 128])

    X_train, y_train = create_sequences(
        train_features, normalized_train_prices, sequence_length
    )
    X_val, y_val = create_sequences(
        val_features, normalized_val_prices, sequence_length
    )

    if len(X_train) == 0 or len(X_val) == 0:
        return float("inf")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # we are minimizing MSE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss, best_state = train_model(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs,
        trial=trial,
    )

    if best_state is not None:
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"rank{GLOBAL_RANK}_trial{trial.number}_best.pth"
        )
        torch.save(best_state, checkpoint_path)
        trial.set_user_attr("checkpoint_path", checkpoint_path)
        trial.set_user_attr("hyperparameters", trial.params)

    return best_val_loss  # Optuna minimizes this

# ---------------------------------------------------------
# Main: multi-process, multi-GPU Optuna
# ---------------------------------------------------------
if __name__ == "__main__":
    STORAGE_PATH = f"sqlite:///{DB_PATH}"
    if GLOBAL_RANK == 0:
        print(f"Starting hyperparameter optimization study: {STUDY_NAME}")
        print(f"Using Optuna storage: {STORAGE_PATH}")
        print(f"WORLD_SIZE (GPUs/processes): {WORLD_SIZE}")

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
    )

    # Distribute total trials across workers
    trials_per_worker = math.ceil(TOTAL_TRIALS / WORLD_SIZE)

    study.optimize(
        objective,
        n_trials=trials_per_worker,
        show_progress_bar=(GLOBAL_RANK == 0),
    )

    # Only rank 0 saves the final "best overall" model
    if GLOBAL_RANK == 0:
        print("\nOptimization finished.")
        best_trial = study.best_trial
        print(f"Best trial value (Validation Loss): {best_trial.value}")
        print(f"Best hyperparameters: {best_trial.params}")

        best_checkpoint = best_trial.user_attrs["checkpoint_path"]
        print(f"Loading best model weights from: {best_checkpoint}")

        best_state_dict = torch.load(best_checkpoint, map_location="cpu")
        final_model = BiLSTMNet(input_size, best_trial.params["hidden_size"], output_size)
        final_model.load_state_dict(best_state_dict)

        date_str = datetime.now().strftime("%Y%m%d")
        model_filename = f"bilstm_option_{date_str}.pth"
        final_path = os.path.join(OUTPUT_DIR, model_filename)
        torch.save(final_model.state_dict(), final_path)
        print(f"Saved the best model to {final_path}")

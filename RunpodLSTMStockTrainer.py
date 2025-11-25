import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna

# ---------------------------------------------------------
# Environment / device setup (one process per GPU)
#This Code Sets Up the Environment for Multi-GPU Training Using PyTorch and Optuna
#On Runpods.io
#To run this code on Runpods.io use the following command:
#torchrun --standalone --nnodes=1 --nproc_per_node=4 RunpodLSTMStockTrainer.py
# Each process will automatically get its own LOCAL_RANK and GLOBAL_RANK
#4 indicates number of GPUs to use
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
DATA_PATH = "/workspace/lstm-project/data/tsla_daily.csv"
BASE_DIR = "/workspace/lstm-project"
DB_PATH = os.path.join(BASE_DIR, "optuna_study.db")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "optuna_checkpoints")
OUTPUT_DIR = "/workspace/output"
STUDY_NAME = "lstm_hyperparameter_study"
TOTAL_TRIALS = 50  # approximate global budget

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

df.columns = df.columns.str.strip().str.lower()
df = df.dropna()

if "close" not in df.columns:
    if GLOBAL_RANK == 0:
        raise SystemExit("Input CSV must contain a 'Close' column (case-insensitive).")
    else:
        raise SystemExit()

prices = df["close"].values.astype(np.float32)

feature_cols = [
    "open", "high", "low", "close", "volume", "rsi", "vwap", "sma_25", "sma_50",
    "sma_100", "sma_200", "wti", "vix", "dxy", "spy", "macd_12_26_9",
]
feature_cols = [c for c in feature_cols if c in df.columns]

if len(feature_cols) == 0:
    if GLOBAL_RANK == 0:
        raise SystemExit("No recognized feature columns found in CSV.")
    else:
        raise SystemExit()

features_df = df[feature_cols].copy()

# Drop zero-variance features
stds = features_df.std()
zero_std_cols = stds[stds < 1e-6].index
if not zero_std_cols.empty and GLOBAL_RANK == 0:
    print(f"Warning: Removing zero-variance features: {list(zero_std_cols)}")
if not zero_std_cols.empty:
    features_df = features_df.drop(columns=zero_std_cols)
    feature_cols = features_df.columns.tolist()

features = features_df.values.astype(np.float32)
input_size = len(feature_cols)
output_size = 1

# Train/val split (no leakage)
train_features, val_features, train_prices, val_prices = train_test_split(
    features, prices, test_size=0.2, shuffle=False
)

train_feature_mean = train_features.mean(axis=0)
train_feature_std = train_features.std(axis=0) + 1e-9
train_price_mean = train_prices.mean()
train_price_std = train_prices.std() + 1e-9

train_features = (train_features - train_feature_mean) / train_feature_std
val_features = (val_features - train_feature_mean) / train_feature_std

normalized_train_prices = (train_prices - train_price_mean) / train_price_std
normalized_val_prices = (val_prices - train_price_mean) / train_price_std

# ---------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------
def create_sequences(features, prices, seq_length):
    xs, ys = [], []
    if len(features) <= seq_length:
        return np.array([]), np.array([])
    for i in range(len(features) - seq_length):
        xs.append(features[i : i + seq_length])
        ys.append(prices[i + seq_length])
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
# Training loop (single-GPU per process)
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
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                epoch_val_loss += criterion(pred, yb).item()

        avg_train_loss = epoch_train_loss / max(len(train_loader), 1)
        avg_val_loss = epoch_val_loss / max(len(val_loader), 1)

        # Scheduler on validation loss
        try:
            scheduler.step(avg_val_loss)
        except Exception:
            pass

        # Optuna pruning
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
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
    criterion = nn.MSELoss()
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

    return best_val_loss

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

    # Distribute total trials roughly across workers
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

        final_path = os.path.join(OUTPUT_DIR, "best_lstm_model.pth")
        torch.save(final_model.state_dict(), final_path)
        print(f"Saved the best model to {final_path}")

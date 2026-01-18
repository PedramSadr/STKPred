import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
FILE_PATH = r"C:\My Documents\Mics\Logs\TSLA_Surface_Vector_Merged.csv"
SAVE_DIR = r"C:\My Documents\Mics\Logs"
BATCH_SIZE = 300
EPOCHS = 50
SEQ_LENGTHS = [3, 4, 5, 6, 7, 8, 9]

#OptionsTrainer.py is the retired scientist who did the foundational
# research (finding out that "3 days" is the magic number).
# Its job is done, and it's enjoying a nice retirement in your archive folder.

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def prepare_data(seq_length):
    """Loads data, scales it, and creates sequences."""
    df = pd.read_csv(FILE_PATH)

    # Drop non-numeric columns (like trade_date)
    data = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    # Create sequences
    # Assuming the goal is to predict the LAST column (iv_change_1d)
    target_col_idx = -1

    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i: i + seq_length])
        y.append(data_scaled[i + seq_length, target_col_idx])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return torch.FloatTensor(X), torch.FloatTensor(y)


def objective(trial):
    # 1. Suggest Hyperparameter
    seq_length = trial.suggest_categorical("seq_length", SEQ_LENGTHS)

    # 2. Prepare Data
    X, y = prepare_data(seq_length)

    # Split into Train/Val (80/20)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    # 3. Build Model
    input_size = X.shape[2]
    model = BiLSTM(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 5. Evaluate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)

    avg_val_loss = val_loss / len(X_val)

    # Log result manually (Optuna also logs automatically)
    print(f"Trial {trial.number}: Seq_Length={seq_length}, Val_Loss={avg_val_loss:.6f}")

    return avg_val_loss


if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(direction="minimize")
    print("Starting optimization...")

    # Run optimization (one trial per seq_length choice roughly, or set n_trials=20 to explore well)
    study.optimize(objective, n_trials=3000)

    print("\nBest params:", study.best_params)
    print("Best value (MSE):", study.best_value)

    # --- Retrain and Save Best Model ---
    best_seq = study.best_params['seq_length']
    print(f"\nRetraining best model with seq_length={best_seq}...")

    X, y = prepare_data(best_seq)
    full_loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    final_model = BiLSTM(X.shape[2]).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        final_model.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            optimizer.step()

    save_path = os.path.join(SAVE_DIR, "best_bilstm_model.pth")
    torch.save(final_model.state_dict(), save_path)
    print(f"Model saved successfully to: {save_path}")
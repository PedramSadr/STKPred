import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import optuna


# --- 1. Data Preparation based on Blueprint ---
def prepare_data(csv_path, seq_length=30, forecast_horizon=10):
    df = pd.read_csv(csv_path)

    # Fill missing values if any
    df = df.ffill().bfill()

    # Calculate Daily Log Returns for Volatility Calculation
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Blueprint Targets ---
    # 1. mu (Expected 2-week log return)
    df['target_mu'] = np.log(df['Close'].shift(-forecast_horizon) / df['Close'])

    # 2. sigma (Expected 2-week realized volatility)
    # Std dev of daily returns over the next 10 days
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=forecast_horizon)
    df['target_sigma'] = df['log_ret'].rolling(window=indexer).std()

    # Drop NaNs created by shifting/rolling
    df.dropna(inplace=True)

    # Features (Technical Indicators + Macro from CSV)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP',
                    'SMA_25', 'SMA_50', 'SMA_200', 'VIX', 'DXY', 'SPY', 'MACD_12_26_9']

    # Scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()  # Scale targets to help convergence

    X_data = scaler_x.fit_transform(df[feature_cols].values)
    y_data = scaler_y.fit_transform(df[['target_mu', 'target_sigma']].values)

    # Create Sequences
    X, y = [], []
    for i in range(len(X_data) - seq_length):
        X.append(X_data[i: i + seq_length])
        y.append(y_data[i + seq_length])

    return np.array(X), np.array(y), scaler_y, len(feature_cols)


# --- 2. Model Architecture (Blueprint: Stock LSTM) ---
class BiLSTM_DualHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BiLSTM_DualHead, self).__init__()

        # Bi-Directional LSTM for "Learned embedding representing price regime"
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        # Head 1: Directional Signal (mu)
        self.head_mu = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Head 2: Volatility Signal (sigma)
        self.head_sigma = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Volatility must be positive
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)

        # Use the last hidden state as the "Regime Embedding"
        embedding = out[:, -1, :]

        mu = self.head_mu(embedding)
        sigma = self.head_sigma(embedding)

        return mu, sigma


# --- 3. Optuna Objective Function ---
def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Load Data
    X, y, _, input_dim = prepare_data('tsla_daily.csv')

    # Train/Val Split (Time-series split: no shuffling)
    train_size = int(len(X) * 0.8)
    X_train, X_val = torch.FloatTensor(X[:train_size]), torch.FloatTensor(X[train_size:])
    y_train, y_val = torch.FloatTensor(y[:train_size]), torch.FloatTensor(y[train_size:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Model Setup
    model = BiLSTM_DualHead(input_dim, hidden_size, num_layers, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Using MSE for regression of both targets

    # Training Loop (Short epoch count for tuning speed)
    model.train()
    for epoch in range(5):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            mu_pred, sigma_pred = model(X_batch)

            # Combined Loss: Minimize error for both Return (0) and Volatility (1)
            loss = criterion(mu_pred, y_batch[:, 0:1]) + criterion(sigma_pred, y_batch[:, 1:2])

            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            mu_pred, sigma_pred = model(X_batch)
            loss = criterion(mu_pred, y_batch[:, 0:1]) + criterion(sigma_pred, y_batch[:, 1:2])
            val_loss += loss.item()

    return val_loss / len(val_loader)


# --- 4. Execution ---
if __name__ == "__main__":
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)  # Set n_trials higher for real training

    print("Best Hyperparameters:", study.best_params)

    # (Optional) Re-train best model here using study.best_params
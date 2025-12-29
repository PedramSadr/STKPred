#!/usr/bin/env python3
import os
import math
import time
import json
import argparse
import random
import gc
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# =========================================================
#  CONFIGURATION & STRATEGY
#  1. Target: Realized Volatility (The "Truth" Blueprint)
#  2. Input: Options Surface + VIX Spread (The "Context")
#  3. Loss: MSE (Punish outliers)
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = int(seq_len)
        self.length = max(0, self.X.shape[0] - self.seq_len)

    def __len__(self): return self.length

    def __getitem__(self, idx):
        # Input: Sequence i to i+seq_len
        # Target: The value calculated for the end of that sequence
        return (torch.from_numpy(self.X[idx: idx + self.seq_len]),
                torch.from_numpy(self.y[idx + self.seq_len - 1]))


class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        # Bidirectional LSTM to see past context clearly
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size * 2)

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Predicts 1 scalar (Realized Vol)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # We take the last time step's output
        last = self.norm(out[:, -1, :])
        return self.head(last)


def main():
    ap = argparse.ArgumentParser()
    # INPUT FILES
    ap.add_argument("--surface-csv", default="TSLA_Surface_Vector.csv", help="The engineered features")
    ap.add_argument("--daily-csv", default="tsla_daily.csv", help="REQUIRED for Realized Vol Target")
    ap.add_argument("--vix-csv", default="vix_daily.csv", help="REQUIRED for Market Context")

    # HYPERPARAMETERS (Blueprint Defaults)
    ap.add_argument("--seq-len", type=int, default=10, help="Short window (10d) to catch acute signals")
    ap.add_argument("--horizon", type=int, default=14, help="Target: 14-day future realized vol")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="stage2_bilstm_vix.pth")

    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- STARTING TRAINING (VIX-ENHANCED) ---")

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print("1. Loading Surface, Stock, and VIX data...")
    if not os.path.exists(args.vix_csv):
        raise FileNotFoundError(f"Missing VIX file: {args.vix_csv}")

    df_surf = pd.read_csv(args.surface_csv)
    df_daily = pd.read_csv(args.daily_csv)
    df_vix = pd.read_csv(args.vix_csv)

    # Standardize Columns
    for df in [df_surf, df_daily, df_vix]:
        df.columns = df.columns.str.strip().str.lower()

    # Standardize Dates
    # Helper to find date column
    def get_date_col(df):
        return 'trade_date' if 'trade_date' in df.columns else 'date'

    df_surf.rename(columns={get_date_col(df_surf): 'date'}, inplace=True)
    df_daily.rename(columns={get_date_col(df_daily): 'date'}, inplace=True)
    df_vix.rename(columns={get_date_col(df_vix): 'date'}, inplace=True)

    for df in [df_surf, df_daily, df_vix]:
        df['date'] = pd.to_datetime(df['date'])

    # ---------------------------------------------------------
    # 2. TRIPLE MERGE (The Alignment)
    # ---------------------------------------------------------
    print("2. Merging Datasets...")
    # Merge Surface + Daily Stock
    df = pd.merge(df_surf, df_daily[['date', 'close']], on='date', how='inner')

    # Merge + VIX (We rename VIX close to avoid collision)
    df_vix = df_vix[['date', 'close']].rename(columns={'close': 'vix_close'})
    df = pd.merge(df, df_vix, on='date', how='inner')

    df = df.sort_values('date').reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. FEATURE ENGINEERING (VIX SPREAD)
    # ---------------------------------------------------------
    print("3. Engineering VIX Context...")
    # TSLA IV is decimal (0.45). VIX is index (20.0).
    # We convert VIX to decimal: 20.0 -> 0.20
    df['vix_decimal'] = df['vix_close'] / 100.0

    # FEATURE: The Fear Spread
    # Positive = TSLA is riskier than Market (Normal)
    # Spike = TSLA is panicking while market is calm (Alpha Signal)
    df['vol_spread'] = df['atm_iv_14d'] - df['vix_decimal']

    # ---------------------------------------------------------
    # 4. TARGET ENGINEERING (BLUEPRINT: REALIZED VOL)
    # ---------------------------------------------------------
    print("4. Calculating Realized Volatility Target...")
    # Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # Realized Vol (Backward looking 14d window)
    df['realized_vol'] = df['log_ret'].rolling(window=args.horizon).std() * np.sqrt(252)

    # SHIFT TARGET (Future looking)
    # The value at Row T becomes the Realized Vol of T+1 to T+14
    df['target'] = df['realized_vol'].shift(-args.horizon)

    # Drop NaNs (Last 14 days have no future target)
    df = df.dropna().reset_index(drop=True)

    print(f"   -> Training Samples: {len(df)}")

    # ---------------------------------------------------------
    # 5. PREPARE INPUTS
    # ---------------------------------------------------------
    # Define Feature List explicitly to avoid garbage
    # We use the Spread, not raw VIX (to reduce correlation noise)
    feature_cols = [
        'atm_iv_14d',
        'skew_25d_14d',
        'term_structure_30_90',
        'net_dollar_flow_14d',
        'iv_change_1d',
        'vol_spread'  # <--- The New VIX Feature
    ]

    # Verify cols exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    print(f"   -> Features used: {feature_cols}")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = scaler_X.fit_transform(df[feature_cols])
    y_raw = scaler_y.fit_transform(df[['target']])

    # Chronological Split (80% Train, 20% Val)
    split_idx = int(len(X_raw) * 0.8)

    train_ds = SequenceDataset(X_raw[:split_idx], y_raw[:split_idx], args.seq_len)
    val_ds = SequenceDataset(X_raw[split_idx:], y_raw[split_idx:], args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---------------------------------------------------------
    # 6. TRAINING LOOP
    # ---------------------------------------------------------
    print("5. Training Model...")
    model = BiLSTMNet(len(feature_cols), 32, 2, 0.3).to(device)
    criterion = nn.MSELoss()  # Strict MSE per Blueprint
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    patience = 15
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item()
        val_loss /= len(val_loader)

        # Logging
        if epoch % 10 == 0:
            print(f"Ep {epoch}: Val MSE {val_loss:.5f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), args.out)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping at Epoch {epoch}")
                break

    print(f"SUCCESS. Best Model saved to {args.out}")


if __name__ == "__main__":
    main()
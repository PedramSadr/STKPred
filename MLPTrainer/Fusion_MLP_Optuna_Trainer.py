import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
BASE_DIR = r"C:\My Documents\Mics\Logs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STOCK_CSV = os.path.join(BASE_DIR, "tsla_daily.csv")
OPTION_CSV = os.path.join(BASE_DIR, "TSLA_Surface_Vector_Merged.csv")

STOCK_MODEL_PATH = os.path.join(BASE_DIR, "best_stock_model.pth")
OPTION_MODEL_PATH = os.path.join(BASE_DIR, "Final_TSLA_Quantile_Model.pth")
BEST_FUSION_MODEL_PATH = os.path.join(BASE_DIR, "best_fusion_model.pth")

SEQ_LEN_STOCK = 30
SEQ_LEN_OPTION = 3
PREDICT_HORIZON = 10


# ============================================================
# 1. PRE-TRAINED MODELS
# ============================================================
class StockBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

    def get_embedding(self, x):
        out, _ = self.lstm(x)
        embed = torch.cat([out[:, -1, :], out.mean(dim=1)], dim=1)
        return torch.relu(self.proj(embed))


class QuantileBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_quantiles):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_quantiles)

    def get_embedding(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

    # ============================================================


# 2. FUSION MLP
# ============================================================
class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        current_dim = hidden_dim
        for _ in range(num_layers - 1):
            next_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. ROBUST DATA PREPARATION
# ============================================================
def prepare_fused_data():
    print("--- Loading and Aligning Data ---")

    if not os.path.exists(STOCK_CSV) or not os.path.exists(OPTION_CSV):
        raise FileNotFoundError(f"Files not found in {BASE_DIR}")

    df_stock = pd.read_csv(STOCK_CSV)
    df_option = pd.read_csv(OPTION_CSV)

    # Clean Names
    df_stock.columns = df_stock.columns.str.strip()
    df_option.columns = df_option.columns.str.strip()

    # --- 1. Identify Option Features BEFORE Merge ---
    date_candidates = ['date', 'trade_date', 'datetime', 'timestamp', 'quote_date']
    opt_date_col = next((c for c in df_option.columns if c.lower() in date_candidates), None)

    numeric_opt_cols = df_option.select_dtypes(include=[np.number]).columns.tolist()
    if opt_date_col and opt_date_col in numeric_opt_cols:
        numeric_opt_cols.remove(opt_date_col)

    print(f"Option Model Input Candidates ({len(numeric_opt_cols)}): {numeric_opt_cols}")

    # --- 2. Fix Dates for Merging ---
    def fix_date_col(df):
        for col in df.columns:
            if col.lower() in date_candidates:
                df.rename(columns={col: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            return df
        raise KeyError(f"Could not find Date column. Cols: {df.columns.tolist()}")

    df_stock = fix_date_col(df_stock)
    df_option = fix_date_col(df_option)

    # --- 3. Merge ---
    df = pd.merge(df_stock, df_option, on='Date', suffixes=('', '_opt')).sort_values('Date').reset_index(drop=True)
    print(f"Aligned Data Rows: {len(df)}")

    # --- AGGRESSIVE SANITIZATION (Fix Object Error) ---
    print("Sanitizing Data (Forcing ALL columns to numeric)...")
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix deprecation and fill NaNs
    df = df.ffill().bfill().fillna(0.0)

    # --- 4. Feature Engineering ---
    all_cols = df.columns.tolist()
    if 'atm_iv_14d' in all_cols:
        iv_col = 'atm_iv_14d'
    elif 'atm_iv_14d_opt' in all_cols:
        iv_col = 'atm_iv_14d_opt'
    else:
        iv_col = numeric_opt_cols[-1]
        if iv_col not in df.columns: iv_col = f"{iv_col}_opt"

    print(f"Selected '{iv_col}' as Implied Volatility proxy.")

    # Targets & VRP
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Realized_Vol_30d'] = df['log_ret'].rolling(30).std() * np.sqrt(252)
    df['VRP'] = df[iv_col] - df['Realized_Vol_30d']
    df['VRP'] = df['VRP'].fillna(0.0)

    df['fwd_close'] = df['Close'].shift(-PREDICT_HORIZON)
    df['target_mu'] = np.log(df['fwd_close'] / df['Close'])
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PREDICT_HORIZON)
    df['target_sigma'] = df['log_ret'].rolling(window=indexer).std()
    df['target_aiv'] = df[iv_col].shift(-PREDICT_HORIZON) - df[iv_col]

    df.dropna(inplace=True)

    # --- 5. Prepare Tensors (Explicit Float32) ---
    stock_feats = ["Open", "High", "Low", "Close", "Volume", "RSI", "VWAP",
                   "SMA_25", "SMA_50", "SMA_200", "MACD_12_26_9", "VIX", "SPY", "DXY"]
    for c in stock_feats:
        if c not in df.columns: df[c] = 0.0

    scaler_stock = MinMaxScaler()
    X_stock_raw = scaler_stock.fit_transform(df[stock_feats].values).astype(np.float32)

    final_opt_cols_for_input = []
    for c in numeric_opt_cols:
        if c in df.columns:
            final_opt_cols_for_input.append(c)
        elif f"{c}_opt" in df.columns:
            final_opt_cols_for_input.append(f"{c}_opt")

    # Input size check/padding logic could go here if needed

    scaler_option = StandardScaler()
    X_option_raw = scaler_option.fit_transform(df[final_opt_cols_for_input].values).astype(np.float32)

    market_feats = df[[iv_col, 'Realized_Vol_30d', 'VRP']].values
    scaler_market = MinMaxScaler()
    X_market_raw = scaler_market.fit_transform(market_feats).astype(np.float32)

    # --- 6. Generate Embeddings ---
    print("--- Generating Embeddings from Frozen LSTMs ---")

    ckpt_s = torch.load(STOCK_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model_s = StockBiLSTM(ckpt_s["input_dim"], ckpt_s["params"]["hidden_dim"],
                          ckpt_s["params"]["num_layers"], ckpt_s["params"]["dropout"]).to(DEVICE)
    model_s.load_state_dict(ckpt_s["model_state"])
    model_s.eval()

    model_o = QuantileBiLSTM(input_size=X_option_raw.shape[1], hidden_size=64, num_layers=2, num_quantiles=3).to(DEVICE)
    model_o.load_state_dict(torch.load(OPTION_MODEL_PATH, map_location=DEVICE, weights_only=False))
    model_o.eval()

    embeddings = []
    targets = []
    start_idx = max(SEQ_LEN_STOCK, SEQ_LEN_OPTION)

    with torch.no_grad():
        for i in range(start_idx, len(df)):
            s_in = torch.tensor(X_stock_raw[i - SEQ_LEN_STOCK:i]).float().unsqueeze(0).to(DEVICE)
            s_emb = model_s.get_embedding(s_in)

            o_in = torch.tensor(X_option_raw[i - SEQ_LEN_OPTION:i]).float().unsqueeze(0).to(DEVICE)
            o_emb = model_o.get_embedding(o_in)

            m_in = torch.tensor(X_market_raw[i]).float().unsqueeze(0).to(DEVICE)

            fused = torch.cat([s_emb, o_emb, m_in], dim=1)
            embeddings.append(fused)

            # Explicit float conversion
            t = df.iloc[i][['target_mu', 'target_sigma', 'target_aiv']].values.astype(np.float32)
            targets.append(torch.tensor(t).unsqueeze(0))

    X_final = torch.cat(embeddings, dim=0)
    y_final = torch.cat(targets, dim=0)

    print(f"Fusion Dataset Ready. Shape: {X_final.shape}")
    return X_final, y_final


# ============================================================
# 4. OPTUNA OPTIMIZATION
# ============================================================
def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    model = FusionMLP(INPUT_DIM, hidden_dim, num_layers, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # FIX: Added drop_last=True to prevent BatchNorm error on single-sample batches
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()

        trial.report(val_loss / len(val_loader), epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss / len(val_loader)


# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    try:
        X_all, y_all = prepare_fused_data()
        INPUT_DIM = X_all.shape[1]

        split = int(len(X_all) * 0.8)
        X_train, X_val = X_all[:split], X_all[split:]
        y_train, y_val = y_all[:split], y_all[split:]

        print("\n--- Starting Optuna Tuning ---")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=25)

        print(f"Best Params: {study.best_params}")

        print("\n--- Retraining Best Fusion Model ---")
        best = study.best_params
        final_model = FusionMLP(INPUT_DIM, best["hidden_dim"], best["num_layers"], best["dropout"]).to(DEVICE)
        optimizer = optim.Adam(final_model.parameters(), lr=best["lr"])
        criterion = nn.MSELoss()

        # FIX: Added drop_last=True here as well
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best["batch_size"], shuffle=True,
                            drop_last=True)

        for epoch in range(50):
            final_model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(final_model(xb), yb)
                loss.backward()
                optimizer.step()

        torch.save({
            "model_state": final_model.state_dict(),
            "params": best,
            "input_dim": INPUT_DIM
        }, BEST_FUSION_MODEL_PATH)

        print(f"Success. Calibrated Fusion Model saved to: {BEST_FUSION_MODEL_PATH}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
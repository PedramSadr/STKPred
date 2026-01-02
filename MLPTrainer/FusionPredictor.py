import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ============================================================
# 1. MODEL ARCHITECTURES
# ============================================================
class StockBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.proj = nn.Linear(hidden_dim * 4, hidden_dim)

        # HEADS (Required to match saved state_dict keys)
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
# 2. PREDICTOR CLASS
# ============================================================
class FusionPredictor:
    def __init__(self, base_dir=r"C:\My Documents\Mics\Logs", device="cpu"):
        self.base_dir = base_dir
        self.device = device

        # Paths
        self.stock_csv = os.path.join(base_dir, "tsla_daily.csv")
        self.option_csv = os.path.join(base_dir, "TSLA_Surface_Vector_Merged.csv")
        self.stock_model_path = os.path.join(base_dir, "best_stock_model.pth")
        self.option_model_path = os.path.join(base_dir, "Final_TSLA_Quantile_Model.pth")
        self.fusion_model_path = os.path.join(base_dir, "best_fusion_model.pth")

        # Config (Must match training)
        self.seq_len_stock = 30
        self.seq_len_option = 3
        self.predict_horizon = 10

        # Load Data & Fit Scalers
        self._load_and_prepare_data()

        # Load Models
        self._load_models()

    def _load_and_prepare_data(self):
        """Re-runs the exact data pipeline to ensure scalers match training distribution."""
        if not os.path.exists(self.stock_csv) or not os.path.exists(self.option_csv):
            raise FileNotFoundError("Missing source CSVs for inference.")

        df_stock = pd.read_csv(self.stock_csv)
        df_option = pd.read_csv(self.option_csv)

        # Cleanup & Merge
        df_stock.columns = df_stock.columns.str.strip()
        df_option.columns = df_option.columns.str.strip()

        # Fix Dates
        def fix_date(df):
            candidates = ['date', 'trade_date', 'datetime', 'timestamp']
            for c in df.columns:
                if c.lower() in candidates:
                    df.rename(columns={c: 'Date'}, inplace=True)
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df
            return df

        df_stock = fix_date(df_stock)
        df_option = fix_date(df_option)

        # Merge
        self.df = pd.merge(df_stock, df_option, on='Date', suffixes=('', '_opt')).sort_values('Date').reset_index(
            drop=True)

        # Sanitize
        for col in self.df.columns:
            if col != 'Date':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.ffill().bfill().fillna(0.0)

        # Feature Engineering (IV Proxy)
        numeric_opt_cols = df_option.select_dtypes(include=[np.number]).columns.tolist()
        self.iv_col = 'atm_iv_14d' if 'atm_iv_14d' in self.df.columns else numeric_opt_cols[-1]

        self.df['log_ret'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['Realized_Vol_30d'] = self.df['log_ret'].rolling(30).std() * np.sqrt(252)
        self.df['VRP'] = self.df[self.iv_col] - self.df['Realized_Vol_30d']
        self.df = self.df.fillna(0.0)

        # FIT SCALERS
        self.stock_feats = ["Open", "High", "Low", "Close", "Volume", "RSI", "VWAP",
                            "SMA_25", "SMA_50", "SMA_200", "MACD_12_26_9", "VIX", "SPY", "DXY"]
        # Ensure cols exist
        for c in self.stock_feats:
            if c not in self.df.columns: self.df[c] = 0.0

        self.scaler_stock = MinMaxScaler()
        self.scaler_stock.fit(self.df[self.stock_feats].values)

        # Option Scaler
        self.final_opt_cols = [c for c in numeric_opt_cols if c in self.df.columns]
        self.scaler_option = StandardScaler()
        self.scaler_option.fit(self.df[self.final_opt_cols].values)

        # Market Scaler
        market_data = self.df[[self.iv_col, 'Realized_Vol_30d', 'VRP']].values
        self.scaler_market = MinMaxScaler()
        self.scaler_market.fit(market_data)

    def _load_models(self):
        # 1. Stock Model
        ckpt_s = torch.load(self.stock_model_path, map_location=self.device, weights_only=False)
        self.model_s = StockBiLSTM(ckpt_s["input_dim"], ckpt_s["params"]["hidden_dim"],
                                   ckpt_s["params"]["num_layers"], ckpt_s["params"]["dropout"]).to(self.device)
        self.model_s.load_state_dict(ckpt_s["model_state"])
        self.model_s.eval()

        # 2. Option Model
        opt_input_dim = len(self.final_opt_cols)
        self.model_o = QuantileBiLSTM(input_size=opt_input_dim, hidden_size=64, num_layers=2, num_quantiles=3).to(
            self.device)

        opt_ckpt = torch.load(self.option_model_path, map_location=self.device, weights_only=False)
        if isinstance(opt_ckpt, dict) and "model_state" in opt_ckpt:
            self.model_o.load_state_dict(opt_ckpt["model_state"])
        else:
            self.model_o.load_state_dict(opt_ckpt)
        self.model_o.eval()

        # 3. Fusion Model
        ckpt_f = torch.load(self.fusion_model_path, map_location=self.device, weights_only=False)
        self.model_f = FusionMLP(ckpt_f["input_dim"], ckpt_f["params"]["hidden_dim"],
                                 ckpt_f["params"]["num_layers"], ckpt_f["params"]["dropout"]).to(self.device)
        self.model_f.load_state_dict(ckpt_f["model_state"])
        self.model_f.eval()

    def predict(self, target_date_str):
        """
        Runs inference for a specific date.
        Returns: { 'mu': float (annualized), 'sigma': float (annualized), 'aiv': float (raw change) }
        """
        target_date = pd.to_datetime(target_date_str)

        # Find index
        row = self.df[self.df['Date'] == target_date]
        if row.empty:
            raise ValueError(f"Date {target_date_str} not found in data.")

        idx = row.index[0]

        # Validation
        if idx < max(self.seq_len_stock, self.seq_len_option):
            raise ValueError("Not enough history for this date to generate prediction.")

        # Prepare Inputs
        # 1. Stock Sequence (i-30 : i)
        raw_s = self.df[self.stock_feats].iloc[idx - self.seq_len_stock: idx].values
        scaled_s = self.scaler_stock.transform(raw_s)
        tensor_s = torch.tensor(scaled_s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Option Sequence (i-3 : i)
        raw_o = self.df[self.final_opt_cols].iloc[idx - self.seq_len_option: idx].values
        scaled_o = self.scaler_option.transform(raw_o)
        tensor_o = torch.tensor(scaled_o, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 3. Market Vector (i)
        raw_m = self.df[[self.iv_col, 'Realized_Vol_30d', 'VRP']].iloc[idx:idx + 1].values
        scaled_m = self.scaler_market.transform(raw_m)
        tensor_m = torch.tensor(scaled_m, dtype=torch.float32).to(self.device)

        # Inference
        with torch.no_grad():
            emb_s = self.model_s.get_embedding(tensor_s)
            emb_o = self.model_o.get_embedding(tensor_o)

            fused_input = torch.cat([emb_s, emb_o, tensor_m], dim=1)
            prediction = self.model_f(fused_input).cpu().numpy()[0]  # [mu, sigma, aiv]

        # Raw Outputs (10-day horizon)
        raw_mu_10d = prediction[0]
        raw_vol_10d = prediction[1]
        raw_aiv_10d = prediction[2]

        # UNIT CONVERSION
        annual_factor = 252.0 / self.predict_horizon
        mu_annual = raw_mu_10d * annual_factor
        sigma_annual = raw_vol_10d * np.sqrt(252)

        # SAFETY GUARDRAILS (Clip Drift to avoid crazy values)
        mu_annual = float(np.clip(mu_annual, -0.8, 0.8))
        sigma_annual = float(max(sigma_annual, 0.05))

        return {
            "mu": mu_annual,
            "sigma": sigma_annual,
            "aiv": float(raw_aiv_10d),
            "raw_mu_10d": float(raw_mu_10d)
        }
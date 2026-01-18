import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler

# Default input directory
DEFAULT_INPUT_DIR = r"C:\My Documents\Mics\Logs"

#
# This Code Performs Monte Carlo Simulations to Predict TSLA Stock Prices
# using both a NumPy-based Geometric Brownian Motion model and a trained LSTM model
#

# 1. Read CSV file and prepare data
csv_path = os.path.join(DEFAULT_INPUT_DIR, 'tsla_monte.csv')
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')

# 2. Set up simulation parameters
target_date = pd.to_datetime('2025-10-24')
start_date = df.index[-1]
n_days = np.busday_count(start_date.date(), target_date.date())
sims = 60000
start_price = df['Close'][-1]

# 3. Calculate historical log returns for the model
log_returns = np.log(1 + df['Close'].pct_change())
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var)
stdev = log_returns.std()

# 4. Run the NumPy-based Monte Carlo simulation
# Create random values
np.random.seed(42)
Z = np.random.standard_normal((n_days, sims))
# Calculate daily returns
daily_returns = np.exp(drift + stdev * Z)

# Create an array to hold the price paths
price_paths = np.zeros_like(daily_returns)
price_paths[0] = start_price

# Generate the price paths
for t in range(1, n_days):
    price_paths[t] = price_paths[t - 1] * daily_returns[t]

# Convert to a DataFrame for easy analysis and plotting
simulated_prices = pd.DataFrame(price_paths)

# 5. Analyze and plot the results
last_day_prices = simulated_prices.iloc[-1]
predicted_mean = last_day_prices.mean()
print(f"Predicted TSLA close price for {target_date.date()}: ${predicted_mean:.2f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(simulated_prices, color='gray', alpha=0.1)
plt.title(f'Monte Carlo Simulation for TSLA Close Price to {target_date.date()}')
plt.ylabel('Simulated Close Price')
plt.xlabel('Days Ahead')
plt.axhline(predicted_mean, color='red', linestyle='--', label=f'Predicted Mean: ${predicted_mean:.2f}')
plt.legend()
plt.show()

# --- LSTM Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=81, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- LSTM Monte Carlo Simulation ---
# 1. Prepare features and scaler (must match training)
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP', 'SMA_25', 'SMA_50',
    'SMA_100', 'SMA_200', 'WTI', 'VIX', 'DXY', 'SPY', 'MACD_12_26_9'
]
close_idx = feature_cols.index('Close')
features = df[feature_cols].values.astype(np.float32)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

SEQ_LENGTH = 7  # Use the same as in training
MODEL_PATH = os.path.join(DEFAULT_INPUT_DIR, 'tsla_lstm_model_final.pth')

# 2. Load LSTM model
model = LSTMModel(input_size=len(feature_cols), hidden_size=81, num_layers=1, output_size=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 3. Monte Carlo simulation using LSTM
simulated_lstm_prices = np.zeros((n_days, sims))
last_seq = features_scaled[-SEQ_LENGTH:].copy()  # shape (SEQ_LENGTH, num_features)

for sim in range(sims):
    seq = last_seq.copy()
    prices = []
    for t in range(n_days):
        input_tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(input_tensor).item()
        # Optionally add noise for stochasticity (simulate uncertainty)
        pred_scaled_noisy = pred_scaled + np.random.normal(0, 0.01)  # 0.01 is stddev, adjust as needed
        # Build next input sequence
        next_row = seq[-1].copy()
        next_row[close_idx] = pred_scaled_noisy  # Only update 'Close' feature
        seq = np.vstack([seq[1:], next_row])
        # Inverse transform to get price
        next_features = seq[-1].reshape(1, -1)
        next_features_inv = scaler.inverse_transform(next_features)
        prices.append(next_features_inv[0, close_idx])
    simulated_lstm_prices[:, sim] = prices

# 4. Analyze and plot the results
last_day_prices = simulated_lstm_prices[-1]
predicted_mean = last_day_prices.mean()
print(f"[LSTM] Predicted TSLA close price for {target_date.date()}: ${predicted_mean:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(simulated_lstm_prices, color='gray', alpha=0.1)
plt.title(f'LSTM Monte Carlo Simulation for TSLA Close Price to {target_date.date()}')
plt.ylabel('Simulated Close Price')
plt.xlabel('Days Ahead')
plt.axhline(predicted_mean, color='red', linestyle='--', label=f'Predicted Mean: ${predicted_mean:.2f}')
plt.legend()
plt.show()

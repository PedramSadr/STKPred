import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

# --- Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=46, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Config ---
SEQ_LENGTH = 2
MODEL_PATH = r'C:\My Documents\Mics\Logs\tsla_lstm_model_best.pth'
CSV_PATH = r'C:\My Documents\Mics\Logs\tsla_daily_test.csv'

# --- Data Loading ---
df = pd.read_csv(CSV_PATH)
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP', 'SMA_25', 'SMA_50',
    'SMA_100', 'SMA_200', 'WTI', 'VIX', 'DXY', 'SPY', 'MACD_12_26_9'
]
close_idx = feature_cols.index('Close')
features = df[feature_cols].values.astype(np.float32)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

def create_sequences(data, seq_length):
    xs, ys, idxs = [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length][close_idx])
        idxs.append(i + seq_length)
    return np.array(xs), np.array(ys), np.array(idxs)

X, y, idxs = create_sequences(features_scaled, SEQ_LENGTH)

# --- Model Loading ---
device = torch.device('cpu')
model = LSTMModel(input_size=len(feature_cols), hidden_size=46, num_layers=1, output_size=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Prediction ---
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    print("X_tensor shape:", X_tensor.shape)
    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(0)
    elif X_tensor.ndim == 1:
        X_tensor = X_tensor.unsqueeze(0).unsqueeze(0)
    preds = model(X_tensor).numpy()

    preds = model(X_tensor).numpy()

# --- Inverse Transform ---
close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']].values)
preds_inv = close_scaler.inverse_transform(preds)
y_inv = close_scaler.inverse_transform(y.reshape(-1, 1))

# --- Plot ---
df['Date'] = pd.to_datetime(df['Date'])
plot_dates = df['Date'].iloc[idxs].values

# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Date': plot_dates,
    'Actual': y_inv.flatten(),
    'Predicted': preds_inv.flatten()
})

# Group by Date and average values to ensure one data point per day for cleaner plotting
plot_df = plot_df.groupby('Date').mean().reset_index()

plt.figure(figsize=(14, 6))
ax = plt.gca() # Get the current axes object
ax.plot(plot_df['Date'], plot_df['Actual'], label='Actual Close', color='blue')
ax.plot(plot_df['Date'], plot_df['Predicted'], label='Predicted Close', color='red', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Actual vs Predicted Close Price')
ax.legend()
plt.tight_layout()

# Option 1: Show a tick every 7 days (adjust the interval as needed)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)) # Set interval to 7 for weekly ticks, adjust as needed
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.show()

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=6, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

SEQ_LENGTH = 60
MODEL_PATH = r'C:\My Documents\Mics\Logs\model.pth'
CSV_PATH = r'C:\My Documents\Mics\Logs\tsla_daily.csv'

df = pd.read_csv(CSV_PATH)
# Use the same 8 features as in training
feature_cols = ['Year', 'Month', 'Day', 'Open', 'Hi', 'Low', 'Close', 'Volume']  # adjust as needed
features = df[feature_cols].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

def create_sequences(data, seq_length):
    xs, ys, idxs = [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length][3])  # Assuming 'Close' is at index 3
        idxs.append(i + seq_length)
    return np.array(xs), np.array(ys), np.array(idxs)

X, y, idxs = create_sequences(features_scaled, SEQ_LENGTH)

device = torch.device('cpu')
model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = model(X_tensor).numpy()

# Inverse transform only the 'Close' column for plotting
close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']].values)
preds_inv = close_scaler.inverse_transform(preds)
y_inv = close_scaler.inverse_transform(y.reshape(-1, 1))

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
plot_dates = df['Date'].iloc[idxs].values

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], label='Actual Close (All)', color='gray', alpha=0.3)
plt.plot(plot_dates, y_inv, label='Actual', color='blue')
plt.plot(plot_dates, preds_inv, label='Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('LSTM Model: Actual vs Predicted Prices (All Data)')
plt.legend()
plt.tight_layout()
plt.show()
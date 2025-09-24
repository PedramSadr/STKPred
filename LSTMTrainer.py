import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --- Step 1: Detect and set device to CUDA if available, otherwise CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data and handle 'Date' Keyerror
df = pd.read_csv(r'C:\My Documents\Mics\Logs\tsla_daily.csv')
df.columns = df.columns.str.strip().str.lower()

# --- FIX: Drop any rows containing NaN values from the entire DataFrame ---
df = df.dropna()

# Parse date column
dates = pd.to_datetime(df['date'])

prices = df['close'].values.astype(np.float32)

# Save the mean and std of the prices before scaling
price_mean = prices.mean()
price_std = prices.std()

# --- Filter out zero-variance columns before normalization ---
feature_cols = ['close', 'high', 'low', 'open', 'volume', 'rsi', 'sma_50', 'sma_100', 'sma_200', 'macd_12_26_9']
features_df = df[feature_cols].copy()

stds = features_df.std()
zero_std_cols = stds[stds < 1e-6].index

if not zero_std_cols.empty:
    print(f"Warning: Removing zero-variance features: {list(zero_std_cols)}")
    features_df = features_df.drop(columns=zero_std_cols)
    feature_cols = features_df.columns.tolist()

features = features_df.values.astype(np.float32)

# Normalize features
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Normalize the target variable (prices)
normalized_prices = (prices - price_mean) / price_std

sequence_length = 30
# Update input_size to match the number of remaining features
input_size = len(feature_cols)
hidden_size = 60
output_size = 1

def create_sequences(features, prices, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:i+seq_length])
        ys.append(prices[i+seq_length])
    return np.array(xs), np.array(ys)

X, y_normalized = create_sequences(features, normalized_prices, sequence_length)

# --- Move tensors to the selected device ---
X_tensor = torch.tensor(X).to(device)
y_tensor = torch.tensor(y_normalized).unsqueeze(1).to(device)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMNet(input_size, hidden_size, output_size)

# --- Step 2: Move the model to the selected device ---
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 300
for epoch in range(epochs):
    for xb, yb in loader:
        # --- Step 3: Move batch data to the selected device ---
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    # Retrieve the predictions back to the CPU for numpy conversion and plotting
    preds_normalized = model(X_tensor).squeeze().cpu().numpy()
    preds = (preds_normalized * price_std) + price_mean

# The original y_normalized was never a tensor, so no need for .cpu()
# Just squeeze it to match the dimensions of preds for plotting.
y_actual = (y_normalized.squeeze() * price_std) + price_mean

plot_dates = dates[sequence_length:]
print(len(plot_dates), len(y_actual), len(preds))

plt.figure(figsize=(12,6))
plt.plot(plot_dates, y_actual, label='Actual Price', color='blue')
plt.plot(plot_dates, preds, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.title('Actual vs Predicted Close Price')
plt.show()

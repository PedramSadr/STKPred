import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --- Step 1: Detect device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Step 2: Load validation data ---
val_df = pd.read_csv(r'C:\My Documents\Mics\Logs\tsla_daily_validation.csv')
val_df.columns = val_df.columns.str.strip().str.lower()
val_df = val_df.dropna()

# --- Step 3: Load training stats for normalization ---
train_df = pd.read_csv(r'C:\My Documents\Mics\Logs\tsla_daily.csv')
train_df.columns = train_df.columns.str.strip().str.lower()
train_df = train_df.dropna()
train_prices = train_df['close'].values.astype(np.float32)
price_mean = train_prices.mean()
price_std = train_prices.std()

feature_cols = [
    'open', 'high', 'low', 'close', 'volume', 'rsi', 'vwap', 'sma_25', 'sma_50',
    'sma_100', 'sma_200', 'wti', 'vix', 'dxy', 'spy', 'macd_12_26_9'
]
features_df = val_df[feature_cols].copy()
stds = features_df.std()
zero_std_cols = stds[stds < 1e-6].index
if not zero_std_cols.empty:
    print(f"Warning: Removing zero-variance features: {list(zero_std_cols)}")
    features_df = features_df.drop(columns=zero_std_cols)
    feature_cols = features_df.columns.tolist()

features = features_df.values.astype(np.float32)
features = (features - features.mean(axis=0)) / features.std(axis=0)
prices = val_df['close'].values.astype(np.float32)
normalized_prices = (prices - price_mean) / price_std

sequence_length = 50
input_size = len(feature_cols)
hidden_size = 30
output_size = 1

def create_sequences(features, prices, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:i + seq_length])
        ys.append(prices[i + seq_length])
    return np.array(xs), np.array(ys)

X, y_normalized = create_sequences(features, normalized_prices, sequence_length)
X_tensor = torch.tensor(X).to(device)
y_tensor = torch.tensor(y_normalized).unsqueeze(1).to(device)
val_dataset = TensorDataset(X_tensor, y_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Step 4: Define LSTM Model ---
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
model.load_state_dict(torch.load(r'C:\My Documents\Mics\Logs\tsla_lstm_model.pth', map_location=device))
model.to(device)
model.eval()

criterion = nn.MSELoss()

# --- Step 5: Evaluate model and plot loss per epoch ---
epochs = 500
val_epoch_losses = []
with torch.no_grad():
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / len(val_dataset)
        val_epoch_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_loss:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), val_epoch_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()


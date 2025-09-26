import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --- Step 1: Detect device, load and process data ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data and convert all column names to lowercase for consistency
df = pd.read_csv(r'C:\My Documents\Mics\Logs\tsla_daily.csv')
df.columns = df.columns.str.strip().str.lower()

# Drop any rows containing NaN values
df = df.dropna()

# Define the target variable (what we want to predict)
prices = df['close'].values.astype(np.float32)

# Save the mean and std of the prices for scaling
price_mean = prices.mean()
price_std = prices.std()

# Define the feature columns for the model
feature_cols = [
    'open', 'high', 'low', 'close', 'volume', 'rsi', 'vwap', 'sma_25', 'sma_50',
    'sma_100', 'sma_200', 'wti', 'vix', 'dxy', 'spy', 'macd_12_26_9'
]
features_df = df[feature_cols].copy()

# Filter out zero-variance columns (good practice)
stds = features_df.std()
zero_std_cols = stds[stds < 1e-6].index
if not zero_std_cols.empty:
    print(f"Warning: Removing zero-variance features: {list(zero_std_cols)}")
    features_df = features_df.drop(columns=zero_std_cols)
    feature_cols = features_df.columns.tolist()

features = features_df.values.astype(np.float32)

# Normalize features and target
features = (features - features.mean(axis=0)) / features.std(axis=0)
normalized_prices = (prices - price_mean) / price_std

# --- Create sequences for the LSTM model ---
sequence_length = 50
input_size = len(feature_cols)
hidden_size = 60
output_size = 1

def create_sequences(features, prices, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:i + seq_length])
        ys.append(prices[i + seq_length])
    return np.array(xs), np.array(ys)

X, y_normalized = create_sequences(features, normalized_prices, sequence_length)

# Move tensors to the selected device
X_tensor = torch.tensor(X).to(device)
y_tensor = torch.tensor(y_normalized).unsqueeze(1).to(device)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define LSTM Model
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
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- Step 2: Train the model ---
epochs = 170
epoch_losses = []  # List to store loss per epoch
for epoch in range(epochs):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Store the loss from the last batch of the epoch
    epoch_losses.append(loss.item())
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# --- Step 3: Save the trained model ---
# <-- FIX: Added 'r' to create a raw string and fix the file path
save_path = r'C:\My Documents\Mics\Logs\tsla_lstm_model.pth'
torch.save(model.state_dict(), save_path)
print(f"\nModel training complete. Saved to {save_path}")

# --- Step 4: Plot Training Loss vs. Epoch ---
plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
csv_path = r'C:\My Documents\Mics\Logs\tsla_daily.csv'
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.lower()
df = df.dropna()

prices = df['close'].values.astype(np.float32)
price_mean = prices.mean()
price_std = prices.std()

feature_cols = [
    'open', 'high', 'low', 'close', 'volume', 'rsi', 'vwap', 'sma_25', 'sma_50',
    'sma_100', 'sma_200', 'wti', 'vix', 'dxy', 'spy', 'macd_12_26_9'
]
features_df = df[feature_cols].copy()
stds = features_df.std()
zero_std_cols = stds[stds < 1e-6].index
if not zero_std_cols.empty:
    print(f"Warning: Removing zero-variance features: {list(zero_std_cols)}")
    features_df = features_df.drop(columns=zero_std_cols)
    feature_cols = features_df.columns.tolist()
features = features_df.values.astype(np.float32)
features = (features - features.mean(axis=0)) / features.std(axis=0)
normalized_prices = (prices - price_mean) / price_std

# Split train/val
train_features, val_features, train_prices, val_prices = train_test_split(
    features, normalized_prices, test_size=0.2, shuffle=False)

# Sequence creation
sequence_length = 32

def create_sequences(features, prices, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:i + seq_length])
        ys.append(prices[i + seq_length])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_features, train_prices, sequence_length)
X_val, y_val = create_sequences(val_features, val_prices, sequence_length)

X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)
X_val_tensor = torch.tensor(X_val).to(device)
y_val_tensor = torch.tensor(y_val).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=95, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=95, shuffle=False)

# Model definition
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out

input_size = len(feature_cols)
hidden_size = 52
output_size = 1
model = LSTMNet(input_size, hidden_size, output_size)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop
epochs = 800
patience = 20
min_delta = 1e-5
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            vloss = criterion(pred, yb)
            val_loss += vloss.item() * xb.size(0)
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Early stopping
    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

print(f"Best validation loss: {best_val_loss:.6f}")
torch.save(best_model_state, r'C:\My Documents\Mics\Logs\tsla_lstm_model_best.pth')

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Best Hyperparameters Training')
plt.legend()
plt.grid(True)
plt.show()


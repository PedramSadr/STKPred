import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

#
#This Code Performs Hyperparameter Optimization for a Bidirectional LSTM Model on TSLA Stock Data
#This code trians multiple BiLSTM models with different hyperparameters to find the best configuration(With the lowest validation loss & Trail = 3000)
#
# --- Step 1: Detect device, load and process data ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data and convert all column names to lowercase for consistency
df = pd.read_csv(r'C:\My Documents\Mics\Logs\tsla_daily.csv')
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

input_size = len(feature_cols)
output_size = 1

train_features, val_features, train_prices, val_prices = train_test_split(
    features, normalized_prices, test_size=0.2, shuffle=False)

def create_sequences(features, prices, seq_length):
    xs, ys = [], []
    if len(features) <= seq_length:
        return np.array([]), np.array([])
    for i in range(len(features) - seq_length):
        xs.append(features[i:i + seq_length])
        ys.append(prices[i + seq_length])
    return np.array(xs), np.array(ys)

# Hyperparameter search space
hidden_size_range = (30, 90)
sequence_length_range = (14, 50)
learning_rate_range = (0.001, 0.01)
epochs_range = (50, 500) # Adjusted for reasonable training times
trials = 3000 # Adjusted for reasonable training times

batch_size_range = (32, 128)
best_loss = float('inf')
best_params = None
best_model_path = r'C:\My Documents\Mics\Logs\tsla_bilstm_model_best.pth'

# --- MODIFIED CODE: Define the Bidirectional LSTM model class ---
# This class is now defined once, outside the loop.
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Change 1: Add bidirectional=True to the LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        # Change 2: The linear layer now takes hidden_size * 2 as input
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # The output from the last time step is fed to the linear layer
        out = self.fc(out[:, -1, :])
        return out

for trial in range(trials):
    hidden_size = random.randint(*hidden_size_range)
    sequence_length = random.randint(*sequence_length_range)
    learning_rate = 10 ** random.uniform(np.log10(learning_rate_range[0]), np.log10(learning_rate_range[1]))
    epochs = random.randint(*epochs_range)
    batch_size = random.randint(*batch_size_range)
    patience = 20
    min_delta = 1e-5

    print(f"\nTrial {trial+1}/{trials}: hidden={hidden_size}, seq_len={sequence_length}, lr={learning_rate:.5f}, epochs={epochs}, batch={batch_size}")

    X_train, y_train = create_sequences(train_features, train_prices, sequence_length)
    X_val, y_val = create_sequences(val_features, val_prices, sequence_length)

    if len(X_val) == 0:
        print("Skipping trial, validation set is too small for this sequence length.")
        continue

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Use the new BiLSTMNet class
    model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_for_trial = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                epoch_val_loss += criterion(pred, yb).item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state_for_trial = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation loss for this trial: {best_val_loss:.6f}")
    if best_val_loss < best_loss:
        best_loss = best_val_loss
        best_params = {
            'hidden_size': hidden_size, 'sequence_length': sequence_length,
            'learning_rate': learning_rate, 'epochs_run': epoch+1, 'batch_size': batch_size
        }
        torch.save(best_model_state_for_trial, best_model_path)
        print(f"--- New best model saved with loss: {best_loss:.6f} ---")

print("\n--- Hyperparameter Search Complete ---")
print("Best hyperparameters found:")
for k, v in best_params.items():
    print(f"{k}: {v}")
print(f"Best validation loss across all trials: {best_loss:.6f}")

# --- Load the BEST model state before saving and plotting ---
print(f"Loading best model from {best_model_path} for final save and plot.")
# Use the BiLSTMNet class to instantiate the final model
final_model = BiLSTMNet(input_size, best_params['hidden_size'], output_size)
final_model.load_state_dict(torch.load(best_model_path))

save_path = r'C:\My Documents\Mics\Logs\tsla_bilstm_model_final.pth'
torch.save(final_model.state_dict(), save_path)
print(f"\nFinal best model saved to {save_path}")

# --- Plotting Best Model Predictions ---
print("\n--- Plotting Best Model Predictions ---")
best_model = final_model # Use the already loaded final_model
best_model.to(device)
best_model.eval()

best_seq_length = best_params['sequence_length']
X_val_plot, y_val_plot = create_sequences(val_features, val_prices, best_seq_length)

if len(X_val_plot) > 0:
    X_val_plot_tensor = torch.tensor(X_val_plot, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions_normalized = best_model(X_val_plot_tensor)

    predictions_normalized = predictions_normalized.cpu().numpy()
    predicted_prices = (predictions_normalized * price_std) + price_mean
    actual_prices = (y_val_plot * price_std) + price_mean

    plt.figure(figsize=(15, 7))
    plt.plot(actual_prices, label='Actual Price', color='blue', linewidth=2)
    plt.plot(predicted_prices, label='Predicted Price', color='red', linestyle='--', linewidth=2)
    plt.title('Best BiLSTM Model: Actual vs. Predicted Prices')
    plt.xlabel('Time Step (in validation set)')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not generate plot because the validation set was too small for the best sequence length.")
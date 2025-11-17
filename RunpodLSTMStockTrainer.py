# python
#This script performs hyperparameter optimization for a Bidirectional LSTM model
#to predict TSLA stock prices using historical stock data.
#This code is adapted to run on Runpod with multi-GPU support and improved data loading.
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
import os

# File: `RunpodLSTMStockTrainer.py`
# Detect device and GPU count
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
multi_gpu = n_gpus > 1
print(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}, GPU count: {n_gpus}")

# Optional performance flags
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Load data and convert all column names to lowercase for consistency
df = pd.read_csv(r'/workspace/lstm-project/data/tsla_daily.csv')
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
# normalize per-feature
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
epochs_range = (50, 500)
trials = 3000

batch_size_range = (32, 128)
best_loss = float('inf')
best_params = None
best_model_path = r'/workspace/output/tsla_dailytsla_bilstm_model_best.pth'

# Define the Bidirectional LSTM model class
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
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

    # Create tensors on CPU so DataLoader can use pin_memory (only CPU tensors can be pinned)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # DataLoader improvements for CUDA
    pin_memory = True if device.type == "cuda" else False
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=0)

    # Create model and enable multi-GPU with DataParallel if available
    model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
    if multi_gpu:
        try:
            model = nn.DataParallel(model)
            print("Wrapped model with nn.DataParallel")
            # Quick sanity check: run a tiny forward to detect Scatter/Gather/ECC issues early
            seq_test_len = max(1, min(5, sequence_length))
            try:
                with torch.no_grad():
                    test_input = torch.zeros((1, seq_test_len, input_size), dtype=torch.float32).to(device)
                    _ = model(test_input)
            except Exception as e:
                print("DataParallel quick-forward test failed, unwrapping to single-GPU. Error:\n", e)
                # unwrap to single GPU
                state = model.module.state_dict() if hasattr(model, 'module') else None
                multi_gpu = False
                model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
                if state is not None:
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        # state likely not compatible at this stage; continue with fresh model
                        pass
        except Exception as e:
            print("Failed to wrap model with DataParallel, falling back to single GPU. Error:\n", e)
            multi_gpu = False
            model = BiLSTMNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_for_trial = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            # Move each batch to the device here (supports pin_memory optimizations)
            xb = xb.to(device)
            yb = yb.to(device)
            try:
                pred = model(xb)
            except Exception as e:
                # Detect CUDA/DataParallel related errors (e.g., ECC issues) and fallback to single-GPU
                msg = str(e)
                if multi_gpu and ("CUDA" in msg or "cuda" in msg or "ECC" in msg or isinstance(e, RuntimeError)):
                    print("CUDA/DataParallel error detected during forward; falling back to single-GPU. Error:\n", e)
                    # Extract underlying state dict (if model was DataParallel)
                    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    # Disable multi-GPU and recreate model on single device
                    multi_gpu = False
                    model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
                    model.load_state_dict(state)
                    # Recreate optimizer/scheduler bound to new model parameters
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
                    print("Model unwrapped to single-GPU and optimizer/scheduler recreated. Continuing training.")
                    pred = model(xb)
                else:
                    raise

            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                try:
                    pred = model(xb)
                except Exception as e:
                    msg = str(e)
                    if multi_gpu and ("CUDA" in msg or "cuda" in msg or "ECC" in msg or isinstance(e, RuntimeError)):
                        print("CUDA/DataParallel error detected during validation forward; falling back to single-GPU. Error:\n", e)
                        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                        multi_gpu = False
                        model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
                        model.load_state_dict(state)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
                        print("Model unwrapped to single-GPU for validation.")
                        pred = model(xb)
                    else:
                        raise

                epoch_val_loss += criterion(pred, yb).item()

        # Safeguard: avoid division by zero if train_loader/val_loader empty
        if len(train_loader) == 0:
            avg_train_loss = float('nan')
        else:
            avg_train_loss = epoch_train_loss / len(train_loader)

        if len(val_loader) == 0:
            avg_val_loss = float('nan')
        else:
            avg_val_loss = epoch_val_loss / len(val_loader)

        # Step scheduler only when we have a numeric val loss
        if not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # store state_dict of underlying model (unwrap DataParallel)
            if hasattr(model, "module"):
                best_model_state_for_trial = model.module.state_dict()
            else:
                best_model_state_for_trial = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation loss for this trial: {best_val_loss:.6f}")
    # record how many epochs were actually run in this trial
    epochs_run = epoch + 1
    if best_val_loss < best_loss and best_model_state_for_trial is not None:
        best_loss = best_val_loss
        best_params = {
            'hidden_size': hidden_size, 'sequence_length': sequence_length,
            'learning_rate': learning_rate, 'epochs_run': epochs_run, 'batch_size': batch_size
        }
        # Save the unwrapped state dict
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(best_model_state_for_trial, best_model_path)
        print(f"--- New best model saved with loss: {best_loss:.6f} ---")

print("\n--- Hyperparameter Search Complete ---")
if best_params is None:
    print("No valid model was found.")
else:
    print("Best hyperparameters found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"Best validation loss across all trials: {best_loss:.6f}")

    # Load the BEST model state before saving and plotting
    print(f"Loading best model from {best_model_path} for final save and plot.")
    final_model = BiLSTMNet(input_size, best_params['hidden_size'], output_size)
    # Wrap if multi-GPU (for consistent forward behavior) and move to device
    if multi_gpu:
        final_model = nn.DataParallel(final_model)
    final_model.to(device)

    # load saved state dict (saved unwrapped)
    map_loc = torch.device('cpu') if device.type == 'cpu' else device
    state = torch.load(best_model_path, map_location=map_loc)
    if hasattr(final_model, "module"):
        final_model.module.load_state_dict(state)
    else:
        final_model.load_state_dict(state)

    save_path = r'C:\My Documents\Mics\Logs\tsla_bilstm_model_final.pth'
    # save unwrapped state_dict for portability
    final_state_to_save = final_model.module.state_dict() if hasattr(final_model, "module") else final_model.state_dict()
    torch.save(final_state_to_save, save_path)
    print(f"\nFinal best model saved to {save_path}")

    # Plotting Best Model Predictions
    print("\n--- Plotting Best Model Predictions ---")
    final_model.eval()

    best_seq_length = best_params['sequence_length']
    X_val_plot, y_val_plot = create_sequences(val_features, val_prices, best_seq_length)

    if len(X_val_plot) > 0:
        X_val_plot_tensor = torch.tensor(X_val_plot, dtype=torch.float32).to(device)

        with torch.no_grad():
            predictions_normalized = final_model(X_val_plot_tensor)

        predictions_normalized = predictions_normalized.cpu().numpy().squeeze()
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
        os.makedirs(r'/workspace/lstm-project/output', exist_ok=True)
        plt.savefig(r'/workspace/lstm-project/output/plot1.png')
    else:
        print("Could not generate plot because the validation set was too small for the best sequence length.")
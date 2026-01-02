import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

# --- Configuration ---
FILE_PATH = r"C:\My Documents\Mics\Logs\TSLA_Surface_Vector_Merged.csv"
SAVE_PATH = r"C:\My Documents\Mics\Logs\Final_TSLA_Quantile_Model.pth"
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 30
LR = 0.001

# Architecture Params
SEQ_LENGTH = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
QUANTILES = [0.1, 0.5, 0.9]  # P10, P50 (Median), P90

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Custom Quantile Loss ---
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i + 1]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss


# --- Model Architecture ---
class QuantileBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_quantiles):
        super(QuantileBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)

        # We predict 'num_quantiles' values instead of just 1
        self.fc = nn.Linear(hidden_size * 2, num_quantiles)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_and_verify():
    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    data = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    # Target is still index -1 (iv_change_1d)
    target_col_idx = -1

    for i in range(len(data_scaled) - SEQ_LENGTH):
        X.append(data_scaled[i: i + SEQ_LENGTH])
        y.append(data_scaled[i + SEQ_LENGTH, target_col_idx])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_val = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
    y_train, y_val = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    print(f"Training Quantile Regression Model...")

    model = QuantileBiLSTM(X.shape[2], HIDDEN_SIZE, NUM_LAYERS, len(QUANTILES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = QuantileLoss(QUANTILES)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)

        avg_val_loss = val_loss / len(X_val)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Val Quantile Loss: {avg_val_loss:.4f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save Best
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Best Model Saved. Loss: {best_loss:.4f}")

    # --- Visualization ---
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).cpu().numpy()
        actuals = y_val.numpy()

    # Plot last 100 days
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[-100:], label='Actual', color='black')
    plt.plot(preds[-100:, 1], label='Median Prediction', color='blue', linestyle='--')  # 50th percentile
    plt.fill_between(range(100), preds[-100:, 0], preds[-100:, 2], color='blue', alpha=0.2,
                     label='10th-90th Confidence')
    plt.title("Quantile Forecast (Last 100 Days)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_and_verify()
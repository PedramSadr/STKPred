import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import copy

# --- Configuration ---
FILE_PATH = r"C:\My Documents\Mics\Logs\TSLA_Surface_Vector_Merged.csv"
SAVE_PATH = r"C:\My Documents\Mics\Logs\Final_TSLA_Huber_Model.pth"
BATCH_SIZE = 300
EPOCHS = 300
PATIENCE = 30

# Optimal Params
SEQ_LENGTH = 3
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

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

    print(f"Training Started with Huber Loss...")

    model = BiLSTM(X.shape[2], HIDDEN_SIZE, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- CHANGED: Using Huber Loss (delta=1.0 is standard) ---
    criterion = nn.HuberLoss(delta=1.0)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    best_epoch = 0

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                # Calculate Huber Loss for validation
                l = criterion(model(xb), yb)
                val_loss += l.item() * xb.size(0)

        avg_val_loss = val_loss / len(X_val)
        loss_history.append(avg_val_loss)

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            best_epoch = epoch
        else:
            no_improve_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Val Huber Loss: {avg_val_loss:.6f} | Best: {best_loss:.6f}")

        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}. Best model was at epoch {best_epoch + 1}.")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nBest Model Saved to: {SAVE_PATH}")
    print(f"Best Validation Huber Loss: {best_loss:.4f}")

    # --- Final Verification (Calculates MSE for apples-to-apples comparison) ---
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).cpu().numpy()
        actuals = y_val.numpy()

    mse = mean_squared_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print("-" * 30)
    print(f"FINAL METRICS (Verification)")
    print("-" * 30)
    print(f"MSE (Scaled): {mse:.4f} (Compare this to 0.752)")
    print(f"R-Squared:    {r2:.4f}")
    print("-" * 30)

    plt.figure(figsize=(12, 6))
    plt.plot(actuals[-100:], label='Actual', color='black', alpha=0.7)
    plt.plot(preds[-100:], label='Predicted', color='green')
    plt.title(f"Huber Loss Training Results\nR2: {r2:.4f} | MSE: {mse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    train_and_verify()
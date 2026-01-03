import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import sys

# --- 1. FORCE DESKTOP PATH (The Fix) ---
# This finds your actual Desktop path regardless of your username
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "AI_TRADING_MODELS")

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the absolute path for the saved model
SAVE_FILENAME = "TSLA_14Day_Specialist.pth"
SAVE_PATH = os.path.join(output_folder, SAVE_FILENAME)

print("\n" + "!" * 50)
print(f"OUTPUT WILL BE SAVED TO: {SAVE_PATH}")
print("!" * 50 + "\n")

# --- Configuration ---
# INPUT: Keep reading from My Documents
FILE_PATH = r"C:\My Documents\Mics\Logs\TSLA_Surface_Vector_Merged.csv"
# BRAIN: Point to your Quantile Model
PRETRAINED_PATH = r"C:\My Documents\Mics\Logs\Final_TSLA_Quantile_Model.pth"

BATCH_SIZE = 128
EPOCHS = 100
LR = 0.001
SEQ_LENGTH = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FORECAST_HORIZON = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Architecture ---
class Hybrid14DayModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=14):
        super(Hybrid14DayModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        features = out[:, -1, :]
        prediction = self.mlp(features)
        return prediction


def run_pipeline():
    # Check Inputs
    if not os.path.exists(FILE_PATH):
        print(f"CRITICAL ERROR: Input CSV not found at {FILE_PATH}")
        return

    print("Loading data...")
    df = pd.read_csv(FILE_PATH)
    data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    target_col_idx = -1

    for i in range(len(data_scaled) - SEQ_LENGTH - FORECAST_HORIZON + 1):
        X.append(data_scaled[i: i + SEQ_LENGTH])
        y.append(data_scaled[i + SEQ_LENGTH: i + SEQ_LENGTH + FORECAST_HORIZON, target_col_idx])

    X = np.array(X)
    y = np.array(y)

    # Train Setup
    split = int(0.8 * len(X))
    X_train = torch.FloatTensor(X[:split])
    y_train = torch.FloatTensor(y[:split])
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing Model...")
    model = Hybrid14DayModel(X.shape[2], HIDDEN_SIZE, NUM_LAYERS).to(device)

    # Load Brain
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading Brain from: {PRETRAINED_PATH}")
        pretrained_dict = torch.load(PRETRAINED_PATH)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'lstm' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        for name, param in model.named_parameters():
            if 'lstm' in name: param.requires_grad = False
    else:
        print(f"WARNING: Brain not found at {PRETRAINED_PATH}. Training from scratch.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.HuberLoss()

    print("Training (Fast)...")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # --- THE CRITICAL SAVE STEP ---
    print("\nAttempting to save to DESKTOP...")
    try:
        torch.save(model.state_dict(), SAVE_PATH)

        # Verify
        if os.path.exists(SAVE_PATH):
            print("\n" + "#" * 60)
            print("SUCCESS! FILE CREATED ON YOUR DESKTOP.")
            print(f"FOLDER: {output_folder}")
            print(f"FILE:   {SAVE_FILENAME}")
            print("#" * 60 + "\n")
        else:
            print("ERROR: Python said save worked, but file is missing.")

    except Exception as e:
        print(f"CRITICAL SAVE ERROR: {e}")

    # Forecast
    model.eval()
    last_seq = data_scaled[-SEQ_LENGTH:]
    input_tensor = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        forecast = model(input_tensor).cpu().numpy()[0]

    # Chart
    plt.plot(forecast, marker='o', color='green')
    plt.title("Training Complete - Check Desktop for File")
    plt.show()


if __name__ == "__main__":
    run_pipeline()
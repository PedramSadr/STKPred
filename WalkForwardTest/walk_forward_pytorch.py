import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
# FIX: Point to the absolute path where surface_generator.py saved the file
DATA_PATH = r"C:\My Documents\Mics\Logs\TSLA_Surface_Vector_Merged.csv"

TRAIN_START_SIZE = 500  # Initial training window
TEST_WINDOW_SIZE = 20  # Re-train every Month
LOOKBACK = 10  # Time steps for LSTM
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. PYTORCH MODEL DEFINITION ---
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)


# --- 2. HELPER FUNCTIONS ---
def prepare_lstm_data(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    best_loss = float('inf')
    patience = 3
    trigger_times = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                return


# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"1. Loading Data from {DATA_PATH} on {DEVICE}...")

    # Check if file exists before crashing
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CRITICAL ERROR: File not found at {DATA_PATH}. Did you run surface_generator.py?")

    df = pd.read_csv(DATA_PATH)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)

    target_col = 'iv_change_1d'
    feature_cols = [c for c in df.columns if c not in ['trade_date', 'feature_date', target_col]]

    print(f"Features: {len(feature_cols)} | Target: {target_col}")

    results = []
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # --- THE WALK-FORWARD LOOP ---
    for i in range(TRAIN_START_SIZE, len(df) - TEST_WINDOW_SIZE, TEST_WINDOW_SIZE):

        train_df = df.iloc[:i]
        test_df = df.iloc[i: i + TEST_WINDOW_SIZE + LOOKBACK]

        # Scaling
        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df[[target_col]].values
        X_test_raw = test_df[feature_cols].values
        y_test_raw = test_df[[target_col]].values

        X_train_scaled = scaler_X.fit_transform(X_train_raw)
        y_train_scaled = scaler_y.fit_transform(y_train_raw)
        X_test_scaled = scaler_X.transform(X_test_raw)

        # Sequences
        train_data_combined = np.hstack((y_train_scaled, X_train_scaled))
        X_train_seq, y_train_seq = prepare_lstm_data(train_data_combined, LOOKBACK)

        test_data_combined = np.hstack((scaler_y.transform(y_test_raw), X_test_scaled))
        X_test_seq, _ = prepare_lstm_data(test_data_combined, LOOKBACK)

        # Loader
        train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Train
        model = BiLSTMModel(input_size=X_train_seq.shape[2]).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"Step {i}: Training on {len(X_train_seq)} samples... predicting next {TEST_WINDOW_SIZE} days.")
        train_model(model, train_loader, criterion, optimizer, EPOCHS)

        # Predict
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test_seq).to(DEVICE)
            preds_scaled = model(X_test_tensor).cpu().numpy()

        preds_real = scaler_y.inverse_transform(preds_scaled)

        # Store
        current_test_dates = df.iloc[i: i + len(preds_real)]['trade_date'].values
        actuals = df.iloc[i: i + len(preds_real)][target_col].values

        for j in range(len(preds_real)):
            results.append({
                'date': current_test_dates[j],
                'actual': actuals[j],
                'prediction': preds_real[j][0],
                'train_cutoff': train_df['trade_date'].iloc[-1]
            })

    # Save to LOGS folder (Keep it clean)
    output_path = os.path.join(r"C:\My Documents\Mics\Logs", "Walk_Forward_Results_Pytorch.csv")
    results_df = pd.DataFrame(results)
    results_df['error'] = results_df['actual'] - results_df['prediction']
    results_df.to_csv(output_path, index=False)

    print("\n--- WALK FORWARD COMPLETE ---")
    print(f"Saved results to: {output_path}")
    print(f"Mean Absolute Error: {results_df['error'].abs().mean():.4f}")
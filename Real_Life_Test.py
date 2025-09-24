
import pandas as pd
import torch
import numpy as np
from YahooDownLoader import LSTMModel, scaler, seq_length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'C:\My Documents\Mics\Logs\test_data.csv')
df = df.sort_values(['Year', 'Month', 'Day'])

features = ['Year', 'Month', 'Day', 'Close', 'Hi', 'Low', 'Open', 'Volume']

# Prepare the last sequence and set the date to 2025-09-15
input_data = df[features].values[-seq_length:].copy()
input_data[-1, 0] = 2025  # Year
input_data[-1, 1] = 9     # Month
input_data[-1, 2] = 12    # Day

input_scaled = scaler.transform(input_data)
input_scaled = input_scaled.reshape(1, seq_length, len(features))
input_tensor = torch.from_numpy(input_scaled).float().to(device)

model = LSTMModel().to(device)
model.load_state_dict(torch.load(r'C:\My Documents\Mics\Logs\model.pth', map_location=device))
model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
    last_row = input_data[-1].copy()
    last_row[3] = prediction.cpu().numpy()[0, 0]  # Set predicted Close
    last_row = last_row.reshape(1, -1)
    inv_row = scaler.inverse_transform(last_row)[0]
    predicted_close = inv_row[3]  # Close
    print(f'Predicted Close price for 2025-09-12: {predicted_close:.2f}')
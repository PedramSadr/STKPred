import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import random
import os
import argparse
import time
import math

# --- MODEL DEFINITION ---
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(BiLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), output_size)
        )

    def forward(self, x):
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        last = self.layer_norm(last)
        out = self.head(last)
        return out
# --- END MODEL DEFINITION ---

# Default number of workers: 0 is safe for memory constrained environments (Windows)
DEFAULT_NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Diagnostic block: print PyTorch / CUDA / cuDNN / GPU memory info
try:
    import torch.backends.cudnn as cudnn
    import subprocess555
    print("Torch version:", torch.__version__)
    print("Torch compiled CUDA:", torch.version.cuda)
    print("CUDA available (torch):", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("CUDA devices:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                try:
                    print(f" - Device {i}:", torch.cuda.get_device_name(i))
                except Exception as e:
                    print(f"   Could not get device name for device {i}: {e}")
        except Exception as e:
            print("Error enumerating CUDA devices:", e)
        try:
            print("cuDNN version:", cudnn.version())
        except Exception as e:
            print("cuDNN check failed:", e)
        try:
            print("CUDA memory allocated (MB):", torch.cuda.memory_allocated() / 1024**2)
            print("CUDA memory reserved  (MB):", torch.cuda.memory_reserved() / 1024**2)
        except Exception as e:
            print("CUDA memory query failed:", e)
    else:
        print("Running on CPU (CUDA not available to torch)")
    # Try to call nvidia-smi to show driver-level info
    try:
        out = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print("\n-- nvidia-smi output --\n", out)
    except Exception as e:
        print("nvidia-smi call failed (not found or error):", e)
except Exception as e:
    print("CUDA diagnostic block failed:", e)

# python
CSV_PATH = r"C:\My Documents\Mics\Logs\TSLA_Options_Chain_Historical_combined.csv"
OUTPUT_MODEL_PATH = r"C:\My Documents\Mics\Logs\tsla_bilstm_option_model_best.pth"
FINAL_MODEL_PATH = r"C:\My Documents\Mics\Logs\tsla_option_bilstm_model_final.pth"
SCALER_PATH = r"C:\My Documents\Mics\Logs\tsla_option_scaler_stats.npz"

# CLI arguments
parser = argparse.ArgumentParser(description='Train BiLSTM on options chain data')
parser.add_argument('--csv', '-c', default=CSV_PATH, help='Path to combined options CSV')
parser.add_argument('--target', '-t', default=None,
                    help="Target column to predict (if not provided, script will use 'close', then 'last', then 'mark')")
parser.add_argument('--trials', type=int, default=None, help='Number of hyperparameter trials (overrides default)')
parser.add_argument('--hidden-min', type=int, default=None, help='Minimum hidden_size')
parser.add_argument('--hidden-max', type=int, default=None, help='Maximum hidden_size')
parser.add_argument('--seq-min', type=int, default=None, help='Minimum sequence_length')
parser.add_argument('--seq-max', type=int, default=None, help='Maximum sequence_length')
parser.add_argument('--epochs-min', type=int, default=None, help='Minimum epochs per trial')
parser.add_argument('--epochs-max', type=int, default=None, help='Maximum epochs per trial')
parser.add_argument('--batch-min', type=int, default=None, help='Minimum batch size')
parser.add_argument('--batch-max', type=int, default=None, help='Maximum batch size')
parser.add_argument('--max-seconds-per-trial', type=int, default=300,
                    help='Target maximum seconds per hyperparameter trial (default 300s)')
parser.add_argument('--max-rows', type=int, default=None, help='Maximum CSV rows to load (safe-memory cap)')
parser.add_argument('--dry-run', action='store_true', help='Print first few rows and chosen target then exit')
parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader num_workers')
args = parser.parse_args()

CSV_PATH = args.csv
TARGET_OVERRIDE = args.target
MAX_SECONDS_PER_TRIAL = int(args.max_seconds_per_trial)
MAX_ROWS = args.max_rows
if args.num_workers is not None:
    DEFAULT_NUM_WORKERS = int(args.num_workers)

print(f"CSV_PATH set to: {CSV_PATH}")
print(f"Target seconds per trial: {MAX_SECONDS_PER_TRIAL}s")
if TARGET_OVERRIDE:
    print(f"Target column override provided: {TARGET_OVERRIDE}")
if MAX_ROWS:
    print(f"Max rows limit enabled: {MAX_ROWS}")
if args.dry_run:
    print("Dry run: will print first few rows and chosen target then exit.")

# Load data (with optional row cap to avoid OOM)
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

# Use iterator reading if file large; cap rows early to reduce memory usage
if MAX_ROWS:
    df = pd.read_csv(CSV_PATH, low_memory=False, nrows=MAX_ROWS)
else:
    df = pd.read_csv(CSV_PATH, low_memory=False)

df.columns = df.columns.str.strip().str.lower()
df.dropna(how='all', inplace=True)

# Choose target column
if TARGET_OVERRIDE:
    if TARGET_OVERRIDE in df.columns:
        target_col = TARGET_OVERRIDE
    else:
        raise KeyError(f"Requested target column '{TARGET_OVERRIDE}' not found in CSV columns: {df.columns.tolist()}")
else:
    if 'last' in df.columns:
        target_col = 'last'
    elif 'close' in df.columns:
        target_col = 'close'
    elif 'mark' in df.columns:
        target_col = 'mark'
    else:
        raise KeyError(
            "Could not find 'last', 'close', or 'mark' in CSV to use as target; provide --target to override")

print(f"Using target column: {target_col}")

# If user asked for dry-run, show head and exit
if args.dry_run:
    print("\nFirst 5 raw CSV rows (selected columns):")
    print(df.head(5).to_string(index=False))
    print(f"\nChosen target column: `{target_col}`")
    exit(0)

# Ensure target is numeric
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df[~df[target_col].isna()].reset_index(drop=True)

feature_cols = [
    'contractid', 'expiration', 'strike', 'type', 'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size',
    'volume', 'open_interest', 'date', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho'
]

present = [c for c in feature_cols if c in df.columns]
missing = [c for c in feature_cols if c not in df.columns]
print(f"Requested features present: {present}")
if missing:
    print(f"Warning: Requested features missing from CSV and will be filled with NaN: {missing}")

features_df = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in feature_cols})

for dcol in ('expiration', 'date'):
    if dcol in features_df.columns:
        try:
            features_df[dcol] = pd.to_datetime(features_df[dcol], errors='coerce').astype('int64') // 10 ** 9
        except Exception:
            features_df[dcol] = pd.to_datetime(features_df[dcol], errors='coerce')
            features_df[dcol] = features_df[dcol].apply(lambda x: int(x.timestamp()) if pd.notna(x) else np.nan)

if 'type' in features_df.columns:
    features_df['type'] = features_df['type'].astype(str).str.lower().map({'call': 1, 'c': 1, 'put': 0, 'p': 0})
if 'contractid' in features_df.columns:
    try:
        features_df['contractid'] = features_df['contractid'].astype('category').cat.codes
    except Exception:
        features_df['contractid'] = pd.to_numeric(features_df['contractid'], errors='coerce')

for col in features_df.columns:
    if features_df[col].dtype == 'object':
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

features_df.fillna(0, inplace=True)

# Prepare numpy arrays in float32 (avoid extra memory by ensuring dtype)
prices = df[target_col].values.astype(np.float32)
features = features_df.values.astype(np.float32)

if len(prices) != len(features):
    raise RuntimeError("Length mismatch between features and prices after preprocessing")

# ===== Reproducibility, lag feature and weight init (inserted here before splits) =====
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# add simple lag-1 price feature and update features array
lag1 = np.concatenate(([0.0], prices[:-1])).astype(np.float32).reshape(-1, 1)
features = np.hstack([features, lag1])

# update input_size to reflect new features (used by estimate and models)
input_size = features.shape[1]
output_size = 1

# weight initialization helper
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
# ===== end insertion =====

# Train/validation split (chronological)
train_features_raw, val_features_raw, train_prices, val_prices = train_test_split(
    features, prices, test_size=0.2, shuffle=False)

# Normalization on training set
feat_mean = train_features_raw.mean(axis=0)
feat_std = train_features_raw.std(axis=0)
feat_std[feat_std == 0] = 1.0
price_mean = train_prices.mean()
price_std = train_prices.std() if train_prices.std() != 0 else 1.0

train_features = (train_features_raw - feat_mean) / feat_std
val_features = (val_features_raw - feat_mean) / feat_std
train_prices_norm = (train_prices - price_mean) / price_std
val_prices_norm = (val_prices - price_mean) / price_std

# Convert to torch tensors on CPU (do not move to GPU until batches)
train_features = torch.from_numpy(train_features).float()
val_features = torch.from_numpy(val_features).float()
train_prices_norm = torch.from_numpy(train_prices_norm).float()
val_prices_norm = torch.from_numpy(val_prices_norm).float()

class SequenceDataset(Dataset):
    def __init__(self, features_tensor, prices_tensor, seq_length):
        assert isinstance(features_tensor, torch.Tensor)
        assert isinstance(prices_tensor, torch.Tensor)
        self.features = features_tensor
        self.prices = prices_tensor
        self.seq_length = int(seq_length)
        self.n = max(0, self.features.size(0))
        self.len = max(0, self.n - self.seq_length)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_length
        x = self.features[start:end]
        y = self.prices[end].unsqueeze(0)
        return x, y

torch.backends.cudnn.benchmark = True
USE_AMP = device.type == 'cuda'
print(f"USE_AMP={USE_AMP}")

# Faster/smaller default hyperparameter ranges to keep trials short
hidden_size_range = (16, 32)
sequence_length_range = (5, 20)
learning_rate_range = (0.001, 0.01)
epochs_range = (3, 6)
trials = 1000
batch_size_range = (32, 512)

# Override ranges with CLI args if provided
if args.trials is not None:
    trials = int(args.trials)
if args.hidden_min is not None:
    hidden_size_range = (int(args.hidden_min), hidden_size_range[1])
if args.hidden_max is not None:
    hidden_size_range = (hidden_size_range[0], int(args.hidden_max))
if args.seq_min is not None:
    sequence_length_range = (int(args.seq_min), sequence_length_range[1])
if args.seq_max is not None:
    sequence_length_range = (sequence_length_range[0], int(args.seq_max))
if args.epochs_min is not None:
    epochs_range = (int(args.epochs_min), epochs_range[1])
if args.epochs_max is not None:
    epochs_range = (epochs_range[0], int(args.epochs_max))
if args.batch_min is not None:
    batch_size_range = (int(args.batch_min), batch_size_range[1])
if args.batch_max is not None:
    batch_size_range = (batch_size_range[0], int(args.batch_max))

print(
    f"Hyperparam ranges: hidden={hidden_size_range}, seq={sequence_length_range}, epochs={epochs_range}, trials={trials}, batch={batch_size_range}, num_workers={DEFAULT_NUM_WORKERS}")

best_loss = float('inf')
best_params = None

def estimate_batch_time(input_size, hidden_size, seq_length, batch_size, device):
    try:
        m = BiLSTMNet(input_size, hidden_size, output_size).to(device)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        # use small synthetic batch to estimate; keep batch_size small to avoid OOM
        small_bs = max(1, min(batch_size, 16))
        xb = torch.randn(small_bs, seq_length, input_size, device=device)
        yb = torch.randn(small_bs, 1, device=device)
        for _ in range(2):
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = m(xb)
                l = crit(out, yb)
            opt.zero_grad()
            l.backward()
            opt.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            out = m(xb)
            l = crit(out, yb)
        opt.zero_grad()
        l.backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        batch_time = max(1e-6, t1 - t0)
        del m, opt, crit, xb, yb, out, l
        torch.cuda.empty_cache()
        return batch_time
    except Exception as e:
        print(f"Batch time estimation failed: {e}")
        return None

for trial in range(trials):
    hidden_size = random.randint(*hidden_size_range)
    sequence_length = random.randint(*sequence_length_range)
    learning_rate = 10 ** random.uniform(np.log10(learning_rate_range[0]), np.log10(learning_rate_range[1]))
    epochs = random.randint(*epochs_range)
    # choose batch as power of two bounded by range and dataset size
    min_bs_pow = int(math.log2(max(1, batch_size_range[0])))
    max_bs_pow = int(math.log2(max(1, batch_size_range[1])))
    max_bs_pow = max(min_bs_pow, max_bs_pow)
    bs_pow = random.randint(min_bs_pow, max_bs_pow)
    batch_size = 2 ** bs_pow
    patience = 3
    min_delta = 1e-5

    print(
        f"\nTrial {trial + 1}/{trials}: hidden={hidden_size}, seq_len={sequence_length}, lr={learning_rate:.5f}, epochs={epochs}, batch={batch_size}")

    train_dataset_len = max(0, len(train_features) - sequence_length)
    val_dataset_len = max(0, len(val_features) - sequence_length)

    if train_dataset_len <= 0 or val_dataset_len <= 0:
        print("Skipping trial, dataset too small for this sequence length.")
        continue

    est_batch = estimate_batch_time(input_size, hidden_size, sequence_length, min(batch_size, 32), device)
    if est_batch is not None and est_batch > 0:
        est_batches_per_epoch = max(1, math.ceil(train_dataset_len / batch_size))
        est_epoch_time = est_batch * est_batches_per_epoch
        est_max_epochs = max(1, int(max(1, MAX_SECONDS_PER_TRIAL) / max(est_epoch_time, 1e-6)))
        if epochs > est_max_epochs:
            print(f"Capping epochs from {epochs} to {est_max_epochs} to meet {MAX_SECONDS_PER_TRIAL}s per trial")
            epochs = est_max_epochs
    else:
        print("Skipping time-based capping for this trial (estimation failed)")

    train_dataset = SequenceDataset(train_features, train_prices_norm, sequence_length)
    val_dataset = SequenceDataset(val_features, val_prices_norm, sequence_length)

    dl_train_kwargs = dict(batch_size=batch_size, shuffle=True,
                           num_workers=DEFAULT_NUM_WORKERS,
                           pin_memory=(device.type == 'cuda'))
    dl_val_kwargs = dict(batch_size=batch_size, shuffle=False,
                         num_workers=DEFAULT_NUM_WORKERS,
                         pin_memory=(device.type == 'cuda'))

    train_loader = DataLoader(train_dataset, **dl_train_kwargs)
    val_loader = DataLoader(val_dataset, **dl_val_kwargs)

    # model, criterion, optimizer, scheduler (improved)
    model = BiLSTMNet(input_size, hidden_size, output_size).to(device)
    model.apply(init_weights)
    criterion = nn.SmoothL1Loss()  # Huber-like
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_for_trial = None
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run += 1
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == 'cuda'))
            yb = yb.to(device, non_blocking=(device.type == 'cuda'))
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = criterion(pred, yb)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_train_loss += loss.item() * xb.size(0)
            train_samples += xb.size(0)

        avg_train_loss = epoch_train_loss / train_samples if train_samples else float('nan')

        model.eval()
        epoch_val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=(device.type == 'cuda'))
                yb = yb.to(device, non_blocking=(device.type == 'cuda'))
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        pred = model(xb)
                        batch_loss = criterion(pred, yb).item()
                else:
                    pred = model(xb)
                    batch_loss = criterion(pred, yb).item()
                epoch_val_loss += batch_loss * xb.size(0)
                val_samples += xb.size(0)

        avg_val_loss = epoch_val_loss / val_samples if val_samples else float('nan')

        # Cosine scheduler: step once per epoch
        try:
            scheduler.step()
        except Exception:
            # fallback: if scheduler expects metric, ignore
            pass

        if epoch % 10 == 0 or epoch == epochs - 1:
            train_loss_str = f"{avg_train_loss:.6f}" if (isinstance(avg_train_loss, float) and not math.isnan(avg_train_loss)) else str(avg_train_loss)
            val_loss_str = f"{avg_val_loss:.6f}" if (isinstance(avg_val_loss, float) and not math.isnan(avg_val_loss)) else str(avg_val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss_str}, Val Loss: {val_loss_str}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state_for_trial = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    best_val_str = f"{best_val_loss:.6f}" if (best_val_loss is not None and isinstance(best_val_loss, float) and not math.isnan(best_val_loss)) else str(best_val_loss)
    print(f"Best validation loss for this trial: {best_val_str}")
    if best_val_loss < best_loss:
        best_loss = best_val_loss
        best_params = {
            'hidden_size': hidden_size, 'sequence_length': sequence_length,
            'learning_rate': learning_rate, 'epochs_run': epochs_run, 'batch_size': batch_size
        }
        if best_model_state_for_trial:
            torch.save(best_model_state_for_trial, OUTPUT_MODEL_PATH)
            print(f"--- New best model saved with loss: {best_loss:.6f} ---")
        else:
            print("No model state saved for this trial (no improvement detected).")

print("\n--- Hyperparameter Search Complete ---")
print("Best hyperparameters found:")
if best_params:
    for k, v in best_params.items():
        print(f"{k}: {v}")
print(f"Best validation loss across all trials: {best_loss:.6f}")

np.savez(SCALER_PATH, feat_mean=feat_mean, feat_std=feat_std, price_mean=price_mean, price_std=price_std)
print(f"Scaler stats saved to {SCALER_PATH}")

if best_params is None:
    raise RuntimeError("No model was trained successfully")

print(f"Loading best model from {OUTPUT_MODEL_PATH} for final save and plot.")
# Guard to satisfy static checks and avoid None
assert best_params is not None, "best_params is None at final save"
hidden_size_final = int(best_params['hidden_size'])
final_model = BiLSTMNet(input_size, hidden_size_final, output_size)
final_model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location='cpu'))
torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f"\nFinal best model saved to {FINAL_MODEL_PATH}")

print("\n--- Plotting Best Model Predictions ---")
best_model = final_model
best_model.to(device)
best_model.eval()

# Guard for seq length
assert best_params is not None
best_seq_length = int(best_params['sequence_length'])
val_plot_dataset = SequenceDataset(val_features, val_prices_norm, best_seq_length)
if len(val_plot_dataset) == 0:
    print("Could not generate plot because the validation set was too small for the best sequence length.")
else:
    plot_batch_size = int(best_params.get('batch_size', 64))
    plot_loader = DataLoader(val_plot_dataset, batch_size=plot_batch_size, shuffle=False,
                             num_workers=DEFAULT_NUM_WORKERS, pin_memory=(device.type == 'cuda'))

    preds_norm = []
    actuals_norm = []
    with torch.no_grad():
        for xb, yb in plot_loader:
            xb = xb.to(device)
            out = best_model(xb)
            preds_norm.extend(out.cpu().numpy().reshape(-1).tolist())
            actuals_norm.extend(yb.cpu().numpy().reshape(-1).tolist())

    import numpy as _np
    predictions_normalized = _np.array(preds_norm)
    actuals_normalized = _np.array(actuals_norm)

    predicted_prices = (predictions_normalized * price_std) + price_mean
    actual_prices = (actuals_normalized * price_std) + price_mean

    plt.figure(figsize=(15, 7))
    plt.plot(actual_prices, label='Actual Price', color='blue', linewidth=2)
    plt.plot(predicted_prices, label='Predicted Price', color='red', linestyle='--', linewidth=2)
    plt.title('Best BiLSTM Model: Actual vs. Predicted Prices')
    plt.xlabel('Time Step (in validation set)')
    plt.ylabel('Option Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
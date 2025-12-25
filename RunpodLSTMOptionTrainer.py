import os
import math
import time
import random
import argparse
import multiprocessing
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Try importing optuna
try:
    import optuna
except ImportError:
    optuna = None
#This code implements a BiLSTM model trainer for time series data, specifically designed for predicting stock option prices.
# It includes model definition, dataset handling, training routines, and hyperparameter optimization using random search or Optuna.
#To run this code
55# torchrun --standalone --nnodes=1 --nproc_per_node=8 trainer.py \
#   --mode random \
#   --csv /workspace/TSLA_Options_Chain_Historical_combined.csv \
#   --trials 80

# =========================
#  Model Definition
# =========================
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(BiLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, max(64, hidden_size)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden_size), output_size),
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0 = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
        c0 = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        last = self.layer_norm(last)
        out = self.head(last)
        return out


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)


# =========================
#  Dataset
# =========================
class SequenceDataset(Dataset):
    def __init__(self, features_tensor, prices_tensor, seq_length: int):
        self.features = features_tensor
        self.prices = prices_tensor
        self.seq_length = int(seq_length)
        self.n = self.features.size(0)
        self.len = max(0, self.n - self.seq_length)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_length
        x = self.features[start:end]
        y = self.prices[end].unsqueeze(0)
        return x, y


# =========================
#  Utility: convert loss -> dollars
# =========================
def val_mae_to_dollars(val_mae: float, effective_price_std: float) -> float:
    """Convert normalized MAE to average absolute error in dollars."""
    return float(val_mae) * float(effective_price_std)


# =========================
#  Training Utility
# =========================
def train_one_config(
        config,
        train_features,
        train_prices_norm,
        val_features,
        val_prices_norm,
        input_size,
        output_size,
        device,
        num_workers,
        use_amp,
        max_seconds_per_trial,
        effective_price_std,
):
    hidden_size = config["hidden_size"]
    seq_length = config["seq_length"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Re-init datasets with new seq_length
    train_dataset = SequenceDataset(train_features, train_prices_norm, seq_length)
    val_dataset = SequenceDataset(val_features, val_prices_norm, seq_length)

    if len(train_dataset) < 2:
        return float("inf"), float("inf"), None, 0.0

    batch_size = min(batch_size, len(train_dataset))
    if batch_size < 1:
        batch_size = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = BiLSTMNet(input_size, hidden_size, output_size, num_layers=2, dropout=0.1)
    model.apply(init_weights)
    model.to(device)

    # Optimization Objective (Gradient Friendly)
    train_criterion = nn.SmoothL1Loss()
    # Selection Objective (Dollar Friendly)
    report_criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Track best by DOLLARS
    best_val_dollars = float("inf")
    best_val_mae = float("inf")  # <-- We now specifically track the BEST MAE
    best_state = None
    epochs_no_improve = 0
    patience = 3

    start_trial_time = time.time()

    for epoch in range(epochs):
        model.train()
        try:
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(xb)
                        loss = train_criterion(pred, yb)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(xb)
                    loss = train_criterion(pred, yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM with config: {config}")
                torch.cuda.empty_cache()
                return float("inf"), float("inf"), None, 0.0
            else:
                raise

        # Validation
        model.eval()
        epoch_val_mae = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(xb)
                        batch_mae = report_criterion(pred, yb).item()
                else:
                    pred = model(xb)
                    batch_mae = report_criterion(pred, yb).item()

                epoch_val_mae += batch_mae * xb.size(0)
                val_samples += xb.size(0)

        avg_val_mae = epoch_val_mae / max(1, val_samples)
        current_dollars = val_mae_to_dollars(avg_val_mae, effective_price_std)

        scheduler.step()

        # Dollar-based early stopping + best tracking
        if current_dollars < best_val_dollars:
            best_val_dollars = current_dollars
            best_val_mae = avg_val_mae  # <-- Store the MAE associated with this best dollar value
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        # Time-based cutoff
        if time.time() - start_trial_time > max_seconds_per_trial:
            break

    total_time = time.time() - start_trial_time

    # RETURN THE BEST MAE (not the last epoch's MAE)
    return best_val_dollars, best_val_mae, best_state, total_time


def main():
    parser = argparse.ArgumentParser(description="Hybrid BiLSTM trainer: 8x RTX 6000 Edition.")

    parser.add_argument("--mode", type=str, default="random", choices=["random", "optuna"])
    parser.add_argument("--csv", "-c", default=r"C:\My Documents\Mics\Logs\TSLA_Options_Chain_Historical_combined.csv")
    parser.add_argument("--target", "-t", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    # === RTX 6000 TUNED DEFAULTS ===
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--hidden-min", type=int, default=256)
    parser.add_argument("--hidden-max", type=int, default=1024)
    parser.add_argument("--seq-min", type=int, default=20)
    parser.add_argument("--seq-max", type=int, default=80)
    parser.add_argument("--batch-min", type=int, default=4096)
    parser.add_argument("--batch-max", type=int, default=32768)
    parser.add_argument("--epochs-min", type=int, default=5)
    parser.add_argument("--epochs-max", type=int, default=20)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=3e-3)

    parser.add_argument("--max-seconds-per-trial", type=int, default=600)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-prefix", type=str, default="tsla_bilstm_model_best")

    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument("--optuna-study-name", type=str, default="tsla_bilstm_optuna")

    args = parser.parse_args()

    # Environment
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Seed
    seed = 42 + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Worker Calc
    try:
        cpu_count = multiprocessing.cpu_count()
    except Exception:
        cpu_count = 8

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = max(1, min(8, cpu_count // max(1, world_size)))

    if global_rank == 0:
        print(
            f"Global Config: Workers={num_workers}, Device Count={torch.cuda.device_count()}, World Size={world_size}")

    # =========================
    #  Load Data
    # =========================
    if not os.path.exists(args.csv):
        if global_rank == 0:
            print(f"CSV not found: {args.csv}. Generating dummy data.")
        df = pd.DataFrame(
            np.random.randn(10000, 20),
            columns=["close", "last", "mark"] + [f"f{i}" for i in range(17)]
        )
        df["date"] = pd.date_range(start="2022-01-01", periods=10000)
    else:
        if args.max_rows:
            df = pd.read_csv(args.csv, low_memory=False, nrows=args.max_rows)
        else:
            df = pd.read_csv(args.csv, low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    df.dropna(how="all", inplace=True)

    if args.target:
        target_col = args.target
    else:
        if "last" in df.columns:
            target_col = "last"
        elif "close" in df.columns:
            target_col = "close"
        elif "mark" in df.columns:
            target_col = "mark"
        else:
            target_col = df.columns[0]

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[~df[target_col].isna()].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != target_col and df[c].dtype != "object"]
    features_df = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)

    prices = df[target_col].values.astype(np.float32)
    features = features_df.values.astype(np.float32)

    # Lag 1
    lag1 = np.concatenate(([0.0], prices[:-1])).astype(np.float32).reshape(-1, 1)
    features = np.hstack([features, lag1])

    input_size = features.shape[1]
    output_size = 1

    del df, features_df
    gc.collect()

    if args.dry_run:
        if global_rank == 0:
            print(f"Dry Run. Input dim: {input_size}. Samples: {len(prices)}")
            print(f"Target column: {target_col}")
        return

    # Split & Normalize
    train_features_raw, val_features_raw, train_prices, val_prices = train_test_split(
        features, prices, test_size=0.2, shuffle=False
    )

    feat_mean = train_features_raw.mean(axis=0)
    feat_std = train_features_raw.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    price_mean = train_prices.mean()
    raw_price_std = train_prices.std() if train_prices.std() != 0 else 1.0
    target_scale = float(args.target_scale)
    effective_price_std = raw_price_std * target_scale

    train_features = (train_features_raw - feat_mean) / feat_std
    val_features = (val_features_raw - feat_mean) / feat_std
    train_prices_norm = (train_prices - price_mean) / effective_price_std
    val_prices_norm = (val_prices - price_mean) / effective_price_std

    del features, prices, train_features_raw, val_features_raw
    gc.collect()

    train_features = torch.from_numpy(train_features).float()
    val_features = torch.from_numpy(val_features).float()
    train_prices_norm = torch.from_numpy(train_prices_norm).float()
    val_prices_norm = torch.from_numpy(val_prices_norm).float()

    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"

    trials = args.trials
    approx_len = train_features.size(0)
    batch_max = args.batch_max
    if approx_len < batch_max:
        batch_max = max(args.batch_min, approx_len // 2)

    print(
        f"[Rank {global_rank}] Config Ranges: "
        f"Hidden=[{args.hidden_min}, {args.hidden_max}], "
        f"Batch=[{args.batch_min}, {batch_max}], "
        f"Trials={trials}"
    )

    # =========================
    #  MODE: RANDOM SEARCH
    # =========================
    if args.mode == "random":
        best_global_dollars = float("inf")
        best_global_mae = float("inf")
        best_global_state = None
        best_global_params = None

        for trial_idx in range(trials):
            hidden_size = random.randint(args.hidden_min, args.hidden_max)
            seq_length = random.randint(args.seq_min, args.seq_max)
            epochs = random.randint(args.epochs_min, args.epochs_max)

            min_bs_pow = int(math.log2(max(1, args.batch_min)))
            max_bs_pow = int(math.log2(max(1, batch_max)))
            bs_pow = random.randint(min_bs_pow, max_bs_pow)
            batch_size = 2 ** bs_pow

            lr = 10 ** random.uniform(math.log10(args.lr_min), math.log10(args.lr_max))

            config = {
                "hidden_size": hidden_size,
                "seq_length": seq_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            }

            print(f"[Rank {global_rank}] Trial {trial_idx + 1}/{trials}: {config}")

            val_dollars, val_mae, state_dict, took = train_one_config(
                config,
                train_features,
                train_prices_norm,
                val_features,
                val_prices_norm,
                input_size,
                output_size,
                device,
                num_workers,
                use_amp,
                args.max_seconds_per_trial,
                effective_price_std,
            )

            print(
                f"[Rank {global_rank}]    -> result=${val_dollars:.4f} "
                f"(norm-MAE={val_mae:.6f}, time={took:.1f}s)"
            )

            if state_dict is not None and val_dollars < best_global_dollars:
                best_global_dollars = val_dollars
                best_global_mae = val_mae
                best_global_state = state_dict
                best_global_params = config
                print(
                    f"[Rank {global_rank}]    *** NEW BEST (Dollars): "
                    f"${best_global_dollars:.4f} ***"
                )

        if best_global_state:
            loss_str = f"{best_global_dollars:.4f}".replace(".", "_")
            out_name = f"{args.output_prefix}_rank{global_rank}_dollars{loss_str}.pth"
            save_obj = {
                "state_dict": best_global_state,
                "best_loss_dollars": best_global_dollars,
                "best_val_mae": best_global_mae,
                "best_params": best_global_params,
                "rank": global_rank,
                "scaler_stats": {
                    "feat_mean": feat_mean,
                    "feat_std": feat_std,
                    "price_mean": price_mean,
                    "price_std": raw_price_std,
                    "effective_price_std": effective_price_std,
                    "target_scale": target_scale,
                },
            }
            torch.save(save_obj, out_name)
            print(f"[Rank {global_rank}] Saved best model to {out_name}")

    # =========================
    #  MODE: OPTUNA
    # =========================
    elif args.mode == "optuna":
        if optuna is None:
            raise ImportError("Optuna is not installed.")

        def optuna_objective(trial: "optuna.trial.Trial"):
            hidden_size = trial.suggest_int("hidden_size", args.hidden_min, args.hidden_max)
            seq_length = trial.suggest_int("seq_length", args.seq_min, args.seq_max)
            epochs = trial.suggest_int("epochs", args.epochs_min, args.epochs_max)
            log_lr = trial.suggest_float("log_lr", math.log10(args.lr_min), math.log10(args.lr_max))

            min_bs_pow = int(math.log2(max(1, args.batch_min)))
            max_bs_pow = int(math.log2(max(1, batch_max)))
            bs_pow = trial.suggest_int("bs_pow", min_bs_pow, max_bs_pow)

            config = {
                "hidden_size": hidden_size,
                "seq_length": seq_length,
                "epochs": epochs,
                "batch_size": 2 ** bs_pow,
                "lr": 10 ** log_lr,
            }

            val_dollars, val_mae, _, took = train_one_config(
                config,
                train_features,
                train_prices_norm,
                val_features,
                val_prices_norm,
                input_size,
                output_size,
                device,
                num_workers,
                use_amp,
                args.max_seconds_per_trial,
                effective_price_std,
            )
            trial.set_user_attr("time", took)
            trial.set_user_attr("val_dollars", val_dollars)
            trial.set_user_attr("val_mae", val_mae)
            return val_dollars

        if args.optuna_storage:
            study = optuna.create_study(
                study_name=args.optuna_study_name,
                storage=args.optuna_storage,
                direction="minimize",
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(direction="minimize")

        print(f"[Rank {global_rank}] Starting Optuna refinement for {trials} trials...")
        study.optimize(optuna_objective, n_trials=trials)

        print(f"[Rank {global_rank}] Optuna best value (dollars) = ${study.best_value:.4f}")
        print(f"[Rank {global_rank}] Optuna best params: {study.best_params}")

        best_p = study.best_params
        final_config = {
            "hidden_size": best_p["hidden_size"],
            "seq_length": best_p["seq_length"],
            "epochs": best_p["epochs"],
            "batch_size": 2 ** best_p["bs_pow"],
            "lr": 10 ** best_p["log_lr"],
        }

        final_dollars, final_mae, best_state, took = train_one_config(
            final_config,
            train_features,
            train_prices_norm,
            val_features,
            val_prices_norm,
            input_size,
            output_size,
            device,
            num_workers,
            use_amp,
            args.max_seconds_per_trial,
            effective_price_std,
        )

        if best_state:
            loss_str = f"{final_dollars:.4f}".replace(".", "_")
            out_name = f"{args.output_prefix}_optuna_rank{global_rank}_dollars{loss_str}.pth"
            save_obj = {
                "state_dict": best_state,
                "best_loss_dollars": final_dollars,
                "best_val_mae": final_mae,
                "best_params": final_config,
                "rank": global_rank,
                "scaler_stats": {
                    "feat_mean": feat_mean,
                    "feat_std": feat_std,
                    "price_mean": price_mean,
                    "price_std": raw_price_std,
                    "effective_price_std": effective_price_std,
                    "target_scale": target_scale,
                },
                "optuna_best_raw": {
                    "value_dollars": study.best_value,
                    "params": study.best_params,
                },
            }
            torch.save(save_obj, out_name)
            print(
                f"[Rank {global_rank}] Saved Optuna-refined model to {out_name} "
                f"(~${final_dollars:.4f}, norm-MAE={final_mae:.6f})"
            )


if __name__ == "__main__":
    main()
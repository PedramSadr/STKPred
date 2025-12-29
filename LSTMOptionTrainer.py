#!/usr/bin/env python3
"""
RunpodLSTMOptionTrainer_fixed.py

Stage-2 (Options/Surface) Bi-LSTM trainer to predict NEXT-14-trading-day realized volatility
using daily surface/regime features (TSLA_Surface_Vector.csv) aligned with close prices
(tsla_daily.csv).

Fixes vs prior version:
- NO data leakage: scalers are fit on TRAIN ONLY.
- GPU support (+ optional AMP).
- Optional multi-GPU via torchrun (DDP). Saves ONLY on rank0 (one file).
- Better dataloader defaults (pin_memory, persistent_workers).
- Saves a single checkpoint dict: state_dict + metadata + scalers.

Example (single GPU):
  python RunpodLSTMOptionTrainer_fixed.py --surface TSLA_Surface_Vector.csv --daily tsla_daily.csv

Example (8 GPUs on Runpod):
  torchrun --standalone --nproc_per_node=8 RunpodLSTMOptionTrainer_fixed.py \
      --surface TSLA_Surface_Vector.csv --daily tsla_daily.csv \
      --global-batch-size 4096 --num-workers 8

"""

import os
import math
import argparse
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.preprocessing import MinMaxScaler

# Default input directory for CSV files
DEFAULT_INPUT_DIR = r"C:\My Documents\Mics\Logs"


# -------------------------
# Utilities: DDP env
# -------------------------
def ddp_env() -> Tuple[int, int, int]:
    """
    Returns (world_size, rank, local_rank).
    Works with torchrun or single-process runs.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def ddp_init_if_needed(world_size: int) -> None:
    if world_size <= 1:
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def ddp_cleanup_if_needed(world_size: int) -> None:
    if world_size <= 1:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Dataset
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.ndim == 3, f"X must be [N, seq, feat], got {X.shape}"
        assert y.ndim == 2, f"y must be [N, 1], got {y.shape}"
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------
# Model
# -------------------------
class BiLSTMRegimeModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# -------------------------
# Training
# -------------------------
@dataclass
class TrainMeta:
    seq_length: int
    pred_window: int
    features: List[str]
    target_col: str
    hidden_size: int
    num_layers: int
    dropout: float
    train_rows: int
    val_rows: int
    train_examples: int
    val_examples: int
    scaler_x_type: str
    scaler_y_type: str


def build_forward_realized_vol_target(df: pd.DataFrame, close_col: str, pred_window: int) -> pd.Series:
    """
    Builds forward-looking realized volatility:
    target at time T = realized vol of log returns from T+1 .. T+pred_window (annualized).

    Implementation:
    - log_ret[t] = log(close[t] / close[t-1])
    - realized_vol[t] = std(log_ret[t-pred_window+1 .. t]) * sqrt(252)
    - target_future_vol[t] = realized_vol[t+pred_window]
    """
    log_ret = np.log(df[close_col] / df[close_col].shift(1))
    realized_vol = log_ret.rolling(window=pred_window).std() * np.sqrt(252.0)
    target = realized_vol.shift(-pred_window)
    return target


def make_sequences(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    seq_length: int,
    train_cutoff_row: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sequences, then split WITHOUT leakage:
    - Sequence i ends at row end_idx = i + seq_length - 1
    - Train sequences: end_idx < train_cutoff_row
    - Val sequences:   end_idx >= train_cutoff_row
      (Val sequences may include history from train rows, which is realistic and not leakage,
       because the label is still in the future and scalers are train-fit only.)
    """
    assert len(X_scaled) == len(y_scaled)
    n_rows = len(X_scaled)
    n_seq = n_rows - seq_length
    if n_seq <= 0:
        raise ValueError(f"Not enough rows ({n_rows}) for seq_length={seq_length}")

    X_list = []
    y_list = []
    end_indices = []
    for i in range(n_seq):
        end_idx = i + seq_length - 1
        X_list.append(X_scaled[i : i + seq_length])
        y_list.append(y_scaled[end_idx])  # target aligned at end of window
        end_indices.append(end_idx)

    X_all = torch.tensor(np.asarray(X_list), dtype=torch.float32)
    y_all = torch.tensor(np.asarray(y_list), dtype=torch.float32)

    end_indices = np.asarray(end_indices)
    train_mask = end_indices < train_cutoff_row
    val_mask = ~train_mask

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_val = X_all[val_mask]
    y_val = y_all[val_mask]

    return X_train, y_train, X_val, y_val


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / max(1, n)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_amp: bool,
    rank: int,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val = evaluate(model, val_loader, criterion, device)

        if rank == 0 and (ep == 1 or ep % 5 == 0):
            print(f"Epoch {ep:4d} | val_mse={val:.6f} | best={best_val:.6f}")

        if val < best_val - 1e-9:
            best_val = val
            bad = 0
            # If DDP, unwrap to keep clean state_dict
            state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            best_state = {k: v.detach().cpu() for k, v in state.items()}
        else:
            bad += 1
            if bad >= patience:
                if rank == 0:
                    print(f"Early stopping at epoch {ep} (patience={patience}).")
                break

    if best_state is None:
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        best_state = {k: v.detach().cpu() for k, v in state.items()}

    return best_val, best_state


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default=os.path.join(DEFAULT_INPUT_DIR, "TSLA_Surface_Vector.csv"))
    p.add_argument("--daily", default=os.path.join(DEFAULT_INPUT_DIR, "tsla_daily.csv"))
    p.add_argument("--model-out", default="stage2_bilstm_futurevol14d.pth")

    p.add_argument("--seq-length", type=int, default=10)
    p.add_argument("--pred-window", type=int, default=14)

    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--global-batch-size", type=int, default=1024, help="Total batch size across all GPUs.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    args = p.parse_args()

    world_size, rank, local_rank = ddp_env()
    ddp_init_if_needed(world_size)

    # Seed (make each rank deterministic but different enough if needed)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"World size: {world_size} | Device: {device} | AMP: {not args.no_amp}")

    # -------------------------
    # Load + merge data
    # -------------------------
    df_surf = pd.read_csv(args.surface)
    df_daily = pd.read_csv(args.daily)

    if "trade_date" not in df_surf.columns:
        raise ValueError("Surface file must contain 'trade_date'")
    if "date" not in df_daily.columns or "close" not in df_daily.columns:
        raise ValueError("Daily file must contain columns: 'date', 'close'")

    df_surf["trade_date"] = pd.to_datetime(df_surf["trade_date"])
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    df = pd.merge(
        df_surf,
        df_daily[["date", "close"]],
        left_on="trade_date",
        right_on="date",
        how="inner",
    ).sort_values("trade_date").reset_index(drop=True)

    # Features (surface only)
    features = ["atm_iv_14d", "skew_25d_14d", "term_structure_30_90", "net_dollar_flow_14d", "iv_change_1d"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in merged df: {missing}")

    # Target
    target_col = "target_future_vol"
    df[target_col] = build_forward_realized_vol_target(df, close_col="close", pred_window=args.pred_window)
    df = df.dropna(subset=features + [target_col]).reset_index(drop=True)

    if len(df) < (args.seq_length + args.pred_window + 50):
        raise ValueError(f"Not enough aligned rows after dropna: {len(df)}")

    # -------------------------
    # Chronological split rows
    # -------------------------
    train_cutoff_row = int(len(df) * 0.8)
    if train_cutoff_row <= args.seq_length:
        raise ValueError("Train cutoff too small relative to seq_length; need more data.")

    train_df = df.iloc[:train_cutoff_row].copy()

    # -------------------------
    # Fit scalers on TRAIN ONLY (no leakage)
    # -------------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_raw = train_df[features].to_numpy(dtype=np.float32)
    y_train_raw = train_df[[target_col]].to_numpy(dtype=np.float32)

    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw)

    X_all_scaled = scaler_X.transform(df[features].to_numpy(dtype=np.float32))
    y_all_scaled = scaler_y.transform(df[[target_col]].to_numpy(dtype=np.float32))

    # -------------------------
    # Build sequences and split sequences by end index
    # -------------------------
    X_train, y_train, X_val, y_val = make_sequences(
        X_scaled=X_all_scaled,
        y_scaled=y_all_scaled,
        seq_length=args.seq_length,
        train_cutoff_row=train_cutoff_row,
    )

    if rank == 0:
        print(f"Rows (after merge+target): {len(df)} | Train rows: {train_cutoff_row} | Val rows: {len(df)-train_cutoff_row}")
        print(f"Train examples: {len(X_train)} | Val examples: {len(X_val)}")
        # Quick degeneracy check
        for c in features:
            nz = float((df[c].to_numpy() != 0).mean())
            print(f"Feature non-zero fraction: {c}: {nz:.3f}")

    # -------------------------
    # DataLoaders
    # -------------------------
    per_rank_bs = max(1, args.global_batch_size // max(1, world_size))
    if args.global_batch_size % max(1, world_size) != 0 and rank == 0:
        print(f"Warning: global_batch_size {args.global_batch_size} not divisible by world_size {world_size}. "
              f"Using per-rank batch size {per_rank_bs} (effective global ~{per_rank_bs*world_size}).")

    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=per_rank_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=per_rank_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    # -------------------------
    # Model
    # -------------------------
    model = BiLSTMRegimeModel(
        input_size=len(features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=False)

    # -------------------------
    # Train
    # -------------------------
    use_amp = (device.type == "cuda") and (not args.no_amp)

    best_val_mse, best_state = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_amp=use_amp,
        rank=rank,
    )

    # -------------------------
    # Compute MAE in real units (optional but useful)
    # -------------------------
    # Evaluate on rank0 only to keep it simple.
    if rank == 0:
        # Load best weights into a non-DDP model for evaluation
        eval_model = BiLSTMRegimeModel(len(features), args.hidden_size, args.num_layers, args.dropout)
        eval_model.load_state_dict(best_state)
        eval_model.to(device)
        eval_model.eval()

        # run val preds
        preds_scaled = []
        ys_scaled = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = eval_model(xb)
                preds_scaled.append(pred.detach().cpu().numpy())
                ys_scaled.append(yb.detach().cpu().numpy())

        preds_scaled = np.vstack(preds_scaled)
        ys_scaled = np.vstack(ys_scaled)

        # invert scaling to realized vol units
        preds = scaler_y.inverse_transform(preds_scaled)
        ys = scaler_y.inverse_transform(ys_scaled)

        mae = float(np.mean(np.abs(preds - ys)))
        rmse = float(np.sqrt(np.mean((preds - ys) ** 2)))
        print(f"Best val MSE (scaled): {best_val_mse:.6f}")
        print(f"Val MAE (realized-vol units): {mae:.6f}")
        print(f"Val RMSE (realized-vol units): {rmse:.6f}")

        meta = TrainMeta(
            seq_length=args.seq_length,
            pred_window=args.pred_window,
            features=features,
            target_col=target_col,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            train_rows=train_cutoff_row,
            val_rows=len(df) - train_cutoff_row,
            train_examples=len(train_ds),
            val_examples=len(val_ds),
            scaler_x_type="MinMaxScaler(train-fit)",
            scaler_y_type="MinMaxScaler(train-fit)",
        )

        ckpt = {
            "state_dict": best_state,
            "best_val_mse_scaled": float(best_val_mse),
            "val_mae_real_units": mae,
            "val_rmse_real_units": rmse,
            "meta": asdict(meta),
            # Store scalers directly (works fine with torch.save / torch.load)
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            # Optional: store last date for sanity
            "data_range": {
                "start": str(df["trade_date"].iloc[0].date()),
                "end": str(df["trade_date"].iloc[-1].date()),
            },
        }

        torch.save(ckpt, args.model_out)
        print(f"Saved checkpoint to: {args.model_out}")

    ddp_cleanup_if_needed(world_size)


if __name__ == "__main__":
    main()

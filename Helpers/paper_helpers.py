import os
import zlib
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Metrics to aggregate (Mean + Std)
AGG_KEYS = [
    "expected_option_price", "expected_pnl", "prob_profit", "VaR_95",
    "mc_sharpe", "downside_sharpe", "p10_value", "p90_value",
    "exit_rate_tp", "exit_rate_sl", "exit_rate_be", "hold_rate", "exit_day_mean"
]


def stable_seed(contract_id: str, base_seed: int, i: int) -> int:
    """
    Generates a deterministic seed based on Contract ID + Index.
    Ensures the same contract gets the same 5 seeds every time.
    """
    h = zlib.crc32(contract_id.encode("utf-8")) & 0xFFFFFFFF
    return (base_seed + h + i + 9973) % (2 ** 32)


def aggregate_metrics(metrics_list):
    """
    Computes Mean and StdDev for all numeric metrics across seeds.
    """
    out = {}
    for k in AGG_KEYS:
        vals = [m.get(k) for m in metrics_list if m.get(k) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std(ddof=0))
    return out


def run_candidate_multi_seed(mc_engine, fusion_output, market_state, contract_id, base_seed, n_seeds):
    metrics_list = []
    per_seed_rows = []

    for i in range(n_seeds):
        seed = stable_seed(contract_id, base_seed, i)
        np.random.seed(seed)

        m = mc_engine.generate_risk_metrics(fusion_output, market_state)

        metrics_list.append(m)
        row = {"seed_idx": i, "seed_val": seed, **m}
        per_seed_rows.append(row)

    agg = aggregate_metrics(metrics_list)
    return per_seed_rows, agg


def ensure_ledger_schema(path, new_headers):
    """
    Checks if the existing ledger header matches new_headers.
    If not, backs up the old file to start fresh.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            existing_header_str = f.readline().strip()
            # Handle empty file case
            if not existing_header_str:
                existing_headers = []
            else:
                existing_headers = existing_header_str.split(",")
    except Exception as e:
        print(f"[LEDGER] Warning: Could not read existing header: {e}")
        existing_headers = []

    # Compare lists (robust check)
    if existing_headers != new_headers:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.replace(".csv", f"_backup_{ts}.csv")
        try:
            os.rename(path, backup_path)
            print(f"\n[LEDGER] 🚨 SCHEMA MISMATCH DETECTED 🚨")
            print(f"Old ledger renamed to: {os.path.basename(backup_path)}")
            print(f"Starting new clean ledger with {len(new_headers)} columns.\n")
        except OSError as e:
            print(f"[LEDGER] Error backing up file: {e}")


def append_ledger_rows(csv_path, rows, headers=None):
    """
    Appends rows to CSV.
    Enforces schema alignment if headers are provided.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df_new = pd.DataFrame(rows)

    # Enforce column order if provided (prevents misalignment)
    if headers:
        df_new = df_new.reindex(columns=headers)

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode='w', header=True, index=False)


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
import os
import zlib
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


def append_ledger_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode='w', header=True, index=False)


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
import pandas as pd
import numpy as np
import os
import uuid


def run_candidate_multi_seed(engine, market_state, fusion_output, contract_id, base_seed=42, n_seeds=5):
    """
    Runs Monte Carlo simulations for a single candidate across varying randomness.
    Signature strictly enforced: engine, market_state, fusion_output
    """
    safe_id = str(contract_id) if contract_id is not None else "UNKNOWN_CONTRACT"
    results = []

    for i in range(n_seeds):
        # FIX 3: Enforce deterministic randomness per iteration so seeds are actually different
        current_seed = base_seed + i
        np.random.seed(current_seed)

        sim_res = engine.generate_risk_metrics(
            fusion_output=fusion_output,
            market_state=market_state
        )

        sim_res['seed_index'] = i
        sim_res['contract_id'] = safe_id
        results.append(sim_res)

    df_res = pd.DataFrame(results)

    # FIX 2: Compute the full, rich suite of metrics expected by run_daily.py
    summary = {
        'expected_pnl_mean': df_res['expected_pnl'].mean() if 'expected_pnl' in df_res else 0.0,
        'prob_profit_mean': df_res['prob_profit'].mean() if 'prob_profit' in df_res else 0.0,
        'downside_sharpe_mean': df_res['downside_sharpe'].mean() if 'downside_sharpe' in df_res else 0.0,
        'Va_R_95_mean': df_res['VaR_95'].mean() if 'VaR_95' in df_res else 0.0,
        'mc_sharpe': df_res['mc_sharpe'].mean() if 'mc_sharpe' in df_res else 0.0,
        'expected_option_price': df_res['expected_option_price'].mean() if 'expected_option_price' in df_res else 0.0,
        'expected_pnl_std': df_res['expected_pnl'].std() if 'expected_pnl' in df_res else 0.0,
        'prob_profit_std': df_res['prob_profit'].std() if 'prob_profit' in df_res else 0.0,
        'downside_sharpe_std': df_res['downside_sharpe'].std() if 'downside_sharpe' in df_res else 0.0,
        'VaR95_std': df_res['VaR_95'].std() if 'VaR_95' in df_res else 0.0,
        'p10_value': df_res['p10_value'].mean() if 'p10_value' in df_res else 0.0,
        'p90_value': df_res['p90_value'].mean() if 'p90_value' in df_res else 0.0,
        'avg_exit_day': df_res['avg_exit_day'].mean() if 'avg_exit_day' in df_res else 0.0,
        'TP_rate': df_res['TP_rate'].mean() if 'TP_rate' in df_res else 0.0,
        'SL_rate': df_res['SL_rate'].mean() if 'SL_rate' in df_res else 0.0,
        'BE_rate': df_res['BE_rate'].mean() if 'BE_rate' in df_res else 0.0,
        'Hold_rate': df_res['Hold_rate'].mean() if 'Hold_rate' in df_res else 0.0,
        'mae_mean': df_res['mae_mean'].mean() if 'mae_mean' in df_res else 0.0,
        'mfe_mean': df_res['mfe_mean'].mean() if 'mfe_mean' in df_res else 0.0
    }

    return results, summary


def append_ledger_rows(file_path, rows, columns):
    """
    FIX 1: Signature updated to accept `columns`.
    Appends rows directly to the CSV without loading the entire file into memory.
    Enforces the exact schema provided in `columns`.
    """
    new_df = pd.DataFrame(rows)

    # Ensure exact column alignment to prevent schema drift
    new_df = new_df.reindex(columns=columns)

    # Write in append mode. Write headers only if the file doesn't exist yet.
    file_exists = os.path.exists(file_path)
    new_df.to_csv(file_path, mode='a', index=False, header=not file_exists)


def make_run_id():
    return str(uuid.uuid4())[:8]
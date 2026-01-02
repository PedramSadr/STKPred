"""
run_daily.py
============
Daily orchestration script for the TSLA options trading system.
UPDATED: Comprehensive Ledger (Market Data, Config, Forecasts, Skips)
"""

import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np

# ============================================================
# 2. CONFIGURATION
# ============================================================
LOGS_DIR = r"C:\My Documents\Mics\Logs"
CATALOG_FILE = os.path.join(LOGS_DIR, "TSLA_Options_Contracts.csv")
LEDGER_FILE = os.path.join(LOGS_DIR, "paper_trade_ledger.csv")

RISK_FREE_RATE = 0.04
NUM_MC_PATHS = 50_000
MAX_CANDIDATES = 20
FAIRNESS_CHECK_MODE = False

# MULTI-SEED CONFIG
BASE_SEED = 42
N_SEEDS = 5
MODEL_VERSION = "Fusion_v1.0"  # Version tag for the ledger

# ============================================================
# 3. PATH & IMPORTS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "CandidateGenerator"))
sys.path.append(os.path.join(PROJECT_ROOT, "MarketStateAdapter"))
sys.path.append(os.path.join(PROJECT_ROOT, "Montecarlo_Sharpe"))
sys.path.append(os.path.join(PROJECT_ROOT, "TradeDecisionBuilder"))
sys.path.append(os.path.join(PROJECT_ROOT, "MLPTrainer"))
sys.path.append(os.path.join(PROJECT_ROOT, "Helpers"))

from candidate_generator import CandidateGenerator, GeneratorConfig
from market_state_adapter import MarketStateAdapter
from MonteCarloEngine import MonteCarloEngine, ExitRules
from TradeDecisionBuilder import TradeDecisionBuilder, DecisionType
from FusionPredictor import FusionPredictor
from paper_helpers import run_candidate_multi_seed, append_ledger_rows, make_run_id

# ============================================================
# MAIN EXECUTION
# ============================================================

# 1. LOAD DATA
print(f"\nLoading catalog from: {CATALOG_FILE}")
if not os.path.exists(CATALOG_FILE):
    raise FileNotFoundError(f"Catalog not found: {CATALOG_FILE}")

catalog = pd.read_csv(CATALOG_FILE, parse_dates=["date", "expiration"])
catalog["date"] = catalog["date"].dt.date
today = date.today()
available_dates = sorted(d for d in catalog["date"].unique() if d <= today)
if not available_dates: raise RuntimeError("No valid dates.")
TRADE_DATE = available_dates[-1]

catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()
catalog_today["row_id"] = catalog_today.index
print(f"Target Trade Date Resolved: {TRADE_DATE}")

# 2. GENERATE CANDIDATES
print(f"\nGenerating candidates...")
gen_config = GeneratorConfig(min_dte=7, max_dte=45, max_candidates=MAX_CANDIDATES)
generator = CandidateGenerator(gen_config)
candidates = generator.generate(catalog_today, trade_date=str(TRADE_DATE))
print(f"Generated {len(candidates)} candidates.")
if not candidates: sys.exit(0)

# 3. PREDICT
print("\nRunning Fusion Model Inference...")
try:
    predictor = FusionPredictor(base_dir=LOGS_DIR)
    fusion_output = predictor.predict(str(TRADE_DATE))

    # [TRACE] Capture Raw Forecasts for Ledger
    raw_mu_log = fusion_output['mu']
    raw_sigma = fusion_output['sigma']
    raw_aiv = fusion_output['aiv']

    # Drift Conversion
    mu_arithmetic = raw_mu_log + 0.5 * (raw_sigma ** 2)
    fusion_output['mu'] = mu_arithmetic

    print(f"  -> Adj. Mu (Arith): {mu_arithmetic:.4f}")
    if FAIRNESS_CHECK_MODE:
        fusion_output.update({'mu': 0.0, 'aiv': 0.0, 'sigma': 0.40})

except Exception as e:
    print(f"CRITICAL PREDICTION FAILURE: {e}")
    sys.exit(1)

# 4. SIMULATE & DECIDE
print(f"\nStarting Evaluation ({NUM_MC_PATHS} paths x {N_SEEDS} seeds)...")

exit_rules = ExitRules(
    use_path_exits=True,
    hold_days=14,
    take_profit_pct=0.80,
    stop_loss_pct=0.50,
    be_trigger_pct=0.25,
    be_exit_buffer=0.01,
    slippage_pct=0.01
)

mc_engine = MonteCarloEngine(num_paths=NUM_MC_PATHS, exit_config=exit_rules)
decision_builder = TradeDecisionBuilder()
adapter = MarketStateAdapter(risk_free_rate=RISK_FREE_RATE)
pred_mu = fusion_output['mu']

RUN_ID = make_run_id()
all_ledger_rows = []

# [NEW] COMPREHENSIVE LEDGER HEADERS
ledger_headers = [
    # Metadata
    "run_id", "timestamp", "trade_date", "row_id", "model_version",
    # Contract Info
    "type", "strike", "dte", "expiration", "contractID",
    # Market Data
    "S0", "IV0", "entry_price", "bid", "ask", "spread_pct", "volume", "open_interest",
    # Forecast Trace
    "mu_log_raw", "sigma_raw", "aiv_10d", "mu_arith_adj",
    # Config Fingerprint
    "hold_days", "mc_mode", "year_days", "iv_daily_std", "iv_min", "iv_max", "slippage",
    "tp_pct", "sl_pct", "be_pct", "be_buffer",
    # MC Outputs (Key Metrics)
    "expected_pnl", "prob_profit", "downside_sharpe", "mc_sharpe", "VaR95", "expected_opt_price",
    # MC Distribution Stats
    "p10_value", "p90_value", "avg_exit_day",
    # Exit Diagnostics
    "TP_rate", "SL_rate", "BE_rate", "Hold_rate",
    # Decision
    "decision", "reason", "seed_type", "seed_val"
]

if not os.path.exists(LEDGER_FILE):
    with open(LEDGER_FILE, 'w', newline='') as f:
        import csv

        csv.writer(f).writerow(ledger_headers)

for idx, candidate in enumerate(candidates, start=1):
    try:
        if hasattr(adapter, 'adapt'):
            market_state = adapter.adapt(candidate)
        else:
            market_state = adapter.build_market_state(candidate)
        market_state['r'] = RISK_FREE_RATE

        leg = candidate['legs'][0]
        cand_type = leg['type'].upper()
        strike = leg['strike']
        dte = leg['dte']
        rid = leg.get('row_id', -1)

        # [NEW] Extract Liquidity Data (Handle missing keys safely)
        bid = leg.get('bid', 0.0)
        ask = leg.get('ask', 0.0)
        vol = leg.get('volume', 0)
        oi = leg.get('open_interest', 0)
        contract_id = leg.get('symbol', f"{cand_type}_{strike}_{dte}")
        entry_price = market_state.get('price', 0.0)

        # Calculate Spread %
        spread_pct = 0.0
        if entry_price > 0:
            spread_pct = (ask - bid) / entry_price

        # Prepare Base Row (Common to SKIP, SEED, AGG)
        base_row = {
            "run_id": RUN_ID,
            "timestamp": datetime.now().isoformat(),
            "trade_date": str(TRADE_DATE),
            "row_id": rid,
            "model_version": MODEL_VERSION,
            "type": cand_type,
            "strike": strike,
            "dte": dte,
            "expiration": leg['expiration'],
            "contractID": contract_id,
            "S0": f"{market_state['S']:.2f}",
            "IV0": f"{market_state['IV']:.4f}",
            "entry_price": f"{entry_price:.2f}",
            "bid": f"{bid:.2f}", "ask": f"{ask:.2f}",
            "spread_pct": f"{spread_pct:.4f}",
            "volume": vol, "open_interest": oi,
            # Forecast Trace
            "mu_log_raw": f"{raw_mu_log:.4f}",
            "sigma_raw": f"{raw_sigma:.4f}",
            "aiv_10d": f"{raw_aiv:.4f}",
            "mu_arith_adj": f"{pred_mu:.4f}",
            # Config Fingerprint
            "hold_days": exit_rules.hold_days,
            "mc_mode": "PATH" if exit_rules.use_path_exits else "TERMINAL",
            "year_days": exit_rules.year_days,
            "iv_daily_std": exit_rules.iv_daily_std,
            "iv_min": exit_rules.iv_min, "iv_max": exit_rules.iv_max,
            "slippage": exit_rules.slippage_pct,
            "tp_pct": exit_rules.take_profit_pct, "sl_pct": exit_rules.stop_loss_pct,
            "be_pct": exit_rules.be_trigger_pct, "be_buffer": exit_rules.be_exit_buffer
        }

        # --- DIRECTIONAL PRE-FILTER (SKIP LOGIC) ---
        skipped = False
        skip_reason = ""
        if not FAIRNESS_CHECK_MODE:
            if pred_mu < 0 and cand_type == 'CALL':
                skipped = True;
                skip_reason = "CALL vs Bearish Signal"
            if pred_mu > 0 and cand_type == 'PUT':
                skipped = True;
                skip_reason = "PUT vs Bullish Signal"

        if skipped:
            print(f"  [Candidate {idx}] SKIPPED ({skip_reason})")
            # Log the SKIP
            skip_row = base_row.copy()
            skip_row.update({
                "decision": "SKIP",
                "reason": skip_reason,
                "seed_type": "N/A", "seed_val": "N/A"
            })
            all_ledger_rows.append(skip_row)
            continue  # Skip Simulation

        # --- SIMULATION (If Not Skipped) ---
        per_seed_rows, agg = run_candidate_multi_seed(
            mc_engine, fusion_output, market_state,
            contract_id=contract_id,
            base_seed=BASE_SEED,
            n_seeds=N_SEEDS
        )

        # Sanity Check (on Aggregate)
        if entry_price > 0 and abs(agg['VaR_95_mean']) > (entry_price * 5):
            print(f"    [WARNING] Sanity Check: VaR ({agg['VaR_95_mean']:.2f}) > 5x Price")

        # Decision
        decision = decision_builder.build_decision(
            expected_pnl=agg['expected_pnl_mean'],
            prob_profit=agg['prob_profit_mean'],
            downside_sharpe=agg['downside_sharpe_mean'],
            cvar=agg['VaR_95_mean'],
            premium=entry_price,
            volatility_consistent=True,
            no_macro_events=True,
            system_healthy=True
        )

        # Print Output
        status = "✅ ACCEPT" if decision.decision == DecisionType.TRADE else "❌ REJECT"
        pnl_share = agg['expected_pnl_mean']
        print(
            f"  [Candidate {idx}] {status} | {cand_type} {strike} (DTE {dte}) | P&L ${pnl_share:.2f} | D_Sharpe {agg['downside_sharpe_mean']:.3f}")

        if decision.decision != DecisionType.TRADE:
            print(f"     [REJECT REASON] {decision.reason}")

        # Add Per-Seed Rows
        for r in per_seed_rows:
            full_row = base_row.copy()
            full_row.update({
                "seed_type": "INDIVIDUAL",
                "seed_val": r["seed_val"],
                "decision": "N/A", "reason": "N/A",
                "expected_pnl": f"{r['expected_pnl']:.4f}",
                "prob_profit": f"{r['prob_profit']:.4f}",
                "downside_sharpe": f"{r['downside_sharpe']:.4f}",
                "mc_sharpe": f"{r['mc_sharpe']:.4f}",
                "VaR95": f"{r['VaR_95']:.4f}",
                "expected_opt_price": f"{r['expected_option_price']:.4f}",
                "p10_value": f"{r['p10_value']:.4f}", "p90_value": f"{r['p90_value']:.4f}",
                "TP_rate": r.get("exit_rate_tp", 0), "SL_rate": r.get("exit_rate_sl", 0),
                "BE_rate": r.get("exit_rate_be", 0), "Hold_rate": r.get("hold_rate", 0),
                "avg_exit_day": r.get("exit_day_mean", 0)
            })
            all_ledger_rows.append(full_row)

        # Add Aggregate Row
        agg_row = base_row.copy()
        agg_row.update({
            "seed_type": "AGGREGATE",
            "seed_val": "MEAN",
            "decision": decision.decision.value,
            "reason": decision.reason,
            "expected_pnl": f"{agg['expected_pnl_mean']:.4f}",
            "prob_profit": f"{agg['prob_profit_mean']:.4f}",
            "downside_sharpe": f"{agg['downside_sharpe_mean']:.4f}",
            "mc_sharpe": f"{agg['mc_sharpe_mean']:.4f}",
            "VaR95": f"{agg['VaR_95_mean']:.4f}",
            "expected_opt_price": f"{agg['expected_option_price_mean']:.4f}",
            "p10_value": f"{agg['p10_value_mean']:.4f}", "p90_value": f"{agg['p90_value_mean']:.4f}",
            "TP_rate": f"{agg.get('exit_rate_tp_mean', 0):.2f}",
            "SL_rate": f"{agg.get('exit_rate_sl_mean', 0):.2f}",
            "BE_rate": f"{agg.get('exit_rate_be_mean', 0):.2f}",
            "Hold_rate": f"{agg.get('hold_rate_mean', 0):.2f}",
            "avg_exit_day": f"{agg.get('exit_day_mean_mean', 0):.1f}"
        })
        all_ledger_rows.append(agg_row)

    except Exception as e:
        print(f"  [Candidate {idx}] Failed: {e}")

# WRITE LEDGER BATCH
if all_ledger_rows:
    append_ledger_rows(LEDGER_FILE, all_ledger_rows)
    print(f"\n[LEDGER] Saved {len(all_ledger_rows)} rows to {LEDGER_FILE}")

print("\nDaily orchestration complete.")
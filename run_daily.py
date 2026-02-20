import os
import sys
import traceback
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# CONFIG
LOGS_DIR = r"C:\My Documents\Mics\Logs"
CATALOG_FILE = os.path.join(LOGS_DIR, "TSLA_Options_Contracts.csv")
LEDGER_FILE = os.path.join(LOGS_DIR, "paper_trade_ledger.csv")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure Python can find all the submodules
for folder in ["CandidateGenerator", "MarketStateAdapter", "Montecarlo_Sharpe", "TradeDecisionBuilder", "Helpers"]:
    sys.path.append(os.path.join(PROJECT_ROOT, folder))

from Montecarlo_Sharpe.MonteCarloEngine import MonteCarloEngine, ExitRules
from Helpers.paper_helpers import run_candidate_multi_seed, append_ledger_rows, make_run_id
from MLPTrainer.FusionPredictor import FusionPredictor
from CandidateGenerator.candidate_generator import CandidateGenerator, GeneratorConfig
from MarketStateAdapter.market_state_adapter import MarketStateAdapter

# =========================================================
# SCHEMA ENFORCEMENT
# =========================================================
LEDGER_COLUMNS = [
    "run_id", "timestamp", "trade_date", "row_id", "model_version", "type", "strike", "dte",
    "expiration", "contractID", "S0", "IV0", "entry_price", "mid_price", "bid", "ask", "spread_abs",
    "spread_pct", "volume", "open_interest", "mu_log_raw", "sigma_raw", "aiv_10d", "mu_arith_adj",
    "hold_days", "mc_mode", "year_days", "iv_daily_std", "iv_min", "iv_max", "slippage", "tp_pct",
    "sl_pct", "be_pct", "be_buffer", "expected_pnl", "prob_profit", "downside_sharpe", "mc_sharpe",
    "VaR95", "expected_opt_price", "expected_pnl_std", "prob_profit_std", "downside_sharpe_std",
    "VaR95_std", "p10_value", "p90_value", "avg_exit_day", "TP_rate", "SL_rate", "BE_rate", "Hold_rate",
    "decision", "reason", "seed_type", "seed_val", "ev_dollars", "spread_cost_dollars", "min_required_edge",
    "limit_price", "fill_price", "slippage_vs_mid", "entry_reason", "exit_reason", "mae_mean", "mfe_mean"
]


def format_schema_row(data_dict):
    """Forces the dictionary to strictly match the 66-column schema."""
    return {col: data_dict.get(col, "") for col in LEDGER_COLUMNS}


# =========================================================
# 1. LOAD DATA & ROBUST DATE GUARD
# =========================================================
catalog = pd.read_csv(CATALOG_FILE, low_memory=False)

# Drop NaT before conversion to prevent poisoning
catalog["date"] = pd.to_datetime(catalog["date"], errors="coerce")
catalog = catalog.dropna(subset=["date"])
catalog["date"] = catalog["date"].dt.date

today = date.today()
valid_dates = sorted(d for d in catalog["date"].unique() if d <= today)
if not valid_dates:
    print("CRITICAL: No valid historical dates found in catalog. Exiting pipeline.")
    sys.exit(1)

TRADE_DATE = valid_dates[-1]

cal_lag = (today - TRADE_DATE).days
bus_lag = np.busday_count(str(TRADE_DATE), str(today))

if bus_lag > 2 or cal_lag > 3:
    print(f"Data is stale. TRADE_DATE={TRADE_DATE} (bus_lag={bus_lag}, cal_lag={cal_lag}). Exiting.")
    sys.exit(1)

catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()
catalog_today["row_id"] = catalog_today.index

# =========================================================
# 2. UNDERLYING PRICE INJECTION
# =========================================================
print(f"--- Running Daily Strategy for: {TRADE_DATE} ---")
spot_price = np.nan
try:
    ticker = yf.Ticker("TSLA")
    start_dt = pd.Timestamp(TRADE_DATE)
    end_dt = start_dt + pd.Timedelta(days=4)  # Weekend/holiday buffer
    hist = ticker.history(start=start_dt, end=end_dt)

    if not hist.empty:
        hist = hist.copy()

        # Safely strip timezone
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_convert(None)

        hist_dates = pd.Index(hist.index.date)
        is_exact = (TRADE_DATE in hist_dates)

        after = hist.loc[hist_dates >= TRADE_DATE]

        if not after.empty:
            spot_price = float(after["Close"].iloc[0])
            match_type = "exactly matched" if is_exact else "closest forward"
            print(f"[Data] yfinance {match_type} spot price: {spot_price:.2f}")
        else:
            spot_price = float(hist["Close"].iloc[-1])
            print(f"[Data] yfinance fallback (last available) spot price: {spot_price:.2f}")

except Exception as e:
    print(f"[Warning] yfinance fetch failed: {e}")

if (pd.isna(spot_price) or spot_price <= 0) and "underlying_price" in catalog_today.columns:
    fallback = catalog_today.loc[catalog_today["underlying_price"] > 0, "underlying_price"]
    if not fallback.empty:
        spot_price = float(fallback.median())
        print(f"[Data] fallback catalog median spot: {spot_price:.2f}")

if pd.isna(spot_price) or spot_price <= 0:
    print("CRITICAL: Could not resolve underlying price. Exiting pipeline.")
    sys.exit(1)

catalog_today["underlying_price"] = spot_price

# =========================================================
# 3. RUN PIPELINE INITIALIZATION
# =========================================================
DAILY_RUN_ID = make_run_id()

candidates = CandidateGenerator(GeneratorConfig(max_candidates=20)).generate(catalog_today, trade_date=str(TRADE_DATE))

# --- PREDICTOR FALLBACK ---
try:
    predictor = FusionPredictor(base_dir=LOGS_DIR)
    fusion_output = predictor.predict(str(TRADE_DATE))
    print(
        f"AI Prediction Loaded: mu={fusion_output['mu']:.4f}, sigma={fusion_output['sigma']:.4f}, aiv={fusion_output['aiv']:.4f}")
except Exception as e:
    print(f"WARNING: FusionPredictor failed ({e}). Using safe neutral fallbacks.")
    fusion_output = {"mu": 0.0, "sigma": 0.50, "aiv": 0.0, "raw_mu_10d": 0.0}

mc_engine = MonteCarloEngine(num_paths=50000, exit_config=ExitRules())
all_ledger_rows = []

# =========================================================
# 4. EXECUTION LOOP
# =========================================================
for idx, candidate in enumerate(candidates, start=1):
    # Guard against silently simulating blocked rows
    if candidate.get("decision", "TRADE") == "BLOCK":
        continue

    try:
        candidate.update(fusion_output)
        leg = candidate['legs'][0]
        opt_type = str(leg.get("type", "")).lower()

        # --- DIRECTIONAL FAIRNESS FILTER ---
        mu_arith = fusion_output.get('mu', 0.0)
        if mu_arith < 0.0 and opt_type == "call":
            print(f"  [Candidate {idx}] Skipped Call due to bearish AI signal (mu={mu_arith:.4f}).")
            continue
        if mu_arith > 0.0 and opt_type == "put":
            print(f"  [Candidate {idx}] Skipped Put due to bullish AI signal (mu={mu_arith:.4f}).")
            continue

        # --- CANONICAL BROKER FORMATTING ---
        exp_raw = leg.get("expiration", "")
        exp_yymmdd = pd.to_datetime(exp_raw).strftime("%y%m%d") if pd.notna(exp_raw) and exp_raw else "XXXXXX"
        cp_flag = opt_type[0].upper() if opt_type else "X"
        strike_val = float(leg.get("strike", 0.0))

        strike_str = f"{strike_val:.1f}"
        contract_id = f"TSLA{exp_yymmdd}{cp_flag}{strike_str}"

        market_state = MarketStateAdapter(0.04).adapt(candidate)
        per_seed_rows, agg = run_candidate_multi_seed(mc_engine, market_state, fusion_output, contract_id=contract_id)

        bid = float(leg.get("bid", 0.0))
        ask = float(leg.get("ask", 0.0))
        entry_price = float(leg.get("entry_price", 0.0))
        mid_price = (bid + ask) / 2.0 if ask > 0 else entry_price
        spread_abs = ask - bid if ask > 0 else 0.0
        spread_pct = spread_abs / entry_price if entry_price > 0 else 0.0

        base_info = {
            "run_id": DAILY_RUN_ID,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_date": str(TRADE_DATE),
            "row_id": leg.get("row_id", ""),
            "model_version": "FusionMLP_v1",
            "type": leg.get("type", ""),
            "strike": strike_val,
            "dte": leg.get("dte", 0),
            "expiration": leg.get("expiration", ""),
            "contractID": contract_id,
            "S0": candidate.get("underlying_price", 0.0),  # FIXED: Pulls float from Candidate dictionary
            "IV0": leg.get("iv", 0.0),
            "entry_price": entry_price,
            "mid_price": mid_price,
            "bid": bid,
            "ask": ask,
            "spread_abs": spread_abs,
            "spread_pct": spread_pct,
            "volume": leg.get("volume", 0),
            "open_interest": leg.get("open_interest", 0),
            "mu_log_raw": fusion_output.get("raw_mu_10d", 0.0),
            "sigma_raw": fusion_output.get("sigma", 0.0),
            "aiv_10d": fusion_output.get("aiv", 0.0),
            "mu_arith_adj": mu_arith,
            "hold_days": 14,
            "mc_mode": "path",
            "year_days": 252.0,
            "iv_daily_std": 0.01,
            "iv_min": 0.01,
            "iv_max": 3.00,
            "slippage": 0.01,
            "tp_pct": 0.50,
            "sl_pct": 0.30,
            "be_pct": 0.20,
            "be_buffer": 0.0,
            "decision": candidate.get("decision", "TRADE"),
            "reason": candidate.get("block_reason", "Passed Evaluation")
        }

        # 1. Log Individual Seeds
        for seed_res in per_seed_rows:
            ind_row = base_info.copy()
            ind_row.update({
                "expected_pnl": seed_res.get("expected_pnl", 0.0),
                "prob_profit": seed_res.get("prob_profit", 0.0),
                "downside_sharpe": seed_res.get("downside_sharpe", 0.0),
                "VaR95": seed_res.get("VaR_95", 0.0),
                "expected_opt_price": seed_res.get("expected_option_price", 0.0),
                "seed_type": "INDIVIDUAL",
                "seed_val": seed_res.get("seed_index", "")
            })
            all_ledger_rows.append(format_schema_row(ind_row))

        # 2. Log Aggregate Result
        agg_row = base_info.copy()
        agg_row.update({
            "expected_pnl": agg.get("expected_pnl_mean", 0.0),
            "prob_profit": agg.get("prob_profit_mean", 0.0),
            "downside_sharpe": agg.get("downside_sharpe_mean", 0.0),
            "mc_sharpe": agg.get("mc_sharpe", 0.0),
            "VaR95": agg.get("Va_R_95_mean", 0.0),
            "expected_opt_price": agg.get("expected_option_price", 0.0),
            "expected_pnl_std": agg.get("expected_pnl_std", 0.0),
            "prob_profit_std": agg.get("prob_profit_std", 0.0),
            "downside_sharpe_std": agg.get("downside_sharpe_std", 0.0),
            "VaR95_std": agg.get("VaR95_std", 0.0),
            "p10_value": agg.get("p10_value", 0.0),
            "p90_value": agg.get("p90_value", 0.0),
            "avg_exit_day": agg.get("avg_exit_day", 0.0),
            "TP_rate": agg.get("TP_rate", 0.0),
            "SL_rate": agg.get("SL_rate", 0.0),
            "BE_rate": agg.get("BE_rate", 0.0),
            "Hold_rate": agg.get("Hold_rate", 0.0),
            "seed_type": "AGGREGATE",
            "seed_val": "ALL",
            "ev_dollars": agg.get("expected_pnl_mean", 0.0) * 100,
            "spread_cost_dollars": spread_abs * 100,
            "mae_mean": agg.get("mae_mean", 0.0),
            "mfe_mean": agg.get("mfe_mean", 0.0)
        })
        all_ledger_rows.append(format_schema_row(agg_row))

        print(f"  [Candidate {idx}] {contract_id} simulated successfully.")

    except Exception as e:
        print(f"\n  [Candidate {idx}] FAILED: {type(e).__name__} - {e}")
        traceback.print_exc(limit=3)
        print("-" * 40)

# =========================================================
# 5. SAVE RESULTS
# =========================================================
if all_ledger_rows:
    append_ledger_rows(LEDGER_FILE, all_ledger_rows, LEDGER_COLUMNS)
    print(f"\n[SUCCESS] Appended {len(all_ledger_rows)} rows to ledger.")
else:
    print("\n[!] No trades were successfully processed.")
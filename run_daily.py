"""
run_daily.py
============
Daily orchestration script for the TSLA options trading system.
UPDATED: Symbols set to CUSTOM format (TSLA260213P445.0) for compatibility.
"""

import os
import sys
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# 2. CONFIGURATION
# ============================================================
LOGS_DIR = r"C:\My Documents\Mics\Logs"
CATALOG_FILE = os.path.join(LOGS_DIR, "TSLA_Options_Contracts.csv")
LEDGER_FILE = os.path.join(LOGS_DIR, "paper_trade_ledger.csv")

UNDERLYING_SYMBOL = "TSLA"
RISK_FREE_RATE = 0.04
NUM_MC_PATHS = 50_000
MAX_CANDIDATES = 20
FAIRNESS_CHECK_MODE = False
BASE_SEED = 42
N_SEEDS = 5
MODEL_VERSION = "Fusion_v1.0"
EXECUTION_SPREAD_FRACTION = 0.25  # Fraction of spread paid to get filled (0.25 = limit order slightly worse than mid)

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

try:
    from candidate_generator import CandidateGenerator, GeneratorConfig
    from market_state_adapter import MarketStateAdapter
    from MonteCarloEngine import MonteCarloEngine, ExitRules
    from TradeDecisionBuilder import TradeDecisionBuilder, DecisionType
    from FusionPredictor import FusionPredictor
    from paper_helpers import run_candidate_multi_seed, append_ledger_rows, ensure_ledger_schema, make_run_id
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


# ============================================================
# HELPER: CUSTOM SYMBOL GENERATOR
# ============================================================
def get_occ_symbol(underlying, expiration, op_type, strike):
    """
    Generates CUSTOM Symbol Format: Root + YYMMDD + Type + Strike(float string)
    Example: TSLA260213P445.0
    """
    try:
        dt = pd.to_datetime(expiration)
        yymmdd = dt.strftime('%y%m%d')
        type_char = 'C' if op_type.upper() == 'CALL' else 'P'

        # FORCE Custom Format: 445 -> "445.0"
        strike_str = str(float(strike))

        root = underlying.strip().upper()
        return f"{root}{yymmdd}{type_char}{strike_str}"
    except Exception:
        # Fallback
        return f"{underlying}_{op_type}_{strike}_{expiration}"


# ============================================================
# MAIN EXECUTION
# ============================================================

# 1. LOAD DATA (ROBUST VERSION)
print(f"\nLoading catalog from: {CATALOG_FILE}")
if not os.path.exists(CATALOG_FILE):
    raise FileNotFoundError(f"Catalog not found: {CATALOG_FILE}")

catalog = pd.read_csv(CATALOG_FILE, low_memory=False)

# Robust Date Parsing
catalog["date"] = pd.to_datetime(catalog["date"], errors='coerce')
catalog = catalog.dropna(subset=['date'])
catalog["date"] = catalog["date"].dt.date

if "expiration" in catalog.columns:
    catalog["expiration"] = pd.to_datetime(catalog["expiration"], errors='coerce')

today = date.today()
available_dates = sorted(d for d in catalog["date"].unique() if pd.notna(d) and d <= today)

if not available_dates:
    raise RuntimeError("No valid dates found in catalog.")

TRADE_DATE = available_dates[-1]
print(f"Target Trade Date Resolved: {TRADE_DATE}")

# Staleness Check (Upgraded to Business Days)
b_days_lag = np.busday_count(str(TRADE_DATE), str(today))
if b_days_lag > 2:  # 2 business days safely covers a 3-day holiday weekend
    print(f"\n[CRITICAL ERROR] Data is stale by {b_days_lag} business days!")
    print(f"  Latest available date: {TRADE_DATE}")
    sys.exit(1)

catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()
catalog_today["row_id"] = catalog_today.index

# --- INJECT UNDERLYING PRICE (ROBUST) ---
print(f"Fetching {UNDERLYING_SYMBOL} price for {TRADE_DATE} via yfinance...")
found_price = False
spot_price = 0.0

try:
    start_dt = pd.Timestamp(TRADE_DATE)
    end_dt = start_dt + timedelta(days=2)

    ticker = yf.Ticker(UNDERLYING_SYMBOL)
    hist = ticker.history(start=start_dt, end=end_dt)

    if not hist.empty:
        hist.index = hist.index.date
        if TRADE_DATE in hist.index:
            spot_price = float(hist.loc[TRADE_DATE]['Close'])
            print(f"  ✅ Found Close Price: ${spot_price:.2f}")
            found_price = True
        else:
            spot_price = float(hist['Close'].iloc[0])
            print(f"  ⚠️ Exact date match failed. Using closest: ${spot_price:.2f}")
            found_price = True
    else:
        print("  ❌ yfinance returned no data.")

except Exception as e:
    print(f"  ❌ Price Fetch Failed: {e}")

if found_price and spot_price > 0:
    catalog_today['underlying_price'] = spot_price
else:
    # FALLBACK: Check if catalog already has a valid price
    if "underlying_price" in catalog_today.columns:
        valid_prices = catalog_today[catalog_today["underlying_price"] > 0]["underlying_price"]
        if not valid_prices.empty:
            fallback_price = valid_prices.median()
            print(f"  ⚠️ Using median price from catalog: ${fallback_price:.2f}")
            catalog_today['underlying_price'] = fallback_price
        else:
            print("[CRITICAL] No valid underlying price found in YFinance OR Catalog. Aborting.")
            sys.exit(1)
    else:
        print("[CRITICAL] underlying_price column missing and YFinance failed. Aborting.")
        sys.exit(1)

# ============================================================
# --- UPGRADE: INITIALIZE EXIT RULES EARLY FOR DYNAMIC DTE ---
# ============================================================
exit_rules = ExitRules(
    use_path_exits=True,
    hold_days=14,
    take_profit_pct=0.80,
    stop_loss_pct=0.50,
    be_trigger_pct=0.25,
    be_exit_buffer=0.01,
    slippage_pct=0.01
)

# 2. GENERATE CANDIDATES
print(f"\nGenerating candidates...")

try:
    # Dynamically scale minimum DTE based on the target holding period + 3 days buffer
    min_dte_effective = max(10, exit_rules.hold_days + 3)

    gen_config = GeneratorConfig(
        min_dte=min_dte_effective,
        max_dte=45,
        max_candidates=MAX_CANDIDATES,
        min_vol=0,
        min_oi=0
    )
    generator = CandidateGenerator(gen_config)
except TypeError:
    print("Warning: Using Default CandidateGenerator.")
    generator = CandidateGenerator()

candidates = generator.generate(catalog_today, trade_date=str(TRADE_DATE))
print(f"Generated {len(candidates)} candidates.")

if not candidates:
    print("No candidates met the criteria.")
    sys.exit(0)

# 3. PREDICT
print("\nRunning Fusion Model Inference...")
try:
    predictor = FusionPredictor(base_dir=LOGS_DIR)
    fusion_output = predictor.predict(str(TRADE_DATE))

    raw_mu_log = fusion_output['mu']
    raw_sigma = fusion_output['sigma']
    raw_aiv = fusion_output['aiv']

    mu_arithmetic = raw_mu_log + 0.5 * (raw_sigma ** 2)
    fusion_output['mu'] = mu_arithmetic

    print(f"  -> Model Prediction: Adj. Mu {mu_arithmetic:.4f} | Sigma {raw_sigma:.4f}")

except Exception as e:
    print(f"  ⚠️ PREDICTION FAILED: {e}")
    print("  -> ACTION: Switching to FALLBACK MODE (Neutral Assumptions)")

    raw_mu_log = RISK_FREE_RATE
    raw_sigma = 0.40
    raw_aiv = 0.0

    fusion_output = {
        'mu': raw_mu_log,
        'sigma': raw_sigma,
        'aiv': raw_aiv
    }
    fusion_output['mu'] = raw_mu_log + 0.5 * (raw_sigma ** 2)

    print(f"  -> Fallback: Mu {fusion_output['mu']:.4f} | Sigma {fusion_output['sigma']:.4f}")

# 4. SIMULATE & DECIDE
print(f"\nStarting Evaluation ({NUM_MC_PATHS} paths x {N_SEEDS} seeds)...")

mc_engine = MonteCarloEngine(num_paths=NUM_MC_PATHS, exit_config=exit_rules)
decision_builder = TradeDecisionBuilder()
adapter = MarketStateAdapter(risk_free_rate=RISK_FREE_RATE)

RUN_ID = make_run_id()
all_ledger_rows = []
accepted_trades = []

ledger_headers = [
    "run_id", "timestamp", "trade_date", "row_id", "model_version",
    "type", "strike", "dte", "expiration", "contractID",
    "S0", "IV0", "entry_price", "mid_price", "bid", "ask", "spread_abs", "spread_pct", "volume", "open_interest",
    "mu_log_raw", "sigma_raw", "aiv_10d", "mu_arith_adj",
    "hold_days", "mc_mode", "year_days", "iv_daily_std", "iv_min", "iv_max", "slippage",
    "tp_pct", "sl_pct", "be_pct", "be_buffer",
    "expected_pnl", "prob_profit", "downside_sharpe", "mc_sharpe", "VaR95", "expected_opt_price",
    "expected_pnl_std", "prob_profit_std", "downside_sharpe_std", "VaR95_std",
    "p10_value", "p90_value", "avg_exit_day",
    "TP_rate", "SL_rate", "BE_rate", "Hold_rate",
    "decision", "reason", "seed_type", "seed_val",
    "ev_dollars", "spread_cost_dollars", "min_required_edge",
    "limit_price", "fill_price", "slippage_vs_mid",
    "entry_reason", "exit_reason", "mae_mean", "mfe_mean"
]

ensure_ledger_schema(LEDGER_FILE, ledger_headers)

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
        expiration = leg['expiration']

        bid = leg.get('bid', 0.0)
        ask = leg.get('ask', 0.0)
        vol = leg.get('volume', 0)
        oi = leg.get('open_interest', 0)

        # Original default from data
        raw_entry_price = market_state.get('price', 0.0)

        skipped = False
        skip_reason = ""
        spread_abs = 0.0

        if bid <= 0 or ask <= 0 or ask < bid:
            skipped = True
            skip_reason = "Missing/invalid quotes; cannot price execution cost"
            mid_price = raw_entry_price
            spread_pct = 0.0

            # Purely for reporting clarity: Zero out execution logs for skipped candidates
            limit_price = 0.0
            fill_price = 0.0
            slippage_vs_mid = 0.0
            entry_price = raw_entry_price
        else:
            spread_abs = ask - bid
            mid_price = (bid + ask) / 2.0
            spread_pct = spread_abs / mid_price

            # --- THE NEW EXECUTION MODEL ---
            limit_price = mid_price + (EXECUTION_SPREAD_FRACTION * spread_abs)
            fill_price = min(ask, limit_price)
            slippage_vs_mid = fill_price - mid_price

            # Override the pipeline's master entry price so EV and Greeks match reality
            entry_price = fill_price
            market_state['price'] = fill_price

        contract_id = leg.get("contractID")
        if not contract_id:
            contract_id = get_occ_symbol(UNDERLYING_SYMBOL, expiration, cand_type, strike)

        base_row = {
            "run_id": RUN_ID,
            "timestamp": datetime.now().isoformat(),
            "trade_date": str(TRADE_DATE),
            "row_id": rid,
            "model_version": MODEL_VERSION,
            "type": cand_type,
            "strike": strike,
            "dte": dte,
            "expiration": expiration,
            "contractID": contract_id,
            "S0": f"{market_state['S']:.2f}",
            "IV0": f"{market_state['IV']:.4f}",
            "entry_price": f"{entry_price:.2f}",
            "mid_price": f"{mid_price:.2f}",
            "bid": f"{bid:.2f}", "ask": f"{ask:.2f}",
            "spread_abs": f"{spread_abs:.2f}",
            "spread_pct": f"{spread_pct:.4f}",
            "volume": vol, "open_interest": oi,
            "mu_log_raw": f"{raw_mu_log:.4f}",
            "sigma_raw": f"{raw_sigma:.4f}",
            "aiv_10d": f"{raw_aiv:.4f}",
            "mu_arith_adj": f"{fusion_output['mu']:.4f}",
            "hold_days": exit_rules.hold_days,
            "mc_mode": "PATH" if exit_rules.use_path_exits else "TERMINAL",
            "year_days": exit_rules.year_days,
            "iv_daily_std": exit_rules.iv_daily_std,
            "iv_min": exit_rules.iv_min, "iv_max": exit_rules.iv_max,
            "slippage": exit_rules.slippage_pct,
            "tp_pct": exit_rules.take_profit_pct, "sl_pct": exit_rules.stop_loss_pct,
            "be_pct": exit_rules.be_trigger_pct, "be_buffer": exit_rules.be_exit_buffer,
            "ev_dollars": "0.00", "spread_cost_dollars": "0.00", "min_required_edge": "0.00",
            "limit_price": f"{limit_price:.2f}",
            "fill_price": f"{fill_price:.2f}",
            "slippage_vs_mid": f"{slippage_vs_mid:.2f}",
            "entry_reason": "N/A",
            "exit_reason": "N/A",
            "mae_mean": "0.00",
            "mfe_mean": "0.00"
        }

        MU_EPS = 0.001

        if not FAIRNESS_CHECK_MODE and not skipped:
            if raw_mu_log < -MU_EPS and cand_type == 'CALL':
                skipped = True
                skip_reason = f"CALL vs Bearish Signal (raw_mu_log < {-MU_EPS})"
            if raw_mu_log > MU_EPS and cand_type == 'PUT':
                skipped = True
                skip_reason = f"PUT vs Bullish Signal (raw_mu_log > {MU_EPS})"

        if skipped:
            print(f"  [Candidate {idx}] SKIPPED ({skip_reason})")
            skip_row = base_row.copy()
            skip_row.update({
                "decision": "SKIP",
                "reason": skip_reason,
                "seed_type": "N/A", "seed_val": "N/A",
                "expected_pnl": 0, "prob_profit": 0, "downside_sharpe": 0
            })
            all_ledger_rows.append(skip_row)
            continue

        # --- SIMULATION ---
        per_seed_rows, agg = run_candidate_multi_seed(
            mc_engine, fusion_output, market_state,
            contract_id=contract_id,
            base_seed=BASE_SEED,
            n_seeds=N_SEEDS
        )

        decision = decision_builder.build_decision(
            expected_pnl=agg['expected_pnl_mean'],
            prob_profit=agg['prob_profit_mean'],
            downside_sharpe=agg['downside_sharpe_mean'],
            cvar=agg['VaR_95_mean'],
            premium=entry_price,
            spread_abs=spread_abs,
            downside_sharpe_std=agg['downside_sharpe_std'],
            volatility_consistent=True,
            no_macro_events=True,
            system_healthy=True
        )

        status = "✅ ACCEPT" if decision.decision == DecisionType.TRADE else "❌ REJECT"
        print(f"  [Candidate {idx}] {status} | {contract_id}")

        if decision.decision == DecisionType.TRADE:
            composite_score = agg['downside_sharpe_mean'] - (0.5 * agg['downside_sharpe_std'])
            if abs(agg['VaR_95_mean']) > 1e-6:
                composite_score += 0.2 * (agg['expected_pnl_mean'] / abs(agg['VaR_95_mean']))

            accepted_trades.append({
                "id": contract_id,
                "desc": f"{cand_type} {strike} DTE {dte}",
                "score": composite_score,
                "ds": agg['downside_sharpe_mean'],
                "ev": agg['expected_pnl_mean'],
                "var": agg['VaR_95_mean']
            })

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
                "avg_exit_day": r.get("exit_day_mean", 0),
                "mae_mean": f"{r.get('mae', 0.0):.2f}",
                "mfe_mean": f"{r.get('mfe', 0.0):.2f}"
            })
            all_ledger_rows.append(full_row)

        # Map Dominant Exit Reason
        tp_r = agg.get('exit_rate_tp_mean', 0)
        sl_r = agg.get('exit_rate_sl_mean', 0)
        be_r = agg.get('exit_rate_be_mean', 0)
        h_r  = agg.get('hold_rate_mean', 0)
        rates = {'TP': tp_r, 'SL': sl_r, 'BE': be_r, 'HOLD': h_r}
        dom_exit = max(rates, key=rates.get) if max(rates.values()) > 0 else "N/A"

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
            "expected_pnl_std": f"{agg['expected_pnl_std']:.4f}",
            "prob_profit_std": f"{agg['prob_profit_std']:.4f}",
            "downside_sharpe_std": f"{agg['downside_sharpe_std']:.4f}",
            "VaR95_std": f"{agg['VaR_95_std']:.4f}",
            "TP_rate": f"{tp_r:.2f}",
            "SL_rate": f"{sl_r:.2f}",
            "BE_rate": f"{be_r:.2f}",
            "Hold_rate": f"{h_r:.2f}",
            "avg_exit_day": f"{agg.get('exit_day_mean_mean', 0):.1f}",
            "ev_dollars": f"{decision.metrics_snapshot.get('ev_dollars', 0.0):.2f}",
            "spread_cost_dollars": f"{decision.metrics_snapshot.get('spread_cost_dollars', 0.0):.2f}",
            "min_required_edge": f"{decision.metrics_snapshot.get('min_required_edge', 0.0):.2f}",
            "entry_reason": decision.reason,
            "exit_reason": dom_exit,
            "mae_mean": f"{agg.get('mae_mean', 0.0):.2f}",
            "mfe_mean": f"{agg.get('mfe_mean', 0.0):.2f}"
        })
        all_ledger_rows.append(agg_row)

    except Exception as e:
        print(f"  [Candidate {idx}] Failed: {e}")

# WRITE LEDGER BATCH
if all_ledger_rows:
    append_ledger_rows(LEDGER_FILE, all_ledger_rows, headers=ledger_headers)
    print(f"\n[LEDGER] Saved {len(all_ledger_rows)} rows to {LEDGER_FILE}")

# SUMMARY
if accepted_trades:
    print("\n" + "=" * 80)
    print("🏆 TOP ACCEPTED TRADES")
    print("=" * 80)
    accepted_trades.sort(key=lambda x: x['score'], reverse=True)
    for rank, t in enumerate(accepted_trades, start=1):
        print(f"#{rank} {t['id']} | Score: {t['score']:.4f} | EV: ${t['ev']:.2f}")
else:
    print("\n⚠️  NO TRADES ACCEPTED TODAY")

print("\nDaily orchestration complete.")
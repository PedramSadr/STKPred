"""
run_daily.py
============
Daily orchestration script for the TSLA options trading system.
UPDATED: Robust Data Loading + Smart Ranking + Staleness Guardrails
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

UNDERLYING_SYMBOL = "TSLA"
RISK_FREE_RATE = 0.04
NUM_MC_PATHS = 50_000
MAX_CANDIDATES = 20
FAIRNESS_CHECK_MODE = False
BASE_SEED = 42
N_SEEDS = 5
MODEL_VERSION = "Fusion_v1.0"

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

# Import your modules (Ensure these files exist in the paths above)
try:
    from candidate_generator import CandidateGenerator, GeneratorConfig
    from market_state_adapter import MarketStateAdapter
    from MonteCarloEngine import MonteCarloEngine, ExitRules
    from TradeDecisionBuilder import TradeDecisionBuilder, DecisionType
    from FusionPredictor import FusionPredictor
    from paper_helpers import run_candidate_multi_seed, append_ledger_rows, ensure_ledger_schema, make_run_id
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure all helper modules (CandidateGenerator, MonteCarloEngine, etc.) are in the correct folders.")
    sys.exit(1)


# ============================================================
# HELPER: OCC SYMBOL GENERATOR
# ============================================================
def get_occ_symbol(underlying, expiration, op_type, strike):
    try:
        dt = pd.to_datetime(expiration)
        yymmdd = dt.strftime('%y%m%d')
        type_char = 'C' if op_type.upper() == 'CALL' else 'P'
        strike_int = int(strike * 1000)
        root = underlying.strip().upper()
        return f"{root}{yymmdd}{type_char}{strike_int:08d}"
    except Exception:
        return f"{underlying}_{op_type}_{strike}_{expiration}"


# ============================================================
# MAIN EXECUTION
# ============================================================

# 1. LOAD DATA (ROBUST VERSION)
print(f"\nLoading catalog from: {CATALOG_FILE}")
if not os.path.exists(CATALOG_FILE):
    raise FileNotFoundError(f"Catalog not found: {CATALOG_FILE}")

# [FIX] Read with low_memory=False to avoid mixed-type warnings
catalog = pd.read_csv(CATALOG_FILE, low_memory=False)

# [FIX] Coerce dates to datetime. Bad rows (like text headers) become NaT.
catalog["date"] = pd.to_datetime(catalog["date"], errors='coerce')

# [FIX] Drop rows where date is invalid (NaT). This removes the corrupted lines.
catalog = catalog.dropna(subset=['date'])

# [FIX] Convert to pure date objects for comparison
catalog["date"] = catalog["date"].dt.date

# [FIX] Clean expiration column too if it exists
if "expiration" in catalog.columns:
    catalog["expiration"] = pd.to_datetime(catalog["expiration"], errors='coerce')

today = date.today()

# [FIX] Safe filtering
available_dates = sorted(d for d in catalog["date"].unique() if pd.notna(d) and d <= today)

if not available_dates:
    raise RuntimeError("No valid dates found in catalog.")

TRADE_DATE = available_dates[-1]
print(f"Target Trade Date Resolved: {TRADE_DATE}")

# --- STALENESS CHECK ---
days_lag = (today - TRADE_DATE).days
if days_lag > 3:
    print(f"\n[CRITICAL ERROR] Data is stale by {days_lag} days!")
    print(f"  Latest available date: {TRADE_DATE}")
    print(f"  Current system date:   {today}")
    print("  ACTION REQUIRED: Check your data downloader or file appender.")
    sys.exit(1)

catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()
catalog_today["row_id"] = catalog_today.index

# 2. GENERATE CANDIDATES
print(f"\nGenerating candidates...")

# Initialize generator (With safety check for old vs new version)
try:
    # Attempt to use Smart Config
    gen_config = GeneratorConfig(
        min_dte=7,
        max_dte=45,
        max_candidates=MAX_CANDIDATES,
        min_vol=50,
        min_oi=100,
        # max_bid_ask_pct=0.05,  <-- These only work if you updated candidate_generator.py
        # target_delta=0.50
    )
    generator = CandidateGenerator(gen_config)
except TypeError:
    # Fallback if your CandidateGenerator is the old version
    print("Warning: Using Default/Old CandidateGenerator config.")
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
accepted_trades = []

# HEADERS
ledger_headers = [
    "run_id", "timestamp", "trade_date", "row_id", "model_version",
    "type", "strike", "dte", "expiration", "contractID",
    "S0", "IV0", "entry_price", "bid", "ask", "spread_pct", "volume", "open_interest",
    "mu_log_raw", "sigma_raw", "aiv_10d", "mu_arith_adj",
    "hold_days", "mc_mode", "year_days", "iv_daily_std", "iv_min", "iv_max", "slippage",
    "tp_pct", "sl_pct", "be_pct", "be_buffer",
    "expected_pnl", "prob_profit", "downside_sharpe", "mc_sharpe", "VaR95", "expected_opt_price",
    "expected_pnl_std", "prob_profit_std", "downside_sharpe_std", "VaR95_std",
    "p10_value", "p90_value", "avg_exit_day",
    "TP_rate", "SL_rate", "BE_rate", "Hold_rate",
    "decision", "reason", "seed_type", "seed_val"
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
        entry_price = market_state.get('price', 0.0)
        mid_price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else entry_price

        contract_id = get_occ_symbol(UNDERLYING_SYMBOL, expiration, cand_type, strike)

        spread_pct = 0.0
        if entry_price > 0:
            spread_pct = (ask - bid) / entry_price

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
            "bid": f"{bid:.2f}", "ask": f"{ask:.2f}",
            "spread_pct": f"{spread_pct:.4f}",
            "volume": vol, "open_interest": oi,
            "mu_log_raw": f"{raw_mu_log:.4f}",
            "sigma_raw": f"{raw_sigma:.4f}",
            "aiv_10d": f"{raw_aiv:.4f}",
            "mu_arith_adj": f"{pred_mu:.4f}",
            "hold_days": exit_rules.hold_days,
            "mc_mode": "PATH" if exit_rules.use_path_exits else "TERMINAL",
            "year_days": exit_rules.year_days,
            "iv_daily_std": exit_rules.iv_daily_std,
            "iv_min": exit_rules.iv_min, "iv_max": exit_rules.iv_max,
            "slippage": exit_rules.slippage_pct,
            "tp_pct": exit_rules.take_profit_pct, "sl_pct": exit_rules.stop_loss_pct,
            "be_pct": exit_rules.be_trigger_pct, "be_buffer": exit_rules.be_exit_buffer
        }

        # --- DIRECTIONAL PRE-FILTER ---
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
            skip_row = base_row.copy()
            skip_row.update({
                "decision": "SKIP",
                "reason": skip_reason,
                "seed_type": "N/A", "seed_val": "N/A"
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

        if entry_price > 0 and abs(agg['VaR_95_mean']) > (entry_price * 5):
            print(f"    [WARNING] Sanity Check: VaR ({agg['VaR_95_mean']:.2f}) > 5x Price")

        decision = decision_builder.build_decision(
            expected_pnl=agg['expected_pnl_mean'],
            prob_profit=agg['prob_profit_mean'],
            downside_sharpe=agg['downside_sharpe_mean'],
            cvar=agg['VaR_95_mean'],
            premium=entry_price,
            downside_sharpe_std=agg['downside_sharpe_std'],
            volatility_consistent=True,
            no_macro_events=True,
            system_healthy=True
        )

        status = "✅ ACCEPT" if decision.decision == DecisionType.TRADE else "❌ REJECT"
        pnl_share = agg['expected_pnl_mean']
        sharpe_mean = agg['downside_sharpe_mean']
        sharpe_std = agg['downside_sharpe_std']
        var95 = agg['VaR_95_mean']
        pop = agg['prob_profit_mean']

        print(f"  [Candidate {idx}] {status} | {contract_id} | {cand_type} {strike} DTE {dte} exp {expiration}")
        print(f"     bid/ask {bid:.2f}/{ask:.2f} mid {mid_price:.2f} | "
              f"EV {pnl_share:.2f} | PoP {pop:.2f} | "
              f"DS {sharpe_mean:.3f}±{sharpe_std:.3f} | VaR95 {var95:.1f}")

        if decision.decision != DecisionType.TRADE:
            print(f"     [REJECT REASON] {decision.reason}")
        else:
            # Composite Score Calculation
            composite_score = sharpe_mean - (0.5 * sharpe_std)
            if abs(var95) > 1e-6:
                composite_score += 0.2 * (pnl_share / abs(var95))

            accepted_trades.append({
                "id": contract_id,
                "desc": f"{cand_type} {strike} DTE {dte}",
                "score": composite_score,
                "ds": sharpe_mean,
                "ds_std": sharpe_std,
                "ev": pnl_share,
                "pop": pop,
                "var": var95
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
            "expected_pnl_std": f"{agg['expected_pnl_std']:.4f}",
            "prob_profit_std": f"{agg['prob_profit_std']:.4f}",
            "downside_sharpe_std": f"{agg['downside_sharpe_std']:.4f}",
            "VaR95_std": f"{agg['VaR_95_std']:.4f}",
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
    append_ledger_rows(LEDGER_FILE, all_ledger_rows, headers=ledger_headers)
    print(f"\n[LEDGER] Saved {len(all_ledger_rows)} rows to {LEDGER_FILE}")

# SUMMARY
if accepted_trades:
    print("\n" + "=" * 80)
    print("🏆 TOP ACCEPTED TRADES (Ranked by Composite Score)")
    print("   Formula: Score = DS_mean - 0.5*DS_std + 0.2*(EV/|VaR|)")
    print("=" * 80)
    accepted_trades.sort(key=lambda x: x['score'], reverse=True)

    for rank, t in enumerate(accepted_trades, start=1):
        print(f"#{rank} {t['id']} | {t['desc']}")
        print(f"   Score: {t['score']:.4f} | DS: {t['ds']:.3f}±{t['ds_std']:.3f} | "
              f"EV: ${t['ev']:.2f} | VaR: {t['var']:.1f}")
        print("-" * 80)
else:
    print("\n" + "=" * 80)
    print("⚠️  NO TRADES ACCEPTED TODAY")
    print("=" * 80)

print("\nDaily orchestration complete.")
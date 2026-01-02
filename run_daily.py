"""
run_daily.py
============

Daily orchestration script for the TSLA options trading system.
Integrates:
1. Candidate Generation (with Row ID Tracing)
2. Fusion Model Prediction (The Brain)
3. Monte Carlo Simulation (The Physics)
4. Trade Decision Logic (The Gatekeeper)
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import sys
from datetime import date
import pandas as pd
import numpy as np

# ============================================================
# 2. CONFIGURATION
# ============================================================
LOGS_DIR = r"C:\My Documents\Mics\Logs"
CATALOG_FILE = os.path.join(LOGS_DIR, "TSLA_Options_Contracts.csv")

RISK_FREE_RATE = 0.04
NUM_MC_PATHS = 50_000
MAX_CANDIDATES = 20

# [NEW] FAIRNESS CHECK MODE
# Set to True to verify the engine produces ~0 P&L with neutral inputs.
# Set to False for real trading mode.
FAIRNESS_CHECK_MODE = False

# ============================================================
# 3. PROJECT PATH SETUP
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add sub-folders to sys.path so Python can find the modules
sys.path.append(os.path.join(PROJECT_ROOT, "CandidateGenerator"))
sys.path.append(os.path.join(PROJECT_ROOT, "MarketStateAdapter"))
sys.path.append(os.path.join(PROJECT_ROOT, "Montecarlo_Sharpe"))
sys.path.append(os.path.join(PROJECT_ROOT, "TradeDecisionBuilder"))
sys.path.append(os.path.join(PROJECT_ROOT, "MLPTrainer"))

# ============================================================
# 4. MODULE IMPORTS
# ============================================================
from candidate_generator import CandidateGenerator, GeneratorConfig
from market_state_adapter import MarketStateAdapter
# [FIX] Import ExitRules to configure path simulation
from MonteCarloEngine import MonteCarloEngine, ExitRules
from TradeDecisionBuilder import TradeDecisionBuilder, DecisionType
from FusionPredictor import FusionPredictor

# ============================================================
# 5. SETUP & LOAD DATA
# ============================================================
print(f"\nLoading catalog from: {CATALOG_FILE}")
if not os.path.exists(CATALOG_FILE):
    print(f"ERROR: Catalog file not found at {CATALOG_FILE}")
    sys.exit(1)

catalog = pd.read_csv(CATALOG_FILE, parse_dates=["date", "expiration"])
catalog["date"] = catalog["date"].dt.date

today = date.today()
available_dates = sorted(d for d in catalog["date"].unique() if d <= today)
if not available_dates:
    raise RuntimeError("No valid catalog dates found.")
TRADE_DATE = available_dates[-1]

# [TRACEABILITY] Preserve Original Index as 'row_id' before filtering
# This allows us to map any candidate back to the exact line in the CSV.
catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()
catalog_today["row_id"] = catalog_today.index

print(f"Target Trade Date Resolved: {TRADE_DATE}")

# ============================================================
# 6. GENERATE CANDIDATES
# ============================================================
print(f"\nGenerating candidates...")
gen_config = GeneratorConfig(min_dte=7, max_dte=45, max_candidates=MAX_CANDIDATES)
generator = CandidateGenerator(gen_config)
candidates = generator.generate(catalog_today, trade_date=str(TRADE_DATE))
print(f"Generated {len(candidates)} candidates.")

if not candidates:
    print("No candidates generated. Exiting.")
    sys.exit(0)

# ============================================================
# 7. RUN PREDICTION (THE BRAIN)
# ============================================================
print("\nRunning Fusion Model Inference...")
try:
    predictor = FusionPredictor(base_dir=LOGS_DIR)
    fusion_output = predictor.predict(str(TRADE_DATE))

    print(f"Prediction for {TRADE_DATE} (Raw Model Output):")
    print(f"  Mu (Log Drift):     {fusion_output['mu']:.4f}")
    print(f"  Sigma (Annualized): {fusion_output['sigma']:.4f}")
    print(f"  AIV (10-day chg):   {fusion_output['aiv']:.4f}")

    # [CRITICAL FIX] DRIFT CONVERSION
    # The model predicts Log Drift (target = ln(Sf/Si)).
    # The MC Engine expects Arithmetic Drift because it subtracts 0.5*sigma^2 internally.
    # We must add back the variance term to avoid double-counting the drag.
    sigma = fusion_output['sigma']
    mu_log = fusion_output['mu']

    mu_arithmetic = mu_log + 0.5 * (sigma ** 2)
    fusion_output['mu'] = mu_arithmetic  # Update for MC engine

    print(f"  -> Adj. Mu (Arith): {mu_arithmetic:.4f} (Input to MC)")

    # [NEW] FAIRNESS CHECK OVERRIDE
    if FAIRNESS_CHECK_MODE:
        print("\n⚠️  WARNING: FAIRNESS CHECK MODE ENABLED ⚠️")
        print("Overriding model prediction with neutral inputs (Mu=0, Sigma=0.40, AIV=0)")
        fusion_output['mu'] = 0.0
        fusion_output['aiv'] = 0.0
        fusion_output['sigma'] = 0.40  # Representative Vol

except Exception as e:
    print(f"CRITICAL PREDICTION FAILURE: {e}")
    sys.exit(1)

# ============================================================
# 8. RUN SIMULATION & DECISION LOGIC
# ============================================================
print(f"\nStarting Evaluation ({NUM_MC_PATHS} paths)...")

# [CRITICAL UPDATE] WIDENED STOPS FOR HIGH VOLATILITY REGIME
# We are betting on a massive directional move (-0.68 drift).
# We must allow the trade room to breathe.
exit_rules = ExitRules(
    use_path_exits=True,
    hold_days=14,
    take_profit_pct=0.80,    # Raised to +80% (Let winners run)
    stop_loss_pct=0.50,      # Widened to -50% (Survive the 48% vol noise)
    be_trigger_pct=0.25,     # Delay BE trigger slightly to 25%
    be_exit_buffer=0.01
)

mc_engine = MonteCarloEngine(num_paths=NUM_MC_PATHS, exit_config=exit_rules)
decision_builder = TradeDecisionBuilder()
adapter = MarketStateAdapter(risk_free_rate=RISK_FREE_RATE)

# Extract prediction signal for pre-filtering (Use the Adjusted Mu)
pred_mu = fusion_output['mu']

for idx, candidate in enumerate(candidates, start=1):
    try:
        # A. Adapt
        if hasattr(adapter, 'adapt'):
            market_state = adapter.adapt(candidate)
        else:
            market_state = adapter.build_market_state(candidate)
        market_state['r'] = RISK_FREE_RATE

        # Get Candidate Details for Filtering/Logging
        leg = candidate['legs'][0]
        cand_type = leg['type'].upper()  # CALL/PUT
        strike = leg['strike']
        dte = leg['dte']
        rid = leg.get('row_id', -1)  # Trace ID

        # --- DIRECTIONAL PRE-FILTER ---
        # Don't simulate trades that fight the model's primary trend
        if not FAIRNESS_CHECK_MODE:
            # Note: We check against the adjusted arithmetic Mu, which represents the drift center.
            if pred_mu < 0 and cand_type == 'CALL':
                print(f"  [Candidate {idx}] SKIPPED ({cand_type} vs Bearish Signal)")
                continue # This continue ensures we skip execution
            if pred_mu > 0 and cand_type == 'PUT':
                print(f"  [Candidate {idx}] SKIPPED ({cand_type} vs Bullish Signal)")
                continue # This continue ensures we skip execution

        # --- DEBUG: TRACE TO CSV ---
        if rid != -1:
            try:
                full_row = catalog.loc[rid]
                trace_dict = full_row[
                    ["expiration", "dte", "type", "strike", "implied_volatility", "opt_price"]].to_dict()
                if isinstance(trace_dict["expiration"], pd.Timestamp):
                    trace_dict["expiration"] = trace_dict["expiration"].strftime("%Y-%m-%d")
            except Exception:
                pass

        # B. Simulate
        metrics = mc_engine.generate_risk_metrics(fusion_output, market_state)

        # C. Decide
        entry_price = market_state.get('price', 0.0)

        # [FIX] UNIT CONSISTENCY: Pass premium in Per-Share units
        decision = decision_builder.build_decision(
            expected_pnl=metrics['expected_pnl'],
            prob_profit=metrics['prob_profit'],
            downside_sharpe=metrics['downside_sharpe'],
            cvar=metrics['VaR_95'],
            premium=entry_price,   # Per-Share units
            volatility_consistent=True,
            no_macro_events=True,
            system_healthy=True
        )

        # D. Output (Enhanced)
        status = "✅ ACCEPT" if decision.decision == DecisionType.TRADE else "❌ REJECT"
        desc_str = f"{cand_type} {strike} (DTE {dte})"

        # Calculate Per-Contract Metrics (x100)
        pnl_share = metrics['expected_pnl']
        pnl_contract = pnl_share * 100

        var_share = metrics['VaR_95']
        var_contract = var_share * 100

        print(f"  [Candidate {idx}] {status} | {desc_str}")
        print(f"     P&L: ${pnl_share:6.2f}/sh  (${pnl_contract:7.0f}/ct)  |  "
              f"VaR95: ${var_share:6.2f}/sh (${var_contract:7.0f}/ct)")
        print(f"     Prob Profit: {metrics['prob_profit']:.1%} | "
              f"Downside Sharpe: {metrics['downside_sharpe']:.3f}") # Shows 3 decimals

        # [NEW] REJECTION DEBUGGING
        if decision.decision != DecisionType.TRADE:
            print(f"     [REJECT REASON] {decision.reason}")
            # print(f"     [GATES] {decision.gates_log}") # Optional: Uncomment for full verbosity

        # [NEW] DIAGNOSTICS LOGGING
        # Only prints if the engine successfully returned these stats (Path Mode)
        if "exit_rate_tp" in metrics:
            print(f"     [DIAGNOSTICS] TP: {metrics['exit_rate_tp']:.1%} | "
                  f"SL: {metrics['exit_rate_sl']:.1%} | "
                  f"BE: {metrics['exit_rate_be']:.1%} | "
                  f"Hold: {metrics['hold_rate']:.1%} | "
                  f"AvgDay: {metrics.get('exit_day_mean', 0):.1f}")

    except Exception as e:
        print(f"  [Candidate {idx}] Failed: {e}")

print("\nDaily orchestration complete.")
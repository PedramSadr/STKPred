"""
run_daily.py
============
Daily orchestration script for the TSLA options trading system.
Integrates:
1. Candidate Generation
2. Fusion Model Prediction (Brain)
3. Monte Carlo Simulation (Physics)
4. Trade Decision Logic (Gatekeeper)
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import sys
from datetime import date
import pandas as pd

# ============================================================
# 2. CONFIGURATION
# ============================================================
LOGS_DIR = r"C:\My Documents\Mics\Logs"
CATALOG_FILE = os.path.join(LOGS_DIR, "TSLA_Options_Contracts.csv")

RISK_FREE_RATE = 0.04
NUM_MC_PATHS = 50_000
MAX_CANDIDATES = 20

# ============================================================
# 3. PROJECT PATH SETUP
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(PROJECT_ROOT, "CandidateGenerator"))
sys.path.append(os.path.join(PROJECT_ROOT, "MarketStateAdapter"))
sys.path.append(os.path.join(PROJECT_ROOT, "Montecarlo_Sharpe"))
sys.path.append(os.path.join(PROJECT_ROOT, "TradeDecisionBuilder"))
sys.path.append(os.path.join(PROJECT_ROOT, "MLPTrainer"))  # <--- For FusionPredictor

# ============================================================
# 4. MODULE IMPORTS
# ============================================================
from candidate_generator import CandidateGenerator, GeneratorConfig
from market_state_adapter import MarketStateAdapter
from MonteCarloEngine import MonteCarloEngine
from TradeDecisionBuilder import TradeDecisionBuilder, DecisionType
from FusionPredictor import FusionPredictor  # <--- NEW

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
if not available_dates: raise RuntimeError("No valid catalog dates found.")
TRADE_DATE = available_dates[-1]
catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()

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
    # Initialize Predictor (Loads models & fits scalers)
    predictor = FusionPredictor(base_dir=LOGS_DIR)

    # Get Prediction
    fusion_output = predictor.predict(str(TRADE_DATE))

    print(f"Prediction for {TRADE_DATE}:")
    print(f"  Mu (Annualized):    {fusion_output['mu']:.4f}")
    print(f"  Sigma (Annualized): {fusion_output['sigma']:.4f}")
    print(f"  AIV (10-day chg):   {fusion_output['aiv']:.4f}")

except Exception as e:
    print(f"CRITICAL PREDICTION FAILURE: {e}")
    sys.exit(1)

# ============================================================
# 8. RUN SIMULATION & DECISION LOGIC
# ============================================================
print(f"\nStarting Evaluation ({NUM_MC_PATHS} paths)...")

mc_engine = MonteCarloEngine(num_paths=NUM_MC_PATHS)
decision_builder = TradeDecisionBuilder()
adapter = MarketStateAdapter(risk_free_rate=RISK_FREE_RATE)

for idx, candidate in enumerate(candidates, start=1):
    try:
        # A. Adapt
        if hasattr(adapter, 'adapt'):
            market_state = adapter.adapt(candidate)
        else:
            market_state = adapter.build_market_state(candidate)
        market_state['r'] = RISK_FREE_RATE

        # B. Simulate
        metrics = mc_engine.generate_risk_metrics(fusion_output, market_state)

        # C. Decide
        entry_price = market_state.get('price', 0.0)
        decision = decision_builder.build_decision(
            expected_pnl=metrics['expected_pnl'],
            prob_profit=metrics['prob_profit'],
            downside_sharpe=metrics['downside_sharpe'],
            cvar=metrics['VaR_95'],
            premium=entry_price * 100.0,
            volatility_consistent=True,
            no_macro_events=True,
            system_healthy=True
        )

        # D. Output
        status = "✅ ACCEPT" if decision.decision == DecisionType.TRADE else "❌ REJECT"
        print(f"  [Candidate {idx}] {status}")
        print(
            f"     Metrics: P&L ${metrics['expected_pnl']:.2f} | PoP {metrics['prob_profit']:.2f} | D-Sharpe {metrics['downside_sharpe']:.2f}")

    except Exception as e:
        print(f"  [Candidate {idx}] Failed: {e}")

print("\nDaily orchestration complete.")
"""
run_daily.py
============

Daily orchestration script for the TSLA options trading system.
This script imports the CandidateGenerator from the external module
'candidate_generator.py'.
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

# Add sub-folders to sys.path so Python can find the modules
sys.path.append(os.path.join(PROJECT_ROOT, "CandidateGenerator"))
sys.path.append(os.path.join(PROJECT_ROOT, "MarketStateAdapter"))
sys.path.append(os.path.join(PROJECT_ROOT, "Montecarlo_Sharpe"))

# ============================================================
# 4. MODULE IMPORTS
# ============================================================
from candidate_generator import CandidateGenerator, GeneratorConfig
from market_state_adapter import MarketStateAdapter
from MonteCarloEngine import MonteCarloEngine

# ============================================================
# 5. LOAD CATALOG
# ============================================================
print(f"\nLoading catalog from: {CATALOG_FILE}")

if not os.path.exists(CATALOG_FILE):
    print(f"ERROR: Catalog file not found at {CATALOG_FILE}")
    sys.exit(1)

catalog = pd.read_csv(
    CATALOG_FILE,
    parse_dates=["date", "expiration"]
)
catalog["date"] = catalog["date"].dt.date

# ============================================================
# 6. RESOLVE TRADE DATE
# ============================================================
today = date.today()
available_dates = sorted(d for d in catalog["date"].unique() if d <= today)

if not available_dates:
    raise RuntimeError("No valid catalog dates found.")

TRADE_DATE = available_dates[-1]
catalog_today = catalog[catalog["date"] == TRADE_DATE].copy()

print(f"Target Trade Date Resolved: {TRADE_DATE}")
print(f"Loaded {len(catalog_today)} records for {TRADE_DATE}")

# ============================================================
# 7. GENERATE CANDIDATES
# ============================================================
print(f"\nGenerating candidates (Limit: {MAX_CANDIDATES})...")

# Configure the external Generator
generator_config = GeneratorConfig(
    min_dte=7,
    max_dte=45,
    min_vol=1,
    min_oi=1,
    max_candidates=MAX_CANDIDATES
)

# Instantiate the imported class
generator = CandidateGenerator(generator_config)
candidates = generator.generate(catalog_today, trade_date=str(TRADE_DATE))

print(f"Generated {len(candidates)} candidates.")

if not candidates:
    print("No candidates generated. Exiting.")
    sys.exit(0)

# ============================================================
# 8. FUSION OUTPUT (PLACEHOLDER)
# ============================================================
fusion_output = {
    "mu": 0.02,
    "sigma": 0.35,
    "aiv": -0.01
}

# ============================================================
# 9. RUN MONTE CARLO
# ============================================================
print(f"\nStarting Monte Carlo Evaluation ({NUM_MC_PATHS} paths)...")

mc_engine = MonteCarloEngine(num_paths=NUM_MC_PATHS)

# FIX: Initialize Adapter ONCE with the required argument
# We assume the class signature is MarketStateAdapter(risk_free_rate=...)
try:
    adapter = MarketStateAdapter(risk_free_rate=RISK_FREE_RATE)
except TypeError:
    # Fallback if the adapter doesn't accept init arguments
    adapter = MarketStateAdapter()

for idx, candidate in enumerate(candidates, start=1):
    try:
        # 1. Adapt Candidate
        # We try the standard methods. If your adapter uses 'build_market_state', we call that.
        if hasattr(adapter, 'adapt'):
            market_state = adapter.adapt(candidate)
        elif hasattr(adapter, 'build_market_state'):
            market_state = adapter.build_market_state(candidate)
        else:
            raise AttributeError("Adapter has no 'adapt' or 'build_market_state' method.")

        # 2. Inject Risk Free Rate (Redundant safety, but keeps Engine happy)
        market_state['r'] = RISK_FREE_RATE

        # 3. Run Simulation
        metrics = mc_engine.generate_risk_metrics(
            fusion_output=fusion_output,
            market_state=market_state
        )

        print(
            f"  [Candidate {idx}] "
            f"P&L: ${metrics['expected_pnl']:.2f} | "
            f"Prob: {metrics['prob_profit']:.3f} | "
            f"Downside Sharpe: {metrics['downside_sharpe']:.3f}"
        )

    except Exception as e:
        print(f"  [Candidate {idx}] Failed: {e}")

# ============================================================
# 10. DONE
# ============================================================
print("\nDaily orchestration complete.")
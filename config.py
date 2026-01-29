import os


class Config:
    # --- 1. GLOBAL SETTINGS ---
    UNDERLYING_SYMBOL = "TSLA"
    APP_ENV = "DEV"
    DEBUG_MODE = True

    # --- 2. PATHS ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # INPUT Data (Catalog, Daily CSVs) - Source of Truth
    DATA_DIR = r"C:\My Documents\Mics\Logs"

    # OUTPUT Logs & Ledger (The Fix)
    # This sends logs and the ledger to your specific "Dev_Logs" folder
    LOGS_DIR = r"C:\My Documents\Mics\Dev_Logs"

    # Ledger Filename (Updated per your request)
    DB_FILE = os.path.join(LOGS_DIR, "paper_trade_ledger.csv")

    # --- 3. ACCOUNT SETTINGS ---
    ACCOUNT_EQUITY = 15000.0  # Base Equity for Risk Calc

    # --- 4. RISK MANAGEMENT (The "2x$200" Rule) ---
    # Layer 1: Per-Trade Limit (1.35% of $15k = ~$202.50)
    RISK_PER_TRADE_PCT = 0.0135

    # Layer 2: Portfolio Limit (2.8% max allows two full trades)
    PORTFOLIO_MAX_LOSS_PCT = 0.028

    # Layer 3: Tail Risk (CVaR)
    PORTFOLIO_CVAR_PCT = 0.020

    # --- 5. CONSTRAINTS ---
    MAX_POS_TOTAL = 2  # Hard Cap: 2 Trades
    MAX_POS_PER_SYMBOL = 2  # Allow both to be TSLA

    # Exposure Caps
    MAX_RISK_PER_FACTOR_PCT = 0.028
    MAX_RISK_PER_EXPIRY_PCT = 0.015
    MAX_RISK_UNMAPPED_PCT = 0.010

    # --- 6. STRATEGY GATES ---
    ENABLE_SPREADS = True
    ENABLE_SINGLE_LEGS = False

    # --- 7. EXIT STRATEGY SETTINGS ---
    EXIT_RULE_DTE = 21
    EXIT_TAKE_PROFIT_PCT = 0.50
    ENABLE_TAKE_PROFIT = False

    # --- 8. SYSTEM SETTINGS ---
    # Automatically creates the Dev_Logs folder if it doesn't exist
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
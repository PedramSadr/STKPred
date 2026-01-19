import os


class Config:
    # --- 1. GLOBAL SETTINGS ---
    # We keep TSLA for now to operationalize the "Time-21" exit rule.
    # Once validated, you can switch this to "NVDA".
    UNDERLYING_SYMBOL = "TSLA"
    APP_ENV = "DEV"
    DEBUG_MODE = True

    # --- 2. PATHS ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Shared Data Directory (Source of Truth for CSVs)
    DATA_DIR = r"C:\My Documents\Mics\Logs"

    # Local Logs & Ledger
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    DB_FILE = os.path.join(LOGS_DIR, "dev_ledger.csv")

    # --- 3. STRATEGY GATES ---
    # Master switch for vertical spreads (Priority 0)
    ENABLE_SPREADS = True

    # Disable single legs to focus computation on spreads (Priority 2)
    ENABLE_SINGLE_LEGS = False

    # --- 4. EXIT STRATEGY SETTINGS ---
    # [cite_start]The "TIME_21" Rule: Exit when DTE <= 21 (Winner of Backtest) [cite: 133, 145]
    EXIT_RULE_DTE = 21

    # Optional: Take Profit settings (Disabled by default per backtest results)
    EXIT_TAKE_PROFIT_PCT = 0.50
    ENABLE_TAKE_PROFIT = False

    # --- 5. SYSTEM SETTINGS ---
    # Ensures logs folder exists
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
import os
import sys
import pandas as pd
import logging
from datetime import datetime

# --- 1. CONFIGURATION IMPORT ---
try:
    from config import Config
except ImportError:
    print("CRITICAL ERROR: 'config.py' not found. Please ensure it exists in the project root.")
    sys.exit(1)

# --- 2. MODULE IMPORTS ---
try:
    # 1. Candidate Generator
    from CandidateGenerator.candidate_generator import CandidateGenerator

    # 2. Trade Decision Builder (Robust Import)
    try:
        from TradeDecisionBuilder.trade_decision_builder import TradeDecisionBuilder
    except ImportError:
        from TradeDecisionBuilder.TradeDecisionBuilder import TradeDecisionBuilder

    # 3. Monte Carlo Engine
    from Montecarlo_Sharpe.monte_carlo_engine import MonteCarloEngine

except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


# --- 3. LOGGING SETUP ---
def setup_logging():
    log_filename = os.path.join(Config.LOGS_DIR, "daily_pipeline.log")

    logging.basicConfig(
        level=logging.DEBUG if Config.DEBUG_MODE else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"--- STARTING DAILY RUN (ENV: {Config.APP_ENV}) ---")
    logging.info(f"Target Ticker: {Config.UNDERLYING_SYMBOL}")


def main():
    setup_logging()

    # --- 4. PATH SETUP (DYNAMIC) ---
    symbol = Config.UNDERLYING_SYMBOL

    # Input Files
    CATALOG_FILE = os.path.join(Config.DATA_DIR, f"{symbol}_Options_Contracts.csv")
    DAILY_FILE = os.path.join(Config.DATA_DIR, f"{symbol.lower()}_daily.csv")

    # Output File
    LEDGER_FILE = Config.DB_FILE

    # Validate Inputs
    if not os.path.exists(CATALOG_FILE):
        logging.error(f"Catalog not found: {CATALOG_FILE}")
        sys.exit(1)
    if not os.path.exists(DAILY_FILE):
        logging.error(f"Daily data not found: {DAILY_FILE}")
        sys.exit(1)

    # --- 5. LOAD MARKET CONTEXT ---
    logging.info("Loading market data...")
    try:
        df_daily = pd.read_csv(DAILY_FILE)

        # Robust Column Normalization
        df_daily.columns = [c.lower().strip() for c in df_daily.columns]
        col_map = {c: 'close' for c in df_daily.columns if 'close' in c}
        col_map.update({c: 'date' for c in df_daily.columns if 'date' in c or 'timestamp' in c})
        if col_map: df_daily.rename(columns=col_map, inplace=True)

        if 'close' not in df_daily.columns:
            raise KeyError("Column 'close' missing from daily data.")

        current_price = df_daily['close'].iloc[-1]

        # Get Trade Date (Critical for Backtesting)
        trade_date = df_daily['date'].iloc[-1] if 'date' in df_daily.columns else datetime.now().strftime('%Y-%m-%d')

        logging.info(f"Market Context: Date={trade_date}, S0=${current_price:.2f}")

    except Exception as e:
        logging.error(f"CRITICAL DATA FAILURE: {e}")
        sys.exit(1)

    # --- 6. GENERATE CANDIDATES ---
    logging.info("Initializing Candidate Generator...")

    # Pass trade_date for correct DTE calculation
    generator = CandidateGenerator(
        catalog_path=CATALOG_FILE,
        current_price=current_price,
        trade_date=trade_date
    )

    candidates = generator.generate_candidates()
    logging.info(f"Generated {len(candidates)} candidates.")

    # --- 7. INITIALIZE ENGINES ---
    mc_engine = MonteCarloEngine()
    decision_builder = TradeDecisionBuilder()

    ledger_entries = []

    # --- 8. MAIN ANALYSIS LOOP ---
    for i, candidate in enumerate(candidates):
        try:
            # --- A. EXTRACT PRE-CALCULATED ECONOMICS ---
            contract_id = candidate.get('contractID', 'UNKNOWN')
            structure_type = candidate.get('strategy', 'UNKNOWN')

            economics = candidate.get('economics', {})
            entry_cost = economics.get('entry_cost', 0.0)
            max_loss = economics.get('max_loss', 0.0)  # <--- EXTRACTED HERE

            # Display Metadata
            legs = candidate.get('legs', [])
            if len(legs) == 2:
                strike_display = f"{legs[0]['strike']}/{legs[1]['strike']}"
                expiration = legs[0]['expiration']
            elif len(legs) == 1:
                strike_display = str(legs[0]['strike'])
                expiration = legs[0]['expiration']
            else:
                logging.warning(f"Skipping {contract_id}: Odd leg count ({len(legs)})")
                continue

            logging.info(f"Analyzing {contract_id} (Cost: ${entry_cost:.2f})...")

            # --- B. RUN MONTE CARLO SIMULATION ---
            # Pass trade_date for correct DTE
            mc_result = mc_engine.analyze(
                candidate,
                current_price,
                entry_cost,
                trade_date=trade_date
            )

            # --- C. BUILD TRADE DECISION ---
            # FIX: Pass max_loss explicitely for "Binary Gamble" check
            decision_result = decision_builder.evaluate(mc_result, max_loss=max_loss)

            # --- D. PREPARE LEDGER ENTRY ---
            entry = {
                'timestamp': datetime.now().isoformat(),
                'trade_date': trade_date,
                'contractID': contract_id,
                'type': structure_type,
                'strike': strike_display,
                'expiration': expiration,
                'entry_price': entry_cost,
                'max_loss': max_loss,
                'decision': decision_result.get('decision', 'ERROR'),
                'reason': decision_result.get('reason', 'Unknown'),

                # MC Metrics
                'expected_pnl': round(mc_result.get('expected_pnl', 0), 2),
                'prob_profit': round(mc_result.get('prob_profit', 0), 4),
                'downside_sharpe': round(mc_result.get('downside_sharpe', 0), 4),
                'VaR95': round(mc_result.get('VaR95', 0), 2),
                'CVaR95': round(mc_result.get('CVaR95', 0), 2)
            }

            ledger_entries.append(entry)
            logging.info(f"-> Decision: {entry['decision']} | Reason: {entry['reason']}")

        except Exception as e:
            logging.error(f"Error processing candidate {i}: {e}", exc_info=True)
            continue

    # --- 9. SAVE TO LEDGER ---
    if ledger_entries:
        df_new = pd.DataFrame(ledger_entries)

        if os.path.exists(LEDGER_FILE):
            try:
                df_old = pd.read_csv(LEDGER_FILE)
                # Align columns
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except pd.errors.EmptyDataError:
                df_final = df_new
        else:
            df_final = df_new

        df_final.to_csv(LEDGER_FILE, index=False)
        logging.info(f"--- [STEP SUCCESS] 5. Daily Strategy Execution ---")
        logging.info(f"[LEDGER] Saved {len(df_new)} rows to {LEDGER_FILE}")
    else:
        logging.warning("No candidates processed.")


if __name__ == "__main__":
    main()
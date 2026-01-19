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

    # 2. Trade Decision Builder
    try:
        from TradeDecisionBuilder.trade_decision_builder import TradeDecisionBuilder
    except ImportError:
        from TradeDecisionBuilder.TradeDecisionBuilder import TradeDecisionBuilder

    # 3. Monte Carlo Engine
    from Montecarlo_Sharpe.MonteCarloEngine import MonteCarloEngine

    # 4. Trade Manager (Lifecycle & Persistence)
    try:
        from TradeManager.trade_manager import TradeManager
    except ImportError:
        logging.warning("TradeManager not found. Exit logic will be skipped.")
        TradeManager = None

    # [NEW] 5. Portfolio Risk Manager (Phase 3 Allocation)
    try:
        from TradeDecisionBuilder.portfolio_risk_manager import PortfolioRiskManager
    except ImportError:
        logging.warning("PortfolioRiskManager not found. Sizing will be default (1 lot).")
        PortfolioRiskManager = None

except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


# --- 3. LOGGING SETUP ---
def setup_logging():
    if hasattr(Config, 'LOGS_DIR'):
        os.makedirs(Config.LOGS_DIR, exist_ok=True)

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
    CATALOG_FILE = os.path.join(Config.DATA_DIR, f"{symbol}_Options_Contracts.csv")
    DAILY_FILE = os.path.join(Config.DATA_DIR, f"{symbol.lower()}_daily.csv")
    LEDGER_FILE = Config.DB_FILE

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
        df_daily.columns = [c.lower().strip() for c in df_daily.columns]
        col_map = {c: 'close' for c in df_daily.columns if 'close' in c}
        col_map.update({c: 'date' for c in df_daily.columns if 'date' in c or 'timestamp' in c})
        if col_map: df_daily.rename(columns=col_map, inplace=True)

        current_price = df_daily['close'].iloc[-1]
        trade_date = df_daily['date'].iloc[-1] if 'date' in df_daily.columns else datetime.now().strftime('%Y-%m-%d')

        logging.info(f"Market Context: Date={trade_date}, S0=${current_price:.2f}")

    except Exception as e:
        logging.error(f"CRITICAL DATA FAILURE: {e}")
        sys.exit(1)

    # --- 6. INITIALIZE MANAGERS ---
    trade_manager = None
    risk_manager = None

    if TradeManager:
        trade_manager = TradeManager()

        # Initialize Risk Manager with path to open_positions.csv
        if PortfolioRiskManager:
            risk_manager = PortfolioRiskManager(Config, trade_manager.positions_file)
            logging.info("Portfolio Risk Manager: ACTIVE")

        # --- CHECK EXITS ---
        logging.info("--- CHECKING OPEN POSITIONS ---")
        open_positions = trade_manager.load_open_positions()
        logging.info(f"Found {len(open_positions)} open positions.")

        for pos in open_positions:
            exit_decision = trade_manager.check_exit(pos, trade_date)

            if exit_decision['action'] == 'ERROR':
                logging.error(f"Error checking exit for {pos.get('contractID')}: {exit_decision['reason']}")
                continue

            if exit_decision['action'] == 'EXIT':
                logging.info(f"[EXIT EXECUTION] {pos.get('contractID')} -> {exit_decision['reason']}")
                pid = pos.get('position_id')
                if not pid or str(pid).lower() == 'nan': pid = pos.get('contractID')

                trade_manager.close_position(
                    position_id=pid,
                    exit_reason=exit_decision['reason'],
                    exit_date=trade_date
                )
            else:
                logging.info(f"[HOLD] {pos.get('contractID')} (DTE: {exit_decision.get('reason')})")

    # --- 7. GENERATE CANDIDATES ---
    logging.info("--- GENERATING NEW ENTRIES ---")
    generator = CandidateGenerator(
        catalog_path=CATALOG_FILE,
        current_price=current_price,
        trade_date=trade_date
    )
    candidates = generator.generate_candidates()
    logging.info(f"Generated {len(candidates)} candidates.")

    # --- 8. INITIALIZE ENGINES ---
    mc_engine = MonteCarloEngine()
    decision_builder = TradeDecisionBuilder()
    ledger_entries = []

    # --- 9. MAIN ANALYSIS LOOP ---
    for i, candidate in enumerate(candidates):
        try:
            contract_id = candidate.get('contractID', 'UNKNOWN')
            structure_type = candidate.get('strategy', 'UNKNOWN')
            economics = candidate.get('economics', {})
            entry_cost = economics.get('entry_cost', 0.0)
            max_loss = economics.get('max_loss', 0.0)

            # Metadata
            legs = candidate.get('legs', [])
            expiration = legs[0]['expiration'] if legs else "N/A"
            strike_display = f"{legs[0]['strike']}/{legs[1]['strike']}" if len(legs) == 2 else "N/A"

            logging.info(f"Analyzing {contract_id} (Cost: ${entry_cost:.2f})...")

            # A. Monte Carlo Simulation
            mc_result = mc_engine.analyze(candidate, current_price, entry_cost, trade_date=trade_date)

            # Inject CVaR back into candidate for Risk Manager
            candidate['CVaR95'] = mc_result.get('CVaR95', 0.0)

            # B. Trade Decision (Edge Check)
            decision_result = decision_builder.evaluate(mc_result, max_loss=max_loss)

            decision = decision_result.get('decision', 'ERROR')
            reason = decision_result.get('reason', 'Unknown')

            qty = 0
            risk_details = {}

            # C. Allocation (Size Check) - ONLY if Edge is Good
            if decision == 'TRADE' and risk_manager:
                # Pass Account Equity (Static $10k for paper trading, or dynamic from config)
                equity = getattr(Config, 'ACCOUNT_EQUITY', 10000.0)
                qty, risk_details = risk_manager.allocate(candidate, equity=equity)

                if qty == 0:
                    decision = 'SKIP'
                    reason = f"Risk Blocked: {risk_details.get('blocked_by')} ({risk_details.get('reason', '')})"
            elif decision == 'TRADE' and not risk_manager:
                qty = 1  # Fallback if Risk Manager missing

            # D. Ledger Entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'trade_date': trade_date,
                'contractID': contract_id,
                'type': structure_type,
                'entry_price': entry_cost,
                'max_loss': max_loss,
                'decision': decision,
                'reason': reason,
                'qty_allocated': qty,
                'expected_pnl': round(mc_result.get('expected_pnl', 0), 2),
                'CVaR95': round(mc_result.get('CVaR95', 0), 2)
            }
            ledger_entries.append(entry)
            logging.info(f"-> Decision: {decision} | Qty: {qty} | Reason: {reason}")

            # E. Execution (Persistence)
            if decision == 'TRADE' and qty > 0 and trade_manager:
                logging.info(f"*** EXECUTION: Buying {qty}x {contract_id} ***")

                # [CRITICAL] Prepare Record with EXACT UNITS for Risk Manager
                # 1. Calculate Per-Contract Risk (x100)
                max_loss_per_contract = max_loss * 100
                cvar_per_contract = abs(candidate.get('CVaR95', 0.0)) * 100

                position_record = {
                    'entry_date': trade_date,
                    'contractID': contract_id,
                    'symbol': symbol,
                    'strategy': structure_type,
                    'expiration': expiration,
                    'qty': qty,

                    # Store Metrics for Risk Manager Layer 2 & 3
                    'max_loss_per_contract': round(max_loss_per_contract, 2),
                    'cvar_per_contract': round(cvar_per_contract, 2),
                    'entry_cost': entry_cost,

                    'status': 'OPEN'
                }

                trade_manager.save_new_position(position_record)

        except Exception as e:
            logging.error(f"Error processing candidate {i}: {e}", exc_info=True)
            continue

    # --- 10. SAVE LEDGER ---
    if ledger_entries:
        df_new = pd.DataFrame(ledger_entries)
        if os.path.exists(LEDGER_FILE):
            try:
                df_old = pd.read_csv(LEDGER_FILE)
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except pd.errors.EmptyDataError:
                df_final = df_new
        else:
            df_final = df_new

        df_final.to_csv(LEDGER_FILE, index=False)
        logging.info(f"[LEDGER] Saved {len(df_new)} rows to {LEDGER_FILE}")


if __name__ == "__main__":
    main()
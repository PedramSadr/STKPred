import os
import sys
import pandas as pd
import logging
import json
from datetime import datetime, timedelta

# --- 1. CONFIGURATION IMPORT ---
try:
    from config import Config
except ImportError:
    print("CRITICAL ERROR: 'config.py' not found. Please ensure it exists in the project root.")
    sys.exit(1)

# --- 2. MODULE IMPORTS ---
try:
    from CandidateGenerator.candidate_generator import CandidateGenerator

    try:
        from TradeDecisionBuilder.trade_decision_builder import TradeDecisionBuilder
    except ImportError:
        from TradeDecisionBuilder.TradeDecisionBuilder import TradeDecisionBuilder
    from Montecarlo_Sharpe.MonteCarloEngine import MonteCarloEngine

    try:
        from TradeManager.trade_manager import TradeManager
    except ImportError:
        logging.warning("TradeManager not found. Exit logic will be skipped.")
        TradeManager = None
    try:
        from TradeDecisionBuilder.portfolio_risk_manager import PortfolioRiskManager
    except ImportError:
        logging.error("PortfolioRiskManager not found. Trading will be BLOCKED (Fail Closed).")
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
        handlers=[logging.FileHandler(log_filename, mode='a'), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"--- STARTING DAILY RUN (ENV: {Config.APP_ENV}) ---")
    logging.info(f"Target Ticker: {Config.UNDERLYING_SYMBOL}")


def calculate_intrinsic_value(legs_json, current_spot):
    """Calculates intrinsic value of a spread at expiration."""
    try:
        legs = json.loads(legs_json)
        value = 0.0
        for leg in legs:
            strike = float(leg['strike'])
            l_type = leg.get('type', 'put').lower()
            l_side = leg.get('side', 'long').lower()
            side_multiplier = 1 if l_side == 'long' else -1

            if l_type == 'put':
                leg_val = max(0.0, strike - current_spot)
            else:
                leg_val = max(0.0, current_spot - strike)

            value += (leg_val * side_multiplier)
        return max(0.0, value)
    except Exception:
        return 0.0


def main():
    setup_logging()

    # --- 4. PATH SETUP ---
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

    # --- 5. LOAD MARKET CONTEXT & KILL SWITCH ---
    logging.info("Loading market data...")
    try:
        df_daily = pd.read_csv(DAILY_FILE)
        df_daily.columns = [c.lower().strip() for c in df_daily.columns]
        col_map = {c: 'close' for c in df_daily.columns if 'close' in c}
        col_map.update({c: 'date' for c in df_daily.columns if 'date' in c or 'timestamp' in c})
        if col_map: df_daily.rename(columns=col_map, inplace=True)

        # [CRITICAL FIX] Ensure data is sorted by date (Fixes VWAP/SMA bugs)
        if 'date' in df_daily.columns:
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily.sort_values('date', inplace=True)

        current_price = df_daily['close'].iloc[-1]
        trade_date_dt = df_daily['date'].iloc[-1]
        trade_date = trade_date_dt.strftime('%Y-%m-%d')

        # [NEW] Data Freshness Kill-Switch
        today = datetime.now()
        days_lag = (today - trade_date_dt).days
        if days_lag > 4:  # Allow for weekends/holidays
            msg = f"CRITICAL: Market data is stale! Last date: {trade_date}. Lag: {days_lag} days."
            logging.error(msg)
            if Config.APP_ENV == 'PROD':
                logging.error("ABORTING RUN (PROD SAFETY).")
                sys.exit(1)
            else:
                logging.warning("Proceeding in DEV mode with stale data...")

        logging.info(f"Market Context: Date={trade_date}, S0=${current_price:.2f}")

    except Exception as e:
        logging.error(f"CRITICAL DATA FAILURE: {e}")
        sys.exit(1)

    # --- 6. INITIALIZE MANAGERS ---
    trade_manager = None
    risk_manager = None
    if TradeManager:
        trade_manager = TradeManager()
        if PortfolioRiskManager:
            risk_manager = PortfolioRiskManager(Config, trade_manager.positions_file)
            logging.info("Portfolio Risk Manager: ACTIVE")

    # --- 7. INITIALIZE GENERATOR ---
    logging.info("Initializing Option Data...")
    generator = CandidateGenerator(CATALOG_FILE, current_price, trade_date)

    # --- 8. EXIT MANAGEMENT ---
    if trade_manager:
        logging.info("--- CHECKING OPEN POSITIONS ---")
        open_positions = trade_manager.load_open_positions()

        for pos in open_positions:
            decision = trade_manager.check_exit(pos, trade_date)

            if decision['action'] == 'EXIT':
                legs_list = json.loads(pos.get('legs_json', '[]'))
                exit_price = generator.get_composite_price(legs_list, price_type='bid')

                if exit_price <= 0.0:
                    if decision['reason'] == 'EXPIRATION':
                        exit_price = calculate_intrinsic_value(pos.get('legs_json', '[]'), current_price)
                        logging.info(f"Expiration Settlement: Intrinsic Value = ${exit_price:.2f}")
                    else:
                        logging.warning(f"Skipping Close for {pos['contractID']}: No Bid Found.")
                        continue

                logging.info(f"[EXIT EXECUTION] {pos.get('contractID')} -> {decision['reason']} @ ${exit_price:.2f}")
                trade_manager.close_position(
                    position_id=pos.get('position_id'),
                    exit_reason=decision['reason'],
                    exit_date=trade_date,
                    exit_price=exit_price
                )

    # --- 9. GENERATE CANDIDATES ---
    logging.info("--- GENERATING NEW ENTRIES ---")
    candidates = generator.generate_candidates()
    logging.info(f"Generated {len(candidates)} candidates.")

    # --- 10. INITIALIZE ENGINES ---
    mc_engine = MonteCarloEngine()
    decision_builder = TradeDecisionBuilder()
    ledger_entries = []

    # --- 11. MAIN ANALYSIS LOOP ---
    for i, candidate in enumerate(candidates):
        try:
            contract_id = candidate.get('contractID', 'UNKNOWN')
            structure_type = candidate.get('strategy', 'UNKNOWN')
            economics = candidate.get('economics', {})

            # --- [NEW] 1. LIQUIDITY & QUALITY GATE ---
            bid = economics.get('bid', 0.0)
            ask = economics.get('ask', 0.0)

            # Reject garbage quotes
            if bid <= 0 or ask <= 0:
                logging.debug(f"SKIP {contract_id}: Zero bid/ask")
                continue

            # Reject wide spreads (e.g. > $0.50 OR > 10% of price)
            spread_width = ask - bid
            if spread_width > 0.50 and spread_width > (ask * 0.10):
                logging.debug(f"SKIP {contract_id}: Spread too wide (${spread_width:.2f})")
                continue

            # --- [NEW] 2. REALISTIC EXECUTION PRICING ---
            # Paper Trade Rule: Fill at Mid + $0.02 (Conservative)
            mid_price = (bid + ask) / 2
            slippage = 0.02
            entry_price = min(ask, mid_price + slippage)

            # Update Max Loss based on new entry price
            max_loss = economics.get('max_loss', entry_price)

            legs = candidate.get('legs', [])
            expiration = legs[0]['expiration'] if legs else "N/A"

            logging.info(f"Analyzing {contract_id} (Mid: ${mid_price:.2f}, Entry: ${entry_price:.2f})...")

            # A. Monte Carlo
            mc_result = mc_engine.analyze(candidate, current_price, entry_price, trade_date=trade_date)
            candidate['CVaR95'] = mc_result.get('CVaR95', 0.0)

            # B. Decision
            decision_result = decision_builder.evaluate(mc_result, max_loss=max_loss)
            decision = decision_result.get('decision', 'ERROR')
            reason = decision_result.get('reason', 'Unknown')

            qty = 0
            risk_details = {}

            # C. Allocation [FAIL CLOSED]
            if decision == 'TRADE':
                if risk_manager:
                    # Explicitly calculate equity from Config
                    equity = getattr(Config, 'ACCOUNT_EQUITY', 15000.0)
                    qty, risk_details = risk_manager.allocate(candidate, equity=equity)

                    if qty == 0:
                        decision = 'SKIP'
                        reason = f"Risk Blocked: {risk_details.get('blocked_by', 'Unknown')}"
                else:
                    decision = 'SKIP'
                    reason = "Risk Manager Missing - Trading Blocked"
                    qty = 0

            # D. Ledger Entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'trade_date': trade_date,
                'contractID': contract_id,
                'type': structure_type,
                'entry_price': round(entry_price, 2),
                'max_loss': round(max_loss, 2),
                'decision': decision,
                'reason': reason,
                'qty_allocated': qty,
                'expected_pnl': round(mc_result.get('expected_pnl', 0), 2),
                'CVaR95': round(mc_result.get('CVaR95', 0), 2)
            }
            ledger_entries.append(entry)

            # Logging status
            log_status = decision
            if decision == 'SKIP':
                if "Risk Blocked" in reason:
                    log_status = "BLOCKED (Risk)"
                elif "Negative EV" in reason:
                    log_status = "SKIP (Neg EV)"

            logging.info(f"-> {log_status:<15} | Qty: {qty} | EV: ${entry['expected_pnl']} | Reason: {reason}")

            # E. Execution
            if decision == 'TRADE' and qty > 0 and trade_manager:
                logging.info(f"*** EXECUTION: Buying {qty}x {contract_id} ***")

                max_loss_per_contract = max_loss * 100
                cvar_per_contract = abs(candidate.get('CVaR95', 0.0)) * 100
                cand_symbol = candidate.get('symbol', symbol)
                factor_map = {'TSLA': 'QQQ', 'NVDA': 'QQQ', 'AMD': 'QQQ', 'SPY': 'SPY'}
                factor = factor_map.get(cand_symbol, 'Unmapped')
                legs_json = json.dumps(legs)

                position_record = {
                    'entry_date': trade_date,
                    'contractID': contract_id,
                    'symbol': cand_symbol,
                    'strategy': structure_type,
                    'factor': factor,
                    'expiration': expiration,
                    'legs_json': legs_json,
                    'qty': qty,
                    'entry_price': entry_price,
                    'fees': 1.0 * qty,
                    'max_loss_per_contract': round(max_loss_per_contract, 2),
                    'cvar_per_contract': round(cvar_per_contract, 2),
                    'status': 'OPEN'
                }

                trade_manager.save_new_position(position_record)

        except Exception as e:
            logging.error(f"Error processing candidate {i}: {e}", exc_info=True)
            continue

    # --- 12. SAVE LEDGER ---
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
import os
import sys
import pandas as pd
import logging
import json
import numpy as np
from datetime import datetime, timedelta

try:
    from config import Config
except ImportError:
    print("CRITICAL ERROR: 'config.py' not found.")
    sys.exit(1)

try:
    from CandidateGenerator.candidate_generator import CandidateGenerator

    try:
        from TradeDecisionBuilder.trade_decision_builder import TradeDecisionBuilder, TradeDecision, DecisionType
    except ImportError:
        from TradeDecisionBuilder.TradeDecisionBuilder import TradeDecisionBuilder, TradeDecision, DecisionType
    from Montecarlo_Sharpe.MonteCarloEngine import MonteCarloEngine

    try:
        from TradeManager.trade_manager import TradeManager
    except ImportError:
        TradeManager = None
    try:
        from TradeDecisionBuilder.portfolio_risk_manager import PortfolioRiskManager
    except ImportError:
        PortfolioRiskManager = None
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


def setup_logging():
    if hasattr(Config, 'LOGS_DIR'): os.makedirs(Config.LOGS_DIR, exist_ok=True)
    log_filename = os.path.join(Config.LOGS_DIR, "daily_pipeline.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename, mode='a'), logging.StreamHandler(sys.stdout)])
    logging.info(f"--- STARTING DAILY RUN (ENV: {Config.APP_ENV}) ---")


def classify_and_price_candidate(candidate, slippage=0.02):
    legs = candidate.get('legs', [])
    bid = candidate.get('economics', {}).get('bid', 0.0)
    ask = candidate.get('economics', {}).get('ask', 0.0)
    mid = (bid + ask) / 2.0

    if len(legs) != 2:
        return "SINGLE_LEG", False, mid + slippage, mid + slippage, 0.0

    long_leg = next((l for l in legs if l.get('side', '').lower() == 'long'), None)
    short_leg = next((l for l in legs if l.get('side', '').lower() == 'short'), None)
    if not long_leg or not short_leg: return "MALFORMED", False, 0.0, 0.0, 0.0

    l_strike = float(long_leg['strike'])
    s_strike = float(short_leg['strike'])
    width = abs(l_strike - s_strike)
    opt_type = long_leg['type'].lower()

    is_credit = False
    strategy = "UNKNOWN"

    if opt_type == 'put':
        if l_strike > s_strike:
            strategy, is_credit = "BEAR_PUT_DEBIT", False
        else:
            strategy, is_credit = "BULL_PUT_CREDIT", True
    elif opt_type == 'call':
        if l_strike < s_strike:
            strategy, is_credit = "BULL_CALL_DEBIT", False
        else:
            strategy, is_credit = "BEAR_CALL_CREDIT", True

    if is_credit:
        entry_premium = max(0.0, max(bid, mid - slippage))
        max_loss = max(0.0, width - entry_premium)
    else:
        entry_premium = min(ask, mid + slippage) if ask > 0 else mid + slippage
        max_loss = entry_premium

    return strategy, is_credit, entry_premium, max_loss, width


def analyze_candidate_robust(candidate, mc_result, decision_builder, is_credit, entry_premium, max_loss, width):
    win_rate = mc_result.get('prob_profit', mc_result.get('win_rate', 0.0))
    expected_value = mc_result.get('expected_pnl', mc_result.get('ev', 0.0))
    cvar = mc_result.get('CVaR95', -max_loss)
    sharpe = mc_result.get('downside_sharpe', mc_result.get('sharpe', 0.0))

    bid = candidate.get('economics', {}).get('bid', 0.0)
    ask = candidate.get('economics', {}).get('ask', 0.0)
    print(
        f"DEBUG: {candidate.get('strategy')} | {'CREDIT' if is_credit else 'DEBIT'} | Width: {width} | Prem: {entry_premium:.2f} | MaxLoss: {max_loss:.2f} | EV: {expected_value:.2f} | PoP: {win_rate:.2f} | Sharpe: {sharpe:.2f} | CVaR: {cvar:.2f} | B/A: {bid:.2f}/{ask:.2f}")

    current_spread = ask - bid
    allowed_spread = max(0.50, width * 0.25)
    if current_spread > allowed_spread:
        return TradeDecision(DecisionType.NO_TRADE, 0.0, datetime.now(),
                             f"Liquidity: ${current_spread:.2f} > Limit ${allowed_spread:.2f}", {})

    if not is_credit and width > 0 and entry_premium > (width * 0.60):
        return TradeDecision(DecisionType.NO_TRADE, 0.0, datetime.now(),
                             f"Expensive Debit: ${entry_premium:.2f} > 60% Width", {})

    req_pop = 0.55 if is_credit else 0.35

    return decision_builder.build_decision(expected_value, win_rate, sharpe, cvar, max_loss, min_pop_threshold=req_pop)


def main():
    setup_logging()
    symbol = Config.UNDERLYING_SYMBOL
    CATALOG_FILE = os.path.join(Config.DATA_DIR, f"{symbol}_Options_Contracts.csv")
    DAILY_FILE = os.path.join(Config.DATA_DIR, f"{symbol.lower()}_daily.csv")
    LEDGER_FILE = Config.DB_FILE

    if not os.path.exists(CATALOG_FILE) or not os.path.exists(DAILY_FILE):
        logging.error("Missing Data Files.")
        return

    df_daily = pd.read_csv(DAILY_FILE)
    df_daily.columns = [c.lower().strip() for c in df_daily.columns]
    current_price = df_daily['close'].iloc[-1]
    trade_date = str(df_daily['date'].iloc[-1])
    logging.info(f"Market: {trade_date} | S0=${current_price:.2f}")

    generator = CandidateGenerator(CATALOG_FILE, current_price, trade_date)
    mc_engine = MonteCarloEngine()
    decision_builder = TradeDecisionBuilder()

    trade_manager = None
    if TradeManager: trade_manager = TradeManager()
    risk_manager = None
    if PortfolioRiskManager and trade_manager: risk_manager = PortfolioRiskManager(Config, trade_manager.positions_file)

    candidates = generator.generate_candidates()
    logging.info(f"Generated {len(candidates)} candidates.")
    ledger_entries = []

    for i, candidate in enumerate(candidates):
        try:
            strategy, is_credit, entry_premium, max_loss, width = classify_and_price_candidate(candidate)
            mc_result = mc_engine.analyze(candidate, current_price, entry_premium, is_credit=is_credit,
                                          trade_date=trade_date)
            decision_obj = analyze_candidate_robust(candidate, mc_result, decision_builder, is_credit, entry_premium,
                                                    max_loss, width)

            decision_enum = decision_obj.decision.name
            reason = decision_obj.reason

            qty = 0
            if decision_enum == 'TRADE':
                if risk_manager:
                    candidate['economics']['max_loss'] = max_loss
                    qty, _ = risk_manager.allocate(candidate, equity=Config.ACCOUNT_EQUITY)

                    # --- DEV OVERRIDE: FORCE TRADE IF VALID BUT BLOCKED ---
                    if qty == 0:
                        logging.warning(
                            f"BLOCKED (Risk): Cost ${max_loss:.2f} > Limit. DEV OVERRIDE: Forcing 1 contract.")
                        qty = 1
                        decision_enum = 'TRADE (FORCED)'  # Mark in ledger

                else:
                    qty = 1

                    # Better Logging
            log_status = decision_enum
            if qty == 0 and decision_enum == 'TRADE': log_status = "BLOCKED"

            logging.info(
                f"-> {log_status:<15} | {strategy:<15} | Prem: ${entry_premium:.2f} | EV: ${mc_result.get('expected_pnl', 0):.2f} | Reason: {reason}")

            ledger_entries.append({
                'timestamp': datetime.now().isoformat(), 'contractID': candidate.get('contractID'),
                'strategy': strategy, 'entry_price': entry_premium, 'decision': decision_enum,
                'reason': reason, 'ev': mc_result.get('expected_pnl'), 'pop': mc_result.get('prob_profit')
            })

            if qty > 0 and trade_manager:
                logging.info(f"*** EXECUTION: {qty}x {strategy} @ ${entry_premium:.2f} ***")
                legs = candidate.get('legs', [])
                pos = {
                    'entry_date': trade_date, 'contractID': candidate.get('contractID'), 'symbol': 'TSLA',
                    'strategy': strategy, 'factor': 'Unmapped', 'expiration': legs[0]['expiration'] if legs else 'N/A',
                    'legs_json': json.dumps(legs), 'qty': qty, 'entry_price': entry_premium, 'fees': 1.0 * qty,
                    'max_loss_per_contract': round(max_loss * 100, 2),
                    'cvar_per_contract': round(abs(mc_result.get('CVaR95', 0)) * 100, 2),
                    'status': 'OPEN'
                }
                trade_manager.save_new_position(pos)

        except Exception as e:
            logging.error(f"Error {i}: {e}")

    if ledger_entries:
        try:
            pd.DataFrame(ledger_entries).to_csv(LEDGER_FILE, mode='a', header=not os.path.exists(LEDGER_FILE),
                                                index=False)
        except:
            pass


if __name__ == "__main__":
    main()
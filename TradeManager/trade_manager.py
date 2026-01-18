import pandas as pd
from datetime import datetime
try:
    from config import Config
except ImportError:
    # Fallback if config isn't found (for testing)
    class Config:
        EXIT_RULE_DTE = 21

class TradeManager:
    """
    Lifecycle Manager: Responsible solely for determining exit signals.
    Decouples exit logic from execution logic.
    """
    def __init__(self):
        # Load exit rule from Config (defaults to 21 if missing)
        self.time_stop_dte = getattr(Config, 'EXIT_RULE_DTE', 21)

    def check_exit(self, trade_record, current_date):
        """
        Evaluates an open trade against exit rules.
        Returns: {'action': 'EXIT'|'HOLD', 'reason': str}
        """
        try:
            expiration = pd.to_datetime(trade_record['expiration'])
            curr_date = pd.to_datetime(current_date)
        except Exception as e:
            return {'action': 'ERROR', 'reason': f"Date Parse Error: {e}"}

        dte = (expiration - curr_date).days

        # RULE 1: TIME STOP (TIME-21)
        # Based on backtest results: Exit if DTE is <= 21 to avoid gamma risk
        if dte <= self.time_stop_dte:
            return {
                'action': 'EXIT',
                'reason': f"TIME_STOP_HIT (DTE {dte} <= {self.time_stop_dte})"
            }

        # RULE 2: EXPIRATION (Safety Net)
        if dte <= 0:
            return {'action': 'EXIT', 'reason': "EXPIRATION"}

        # DEFAULT: HOLD
        return {'action': 'HOLD', 'reason': f"DTE {dte} > {self.time_stop_dte}"}
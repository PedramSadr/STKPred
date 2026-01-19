import pandas as pd
import os
import uuid
import logging
from datetime import datetime

try:
    from config import Config
except ImportError:
    # Fallback for testing/standalone use
    class Config:
        EXIT_RULE_DTE = 21
        LOGS_DIR = "logs"


class TradeManager:
    """
    Lifecycle Manager: Handles Exit Logic AND Position Persistence.
    Hardened for Production: Uses UUIDs and Safe File Operations.
    """

    def __init__(self):
        # 1. Load Rules
        self.time_stop_dte = getattr(Config, 'EXIT_RULE_DTE', 21)

        # 2. Define Persistence Path
        self.positions_file = os.path.join(Config.LOGS_DIR, "open_positions.csv")

        # [FIX P1] Ensure directory exists immediately to prevent IOErrors
        os.makedirs(os.path.dirname(self.positions_file), exist_ok=True)

    # --- CORE LOGIC ---
    def check_exit(self, trade_record, current_date):
        """
        Evaluates an open trade against exit rules.
        Returns: {'action': 'EXIT'|'HOLD'|'ERROR', 'reason': str}
        """
        try:
            expiration = pd.to_datetime(trade_record['expiration'])
            curr_date = pd.to_datetime(current_date)
        except Exception as e:
            return {'action': 'ERROR', 'reason': f"Date Parse Error: {e}"}

        dte = (expiration - curr_date).days

        # [FIX] RULE 1: EXPIRATION (Check this FIRST)
        if dte <= 0:
            return {'action': 'EXIT', 'reason': "EXPIRATION"}

        # [FIX] RULE 2: TIME STOP (Check this SECOND)
        if dte <= self.time_stop_dte:
            return {
                'action': 'EXIT',
                'reason': f"TIME_STOP_HIT (DTE {dte} <= {self.time_stop_dte})"
            }

        return {'action': 'HOLD', 'reason': f"DTE {dte} > {self.time_stop_dte}"}

    # --- PERSISTENCE METHODS ---
    def load_open_positions(self):
        """Reads the portfolio file and returns active trades."""
        if not os.path.exists(self.positions_file):
            return []
        try:
            df = pd.read_csv(self.positions_file)

            # [FIX] Warn if file exists but is malformed
            if 'status' not in df.columns:
                logging.warning(f"Positions file found but missing 'status' column: {self.positions_file}")
                return []

            return df[df['status'] == 'OPEN'].to_dict('records')
        except Exception as e:
            logging.error(f"Failed to load positions: {e}")
            return []

    def save_new_position(self, trade_dict):
        """Appends a new accepted trade to the positions file with a UUID."""
        # [FIX P1] Add Unique ID to prevent 'double counting' issues
        trade_dict['position_id'] = str(uuid.uuid4())

        # Enforce defaults
        trade_dict['status'] = 'OPEN'
        trade_dict['exit_date'] = None
        trade_dict['exit_reason'] = None
        trade_dict['realized_pnl'] = 0.0  # Placeholder for Phase 3

        df_new = pd.DataFrame([trade_dict])

        # Append to CSV (create header if file doesn't exist)
        if os.path.exists(self.positions_file):
            df_new.to_csv(self.positions_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.positions_file, mode='w', header=True, index=False)

        logging.info(f"Persisted new position {trade_dict['position_id']} ({trade_dict['contractID']})")

    def close_position(self, position_id, exit_reason, exit_date):
        """Updates an existing position to CLOSED status using Unique ID."""
        if not os.path.exists(self.positions_file): return

        try:
            df = pd.read_csv(self.positions_file)

            # [FIX P1] Match by Unique position_id, not just contractID
            if 'position_id' in df.columns:
                mask = df['position_id'] == position_id
            else:
                # Fallback for legacy files created before this update
                mask = df['contractID'] == position_id

            if mask.any():
                df.loc[mask, 'status'] = 'CLOSED'
                df.loc[mask, 'exit_reason'] = exit_reason
                df.loc[mask, 'exit_date'] = exit_date

                df.to_csv(self.positions_file, index=False)
                logging.info(f"Closed position {position_id}")
            else:
                logging.error(f"Could not find position_id {position_id} to close.")

        except Exception as e:
            logging.error(f"Failed to close position {position_id}: {e}")
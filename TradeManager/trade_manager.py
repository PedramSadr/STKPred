import pandas as pd
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path

try:
    from config import Config
except ImportError:
    class Config:
        EXIT_RULE_DTE = 21
        LOGS_DIR = "logs"


class TradeManager:
    """
    Lifecycle Manager: Phase 4 Hardening.
    """
    SCHEMA = [
        "position_id", "status", "entry_date", "exit_date", "exit_reason",
        "symbol", "strategy", "contractID", "factor",
        "expiration", "legs_json",
        "entry_price", "exit_price", "qty", "fees",
        "max_loss_per_contract", "cvar_per_contract", "realized_pnl"
    ]

    def __init__(self):
        self.time_stop_dte = getattr(Config, 'EXIT_RULE_DTE', 21)
        self.positions_file = os.path.join(Config.LOGS_DIR, "open_positions.csv")
        os.makedirs(os.path.dirname(self.positions_file), exist_ok=True)

    def _load_df(self):
        """Safe loader with Strict Schema Validation."""
        if not Path(self.positions_file).exists():
            return pd.DataFrame(columns=self.SCHEMA)

        try:
            df = pd.read_csv(self.positions_file)

            # [FIX] Fail Closed on Schema Drift
            missing = set(self.SCHEMA) - set(df.columns)
            if missing:
                logging.error(f"CRITICAL: Schema Mismatch in {self.positions_file}. Missing: {missing}")
                return pd.DataFrame(columns=self.SCHEMA)

            return df[self.SCHEMA]
        except Exception as e:
            logging.error(f"Corrupt Positions File: {e}")
            return pd.DataFrame(columns=self.SCHEMA)

    def _atomic_write(self, df):
        temp = self.positions_file + ".tmp"
        try:
            df.to_csv(temp, index=False)
            if os.path.exists(self.positions_file):
                os.remove(self.positions_file)
            os.rename(temp, self.positions_file)
        except Exception as e:
            logging.error(f"Atomic Write Failed: {e}")

    def save_new_position(self, trade_dict):
        df = self._load_df()

        # Idempotency
        if not df.empty and 'contractID' in df.columns:
            if not df[(df['contractID'] == trade_dict['contractID']) & (df['status'] == 'OPEN')].empty:
                logging.warning(f"IDEMPOTENCY: Skipping duplicate {trade_dict['contractID']}")
                return False

        trade_dict['position_id'] = str(uuid.uuid4())
        defaults = {'status': 'OPEN', 'realized_pnl': 0.0, 'exit_price': 0.0, 'fees': 0.0}
        for k, v in defaults.items():
            if k not in trade_dict: trade_dict[k] = v

        new_row = pd.DataFrame([trade_dict]).reindex(columns=self.SCHEMA)
        df = pd.concat([df, new_row], ignore_index=True)
        self._atomic_write(df)
        logging.info(f"Persisted {trade_dict['contractID']}")
        return True

    def close_position(self, position_id, exit_reason, exit_date, exit_price, fees=0.0):
        df = self._load_df()
        if df.empty: return

        mask = df['position_id'] == position_id
        if not mask.any(): mask = df['contractID'] == position_id

        if mask.any():
            entry_price = pd.to_numeric(df.loc[mask, 'entry_price']).iloc[0]
            qty = pd.to_numeric(df.loc[mask, 'qty']).iloc[0]

            # Spread PnL (Sell to Close)
            gross = (exit_price - entry_price) * qty * 100
            net = gross - fees

            df.loc[mask, 'status'] = 'CLOSED'
            df.loc[mask, 'exit_reason'] = exit_reason
            df.loc[mask, 'exit_date'] = exit_date
            df.loc[mask, 'exit_price'] = exit_price
            df.loc[mask, 'realized_pnl'] = round(net, 2)

            self._atomic_write(df)
            logging.info(f"Closed {position_id} @ ${exit_price:.2f} (PnL ${net:.2f})")

    def load_open_positions(self):
        df = self._load_df()
        return df[df['status'] == 'OPEN'].to_dict('records') if not df.empty else []

    def check_exit(self, trade_record, current_date):
        try:
            expiration = pd.to_datetime(trade_record['expiration'])
            curr_date = pd.to_datetime(current_date)
            dte = (expiration - curr_date).days
            if dte <= 0: return {'action': 'EXIT', 'reason': "EXPIRATION"}
            if dte <= self.time_stop_dte: return {'action': 'EXIT', 'reason': f"TIME_STOP_HIT"}
            return {'action': 'HOLD', 'reason': f"DTE {dte}"}
        except:
            return {'action': 'ERROR', 'reason': 'Date Parse'}
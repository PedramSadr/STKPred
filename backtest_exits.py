import pandas as pd
import numpy as np
from scipy.stats import norm
import logging
from datetime import timedelta
import os


# --- CONFIGURATION ---
# We use a standalone config here for the research script to keep it isolated
class BacktestConfig:
    SYMBOL = "TSLA"
    DATA_DIR = r"C:\My Documents\Mics\Logs"  # Update if needed
    START_DATE = "2025-06-01"  # Simulate last ~6 months

    # Trade Settings
    ENTRY_DTE = 45
    SPREAD_WIDTH = 5
    RISK_FREE_RATE = 0.04
    SLIPPAGE_PER_LEG = 0.02  # $2.00 slippage per spread execute (entry + exit)

    # Strategy Parameters
    TP_PCT = 0.50  # Take Profit at 50% of Max Gain
    TIME_STOP_DTE = 21  # Exit if DTE <= 21


# --- 1. BLACK-SCHOLES PRICING ENGINE ---
class BlackScholes:
    """
    Standard BSM Model to reprice options daily.
    Used to generate the 'Mark' for the spread without needing 1TB of historical CSVs.
    """

    @staticmethod
    def price(S, K, T, r, sigma, type='PUT'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if type == 'CALL':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # PUT
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price


# --- 2. TRADE OBJECT ---
class Trade:
    def __init__(self, entry_date, S0, iv):
        self.entry_date = entry_date
        self.entry_S = S0
        self.iv = iv

        # Setup ATM Put Debit Spread
        # Long Strike = Current Price (ATM)
        # Short Strike = ATM - Width
        self.long_strike = round(S0, 0)
        self.short_strike = self.long_strike - BacktestConfig.SPREAD_WIDTH

        # Expiration
        self.expiration = entry_date + timedelta(days=BacktestConfig.ENTRY_DTE)

        # Economics
        self.entry_debit = self._calculate_price(S0, BacktestConfig.ENTRY_DTE)
        self.max_gain = BacktestConfig.SPREAD_WIDTH - self.entry_debit
        self.max_loss = self.entry_debit

        # State
        self.is_open = True
        self.exit_reason = None
        self.exit_date = None
        self.exit_price = 0.0
        self.pnl_path = []  # Track daily value

    def _calculate_price(self, current_S, dte):
        if dte < 0: return 0.0
        T = dte / 365.0
        r = BacktestConfig.RISK_FREE_RATE

        # Price both legs
        p_long = BlackScholes.price(current_S, self.long_strike, T, r, self.iv, 'PUT')
        p_short = BlackScholes.price(current_S, self.short_strike, T, r, self.iv, 'PUT')

        # Spread Value = Long - Short
        return max(0.0, p_long - p_short)

    def update(self, current_date, current_S, strategy_type):
        """
        Daily Mark-to-Market and Exit Logic Check
        strategy_type: 'HOLD', 'TP_50', 'TIME_21'
        """
        if not self.is_open: return

        dte = (self.expiration - current_date).days
        mark_value = self._calculate_price(current_S, dte)
        self.pnl_path.append(mark_value - self.entry_debit)

        # --- REFINEMENT #1: EXIT BASED ON SPREAD MARK ---

        # 1. Check Expiry
        if dte <= 0:
            self.close(current_date, mark_value, "EXPIRATION")
            return

        # 2. Strategy: TIME STOP (Exit if DTE <= X)
        if strategy_type == 'TIME_21' and dte <= BacktestConfig.TIME_STOP_DTE:
            self.close(current_date, mark_value, f"TIME_STOP_{dte}")
            return

        # 3. Strategy: 50% TAKE PROFIT
        # Target Value = Entry Cost + (0.50 * Max Potential Profit)
        if strategy_type == 'TP_50':
            target_value = self.entry_debit + (BacktestConfig.TP_PCT * self.max_gain)
            if mark_value >= target_value:
                self.close(current_date, mark_value, "TAKE_PROFIT_50")
                return

        # Note: We could add Stop Loss here (e.g., mark_value <= 0.5 * entry_debit)

    def close(self, date, price, reason):
        # --- REFINEMENT #2: EXECUTION SLIPPAGE ---
        # We assume we sell the spread, so we lose a bit on the bid
        slippage_adjust = price - BacktestConfig.SLIPPAGE_PER_LEG

        self.is_open = False
        self.exit_date = date
        self.exit_price = max(0, slippage_adjust)
        self.exit_reason = reason

    def result(self):
        return (self.exit_price - self.entry_debit) * 100  # Dollar PnL (x100 multiplier)


# --- 3. BACKTEST ENGINE ---
def run_backtest():
    print(f"--- STARTING EXIT OPTIMIZATION ({BacktestConfig.SYMBOL}) ---")

    # 1. Load History
    daily_file = os.path.join(BacktestConfig.DATA_DIR, f"{BacktestConfig.SYMBOL.lower()}_daily.csv")
    if not os.path.exists(daily_file):
        print(f"Error: Could not find {daily_file}")
        return

    df = pd.read_csv(daily_file)
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    col_map = {c: 'close' for c in df.columns if 'close' in c}
    col_map.update({c: 'date' for c in df.columns if 'date' in c or 'timestamp' in c})
    df.rename(columns=col_map, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Filter for simulation period
    df = df[df['date'] >= BacktestConfig.START_DATE].copy()
    print(f"Loaded {len(df)} trading days starting {BacktestConfig.START_DATE}")

    # 2. Define Strategies to Test
    strategies = ['HOLD', 'TP_50', 'TIME_21']
    results = {}

    for strat in strategies:
        print(f"\n>> Simulating Strategy: {strat}...")
        trades = []
        active_trade = None

        # Simulation Loop
        for i, row in df.iterrows():
            current_date = row['date']
            S0 = row['close']

            # Simple Entry Logic: Enter a new trade every MONDAY if no trade is active
            # (This isolates the exit logic behavior)
            if active_trade is None and current_date.dayofweek == 0:
                # Assuming 40% IV for synthetic pricing (conservative average for TSLA)
                t = Trade(current_date, S0, iv=0.40)
                active_trade = t
                trades.append(t)

            # Manage Active Trade
            if active_trade:
                active_trade.update(current_date, S0, strat)
                if not active_trade.is_open:
                    active_trade = None  # Trade closed, ready for next one

        # Calculate Metrics for this Strategy
        pnl_values = [t.result() for t in trades if not t.is_open]
        if not pnl_values: continue

        wins = [p for p in pnl_values if p > 0]
        total_pnl = sum(pnl_values)
        win_rate = len(wins) / len(pnl_values) if pnl_values else 0

        # Sharpe (simplified annualized)
        avg_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values) if len(pnl_values) > 1 else 1
        sharpe = (avg_pnl / std_pnl) * np.sqrt(52)  # Weekly trades approx

        results[strat] = {
            'Total PnL': total_pnl,
            'Win Rate': win_rate,
            'Sharpe': sharpe,
            'Trades': len(pnl_values),
            'Avg PnL': avg_pnl
        }

    # 3. Print Scoreboard
    print("\n--- PERFORMANCE SCOREBOARD ---")
    score_df = pd.DataFrame(results).T
    print(score_df[['Total PnL', 'Win Rate', 'Sharpe', 'Avg PnL']].round(2))

    # Pick Winner
    best_strat = score_df['Sharpe'].idxmax()
    print(f"\n🏆 WINNER (Risk-Adjusted): {best_strat}")
    print(f"Reason: Highest Sharpe Ratio implies smoothest equity curve.")


if __name__ == "__main__":
    run_backtest()
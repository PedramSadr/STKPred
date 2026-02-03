import numpy as np
import pandas as pd


class MonteCarloEngine:
    """
    Simulates underlying price paths to estimate Option/Spread P&L.
    NOW SUPPORTS: Credit vs Debit logic natively.
    """

    def __init__(self, n_sims=5000, dt_days=1):
        self.n_sims = n_sims
        self.dt_days = dt_days

    def analyze(self, candidate, current_price, entry_cost, is_credit=False, trade_date=None):
        """
        Runs Monte Carlo simulation.
        entry_cost: Always positive magnitude (Debit paid or Credit received).
        is_credit: If True, PnL = Payoff + Entry_Credit (since Payoff is negative for short structures).
        """
        # 1. Setup Simulation
        if trade_date:
            try:
                # Handle YYYY-MM-DD format strictly
                current_date = pd.Timestamp(str(trade_date).split(' ')[0])
            except:
                current_date = pd.Timestamp.now()
        else:
            current_date = pd.Timestamp.now()

        legs = candidate.get('legs', [])
        if not legs: return {}

        # Get Volatility (IV) from legs or default
        # Use the first leg's IV and DTE
        iv = float(legs[0].get('iv', 0.50))
        dte_str = legs[0].get('expiration', '')

        # Calculate DTE in years
        try:
            exp_date = pd.Timestamp(dte_str)
            dte_days = (exp_date - current_date).days
        except:
            dte_days = 30  # Fallback

        T = max(1.0 / 365.0, dte_days / 365.0)

        # Simulate Terminal Prices (Geometric Brownian Motion)
        drift = -0.5 * (iv ** 2) * T
        diffusion = iv * np.sqrt(T) * np.random.normal(0, 1, self.n_sims)
        ST_prices = current_price * np.exp(drift + diffusion)

        # 2. Calculate Payoff at Expiration (Net Liquidation Value)
        payoffs = np.zeros(self.n_sims)

        for leg in legs:
            strike = float(leg['strike'])
            l_type = leg['type'].lower()
            l_side = leg['side'].lower()

            # Value of the option leg at expiration (Always >= 0)
            if l_type == 'call':
                opt_val = np.maximum(0.0, ST_prices - strike)
            else:  # put
                opt_val = np.maximum(0.0, strike - ST_prices)

            # Add to spread value (Long adds equity, Short subtracts equity)
            if l_side == 'long':
                payoffs += opt_val
            else:
                payoffs -= opt_val

        # 3. Calculate P&L
        # Logic: PnL = Final_Value - Initial_Cost
        # Debit: Cost is positive. PnL = Payoffs - Cost
        # Credit: Cost is negative (we received it). PnL = Payoffs - (-Credit) = Payoffs + Credit

        if is_credit:
            pnl_outcomes = payoffs + entry_cost
        else:
            pnl_outcomes = payoffs - entry_cost

        # 4. Metrics
        expected_pnl = np.mean(pnl_outcomes)
        prob_profit = np.mean(pnl_outcomes > 0)

        # CVaR (5%)
        # CVaR is the average of the worst 5% of outcomes
        pnl_sorted = np.sort(pnl_outcomes)
        cutoff_index = int(self.n_sims * 0.05)
        if cutoff_index == 0: cutoff_index = 1
        tail = pnl_sorted[:cutoff_index]
        cvar_95 = np.mean(tail)

        # Downside Sharpe
        # Ratio of Mean PnL to Downside Deviation
        downside_losses = pnl_outcomes[pnl_outcomes < 0]
        if len(downside_losses) > 0:
            downside_std = np.std(downside_losses)
        else:
            downside_std = 0.0001  # No losses? Infinite sharpe.

        if downside_std < 0.0001: downside_std = 0.0001

        sharpe = expected_pnl / downside_std

        return {
            'expected_pnl': expected_pnl,
            'prob_profit': prob_profit,
            'CVaR95': cvar_95,
            'downside_sharpe': sharpe,  # Standardized Key
            'sharpe': sharpe
        }
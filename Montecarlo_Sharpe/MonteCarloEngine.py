import numpy as np
import pandas as pd


class MonteCarloEngine:
    """
    Simulates underlying price paths to estimate Option/Spread P&L.
    Priority 1 Fixes:
    - Clamps P&L to spread economics (Max Gain/Max Loss).
    - Uses trade_date for consistent DTE calculation.
    - Computes CVaR (Conditional Value at Risk).
    """

    def __init__(self, n_sims=5000, dt_days=1):
        self.n_sims = n_sims
        self.dt_days = dt_days  # Step size in days

    def analyze(self, candidate, current_price, entry_cost, trade_date=None):
        """
        Runs Monte Carlo simulation with time-consistency.
        """
        # 1. Setup Time Context
        if trade_date:
            current_date = pd.Timestamp(trade_date)
        else:
            current_date = pd.Timestamp.now()

        # Extract Candidate Data
        economics = candidate.get('economics', {})
        strategy = candidate.get('strategy', 'SINGLE_PUT')
        legs = candidate.get('legs', [])

        if not legs: return {}

        # Risk bounds (P1 Compliance: Use explicit bounds from Generator)
        max_gain = economics.get('max_gain', float('inf'))
        max_loss = economics.get('max_loss', entry_cost)

        # Volatility & Drift (Simplified for now - can be connected to Fusion Model later)
        # Using a fixed annual vol of 50% for TSLA paper trading if not provided
        sigma = candidate.get('implied_volatility', 0.50)
        mu = 0.05  # Conservative 5% annual drift

        # Time to Expiration (FIX: Relative to trade_date)
        expiration = pd.to_datetime(legs[0]['expiration'])
        dte = (expiration - current_date).days

        # Safety: If trading 0-DTE or negative (data error), clamp to 1 day
        if dte < 1: dte = 1
        T = dte / 365.0

        # 2. Simulate Paths (Geometric Brownian Motion)
        # S_T = S_0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        Z = np.random.standard_normal(self.n_sims)
        ST = current_price * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

        # 3. Calculate Payoff at Expiration
        pnl_outcomes = []

        if 'SPREAD' in strategy or 'VERTICAL' in strategy:
            # Vertical Spread Logic
            long_strike = legs[0]['strike']
            short_strike = legs[1]['strike']

            # Put Debit Spread Value = Max(0, Long_Strike - ST) - Max(0, Short_Strike - ST)
            spread_value = np.maximum(0, long_strike - ST) - np.maximum(0, short_strike - ST)

            # PnL = Final Value - Entry Cost
            pnl_outcomes = spread_value - entry_cost

        else:
            # Single Put Logic
            strike = legs[0]['strike']
            option_value = np.maximum(0, strike - ST)
            pnl_outcomes = option_value - entry_cost

        # 4. PRIORITY 1: Hard Clamp (Safety Net)
        # Ensure floating point math didn't exceed bounds defined by the Generator
        pnl_outcomes = np.clip(pnl_outcomes, -max_loss, max_gain)

        # 5. Calculate Metrics
        expected_pnl = np.mean(pnl_outcomes)
        prob_profit = np.mean(pnl_outcomes > 0)

        # Downside Risk Metrics
        losses = pnl_outcomes[pnl_outcomes < 0]

        # VaR (95%)
        if len(pnl_outcomes) > 0:
            var_95 = np.percentile(pnl_outcomes, 5)  # 5th percentile (negative number)
        else:
            var_95 = -max_loss

        # CVaR (95%) - PRIORITY 1 FIX
        # Average of losses worse than VaR (The "Tail Risk")
        tail_losses = pnl_outcomes[pnl_outcomes <= var_95]
        if len(tail_losses) > 0:
            cvar_95 = np.mean(tail_losses)
        else:
            cvar_95 = var_95

        # Downside Sharpe (Sortino-ish)
        downside_std = np.std(losses) if len(losses) > 0 else 1.0
        downside_sharpe = expected_pnl / downside_std if downside_std > 0 else 0

        return {
            'expected_pnl': expected_pnl,
            'prob_profit': prob_profit,
            'VaR95': var_95,
            'CVaR95': cvar_95,  # New Metric
            'downside_sharpe': downside_sharpe,
            'simulated_price_mean': np.mean(ST),
            'drift': mu
        }
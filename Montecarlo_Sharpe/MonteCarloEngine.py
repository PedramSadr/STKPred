import numpy as np
from scipy.stats import norm


class MonteCarloEngine:
    """
    Monte Carlo Engine for Option Valuation.
    Simulates underlying price paths and calculates P&L based on a fixed holding period.
    """

    def __init__(self, num_paths=10000, hold_days=14):
        self.num_paths = num_paths
        self.hold_days = hold_days

    def generate_risk_metrics(self, fusion_output, market_state):
        # 1) Inputs
        mu_fused = float(fusion_output["mu"])
        sigma_fused = float(fusion_output["sigma"])
        aiv_fused = float(fusion_output["aiv"])

        S0 = float(market_state["S"])
        IV0 = float(market_state["IV"])
        T_exp = float(market_state["T"])  # years to expiry
        r = float(market_state["r"])
        K = float(market_state["K"])
        is_call = bool(market_state["is_call"])

        # Use REAL entry if available (preferred)
        entry_price = float(market_state.get("price", 0.0))
        if entry_price <= 0:
            entry_price = self._black_scholes_single(S0, K, r, T_exp, IV0, is_call)

        # 2) Horizon setup (2-week hold, but never beyond expiry)
        hold_T = min(self.hold_days / 365.0, T_exp)
        # Remaining time at horizon
        T_rem = max(T_exp - hold_T, 0.0)

        # 3) Simulate S at horizon (Geometric Brownian Motion)
        Z = np.random.standard_normal(self.num_paths)
        drift = (mu_fused - 0.5 * sigma_fused ** 2) * hold_T
        diffusion = sigma_fused * np.sqrt(hold_T) * Z
        S_h = S0 * np.exp(drift + diffusion)

        # 4) Simulate IV at horizon (simple dynamic)
        # We add some noise to the IV to simulate volatility risk
        vol_noise = np.random.normal(0, 0.05, self.num_paths)
        IV_h = np.maximum(0.01, IV0 + aiv_fused + vol_noise)

        # 5) Option price at horizon WITH remaining time value
        # This is the critical fix: Pricing the option at T_rem, not T=0
        terminal_option_values = self._black_scholes_vectorized(
            S_h, K, r, T_rem, IV_h, is_call
        )

        # 6) P&L distribution
        pnl = terminal_option_values - entry_price

        metrics = {
            "expected_option_price": float(np.mean(terminal_option_values)),
            "expected_pnl": float(np.mean(pnl)),
            "p10_value": float(np.percentile(terminal_option_values, 10)),
            "p90_value": float(np.percentile(terminal_option_values, 90)),
            "prob_profit": float(np.mean(pnl > 0)),
            "VaR_95": float(np.percentile(pnl, 5)),
        }
        metrics.update(self._calculate_sharpe_metrics(pnl))
        return metrics

    def _calculate_sharpe_metrics(self, pnl_array):
        mean_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array)
        if std_pnl < 1e-6:
            return {"mc_sharpe": 0.0, "downside_sharpe": 0.0}

        mc_sharpe = mean_pnl / std_pnl

        # Downside deviation (risk of loss only)
        negative = pnl_array[pnl_array < 0]
        downside_dev = np.sqrt(np.mean(negative ** 2)) if len(negative) else 1e-6
        downside_sharpe = mean_pnl / downside_dev

        return {"mc_sharpe": float(mc_sharpe), "downside_sharpe": float(downside_sharpe)}

    def _black_scholes_single(self, S, K, r, T, sigma, is_call):
        """Helper for scalar inputs (initial price check)"""
        if T <= 1e-8 or sigma <= 1e-8:
            payoff = (S - K) if is_call else (K - S)
            return max(0.0, payoff)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _black_scholes_vectorized(self, S, K, r, T, sigma, is_call):
        """Helper for vectorized inputs (simulation paths)"""
        # If no time left, intrinsic only
        if T <= 1e-8:
            payoff = (S - K) if is_call else (K - S)
            return np.maximum(0.0, payoff)

        sigma = np.maximum(sigma, 1e-8)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
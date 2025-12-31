import numpy as np
from scipy.stats import norm


class MonteCarloEngine:
    def __init__(self, num_paths=10000):
        self.num_paths = num_paths

    def generate_risk_metrics(self, fusion_output, market_state):
        # 1. Unpack Inputs
        mu_fused = fusion_output['mu']
        sigma_fused = fusion_output['sigma']
        aiv_fused = fusion_output['aiv']

        S0 = market_state['S']
        IV0 = market_state['IV']
        T = market_state['T']
        r = market_state['r']
        K = market_state['K']
        is_call = market_state['is_call']

        # Calculate Initial Option Price (Theoretical "Fair Value" at t=0)
        # Needed to determine P&L for Sharpe calculation
        initial_price = self._black_scholes_single(S0, K, r, T, IV0, is_call)

        # 2. Vectorized Simulation (GBM + Volatility Dynamics)
        Z = np.random.standard_normal(self.num_paths)
        drift = (mu_fused - 0.5 * sigma_fused ** 2) * T
        diffusion = sigma_fused * np.sqrt(T) * Z
        simulated_prices = S0 * np.exp(drift + diffusion)

        # Simulate Terminal IV (Current IV + Predicted Change + Noise)
        vol_noise = np.random.normal(0, 0.05, self.num_paths)
        simulated_ivs = np.maximum(0.01, IV0 + aiv_fused + vol_noise)

        # 3. Price Options at Horizon
        # Note: If Horizon == Expiration, T_remaining = 0
        terminal_option_values = self._black_scholes_vectorized(
            simulated_prices, K, r, 0, simulated_ivs, is_call
        )

        # 4. Calculate P&L Distribution
        # Profit = Terminal Value - Initial Cost
        pnl_distribution = terminal_option_values - initial_price

        # 5. Advanced Risk Metrics
        metrics = {
            "expected_option_price": np.mean(terminal_option_values),
            "expected_pnl": np.mean(pnl_distribution),  # [cite: 102]
            "p10_value": np.percentile(terminal_option_values, 10),
            "p90_value": np.percentile(terminal_option_values, 90),
            "prob_profit": np.mean(pnl_distribution > 0),  # [cite: 111]
            "VaR_95": np.percentile(pnl_distribution, 5),  # [cite: 112]
        }

        # 6. Calculate Sharpe Ratios [cite: 98, 105]
        metrics.update(self._calculate_sharpe_metrics(pnl_distribution))

        return metrics

    def _calculate_sharpe_metrics(self, pnl_array):
        """
        Calculates Standard and Downside-Adjusted Sharpe Ratios based on
        simulated P&L distribution.
        """
        mean_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array)

        # Avoid division by zero
        if std_pnl < 1e-6:
            return {"mc_sharpe": 0.0, "downside_sharpe": 0.0}

        # Standard Monte Carlo Sharpe [cite: 104]
        # (Mean P&L / Std Dev of P&L)
        mc_sharpe = mean_pnl / std_pnl

        # Downside-Adjusted Sharpe (Sortino-style) [cite: 105, 107]
        # Only considers volatility where P&L < 0
        negative_returns = pnl_array[pnl_array < 0]

        if len(negative_returns) == 0:
            downside_deviation = 1e-6  # Perfect trade, no downside found
        else:
            # Root Mean Square of negative returns
            downside_deviation = np.sqrt(np.mean(negative_returns ** 2))

        downside_sharpe = mean_pnl / downside_deviation

        return {
            "mc_sharpe": mc_sharpe,
            "downside_sharpe": downside_sharpe
        }

    def _black_scholes_single(self, S, K, r, T, sigma, is_call):
        """Helper for scalar BS calculation at t=0"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _black_scholes_vectorized(self, S, K, r, T, sigma, is_call):
        """Vectorized BS for simulation arrays"""
        if T <= 1e-5:
            payoff = S - K if is_call else K - S
            return np.maximum(0, payoff)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --- Usage Example ---
if __name__ == "__main__":
    # Example Market Data
    fusion_output = {'mu': 0.12, 'sigma': 0.35, 'aiv': -0.01}
    market_state = {'S': 100, 'IV': 0.30, 'T': 30 / 365, 'r': 0.04, 'K': 105, 'is_call': True}

    engine = MonteCarloEngine(num_paths=50000)
    metrics = engine.generate_risk_metrics(fusion_output, market_state)

    print(f"Expected P&L:       ${metrics['expected_pnl']:.2f}")
    print(f"MC Sharpe Ratio:     {metrics['mc_sharpe']:.2f}")
    print(f"Downside Sharpe:     {metrics['downside_sharpe']:.2f} (Recommended Filter)")
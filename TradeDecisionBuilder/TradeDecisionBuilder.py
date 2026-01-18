class TradeDecisionBuilder:
    """
    Evaluates Monte Carlo results against risk thresholds to output a GO/NO-GO decision.
    NOW CONTRACT-AWARE: Uses explicit max_loss to detect binary gambles.
    """

    def __init__(self):
        # Thresholds
        self.MIN_PROB_PROFIT = 0.60  # 60% Win Rate
        self.MIN_EV_RATIO = 0.15  # Expect to make 15% return on risk
        self.MAX_TAIL_RISK_RATIO = 0.90  # Reject if CVaR is > 90% of Max Loss

    def evaluate(self, mc_result: dict, max_loss: float = None) -> dict:
        """
        Returns {'decision': 'TRADE'|'SKIP', 'reason': str}
        """
        # 1. Unpack Metrics
        ev = mc_result.get('expected_pnl', 0)
        prob_profit = mc_result.get('prob_profit', 0)
        cvar_95 = mc_result.get('CVaR95', -9999)  # Expected tail loss (negative number)

        # 2. Gate 1: Positive Expectancy
        # We want the house edge, not a fair coin flip.
        if ev <= 0:
            return {'decision': 'SKIP', 'reason': f"Negative EV: ${ev:.2f}"}

        # 3. Gate 2: Probability of Profit
        if prob_profit < self.MIN_PROB_PROFIT:
            return {'decision': 'SKIP', 'reason': f"Low Win Rate: {prob_profit:.1%}"}

        # 4. Gate 3: Tail Risk (CVaR vs EV)
        # Ensure the reward (EV) justifies the risk (CVaR).
        # We generally want EV to be at least 10-15% of the average bad outcome.
        if cvar_95 < 0:
            risk_reward_ratio = ev / abs(cvar_95)
            if risk_reward_ratio < 0.10:
                return {'decision': 'SKIP', 'reason': f"Poor Risk/Reward (CVaR): {risk_reward_ratio:.2f}"}

        # 5. Gate 4: "Binary Gamble" Check (The P2 Fix)
        # If max_loss is provided, ensure we aren't risking the ENTIRE collateral
        # just to pick up pennies. If CVaR is basically the Max Loss, it's an "All or Nothing" bet.
        if max_loss and max_loss > 0:
            # Calculate how much of the max_loss is consumed by the tail risk
            tail_consumption = abs(cvar_95) / max_loss

            if tail_consumption > self.MAX_TAIL_RISK_RATIO:
                return {
                    'decision': 'SKIP',
                    'reason': f"Binary Gamble: CVaR is {tail_consumption:.1%} of Max Loss"
                }

        return {'decision': 'TRADE', 'reason': f"PASS | EV:${ev:.2f} | Win:{prob_profit:.1%}"}
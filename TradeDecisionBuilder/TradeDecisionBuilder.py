import logging


class TradeDecisionBuilder:
    """
    Evaluates Monte Carlo results to make a YES/NO trading decision.
    Updated: Lowered thresholds to allow valid paper trades (50% WR).
    """

    def __init__(self):
        # [TUNING] Set to 0.50 to allow 50/50 bets if EV is positive.
        self.MIN_WIN_RATE = 0.50

        # Keep EV > 0. We don't want to lose money, but even $0.01 is technically an edge.
        self.MIN_EV = 0.00

    def evaluate(self, mc_result, max_loss):
        """
        Returns {'decision': 'TRADE'|'SKIP', 'reason': '...'}
        """
        try:
            expected_pnl = mc_result.get('expected_pnl', -1.0)
            win_rate = mc_result.get('win_rate', 0.0)

            # 1. EV Check (Must be positive)
            if expected_pnl <= self.MIN_EV:
                return {
                    'decision': 'SKIP',
                    'reason': f"Negative EV: ${expected_pnl:.2f}"
                }

            # 2. Win Rate Check
            if win_rate < self.MIN_WIN_RATE:
                return {
                    'decision': 'SKIP',
                    'reason': f"Low Win Rate: {win_rate * 100:.1f}% (Req: {self.MIN_WIN_RATE * 100:.0f}%)"
                }

            # 3. Decision Passed
            return {
                'decision': 'TRADE',
                'reason': f"Acceptable: EV ${expected_pnl:.2f} | WR {win_rate * 100:.1f}%"
            }

        except Exception as e:
            logging.error(f"Decision Logic Failed: {e}")
            return {'decision': 'ERROR', 'reason': str(e)}
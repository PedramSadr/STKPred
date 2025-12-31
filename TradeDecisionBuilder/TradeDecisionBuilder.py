from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import datetime


# --- 1. Enums & Constants ---
class DecisionType(Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


# --- 2. The Immutable Trade Decision Object (Section 5) ---
@dataclass(frozen=True)
class TradeDecision:
    """
    Immutable object containing the final decision, risk metrics,
    and human-readable justification.
    """
    decision: DecisionType
    confidence_score: float
    timestamp: datetime.datetime

    # Justification for the decision (e.g., "Passed all gates" or "Failed CVaR gate")
    reason: str

    # Snapshot of metrics used for the decision (for auditing)
    metrics_snapshot: dict = field(default_factory=dict)

    # List of specific gates that passed/failed
    gates_log: List[str] = field(default_factory=list)


# --- 3. The Confidence Calculator (Section 6) ---
class ConfidenceCalculator:
    """
    Deterministic score derived from normalized metrics[cite: 31].
    Never learned or optimized[cite: 32].
    """
    WEIGHT_DOWNSIDE_SHARPE = 0.50
    WEIGHT_POP = 0.30
    WEIGHT_EXPECTED_PNL = 0.20

    CAP_DOWNSIDE_SHARPE = 3.0
    CAP_POP = 0.90
    CAP_PNL = 500.0

    def calculate(self, prob_profit: float, downside_sharpe: float, expected_pnl: float) -> float:
        norm_sharpe = self._normalize(downside_sharpe, 0.0, self.CAP_DOWNSIDE_SHARPE)
        norm_pop = self._normalize(prob_profit, 0.0, self.CAP_POP)
        norm_pnl = self._normalize(expected_pnl, 0.0, self.CAP_PNL)

        raw_score = (
                (norm_sharpe * self.WEIGHT_DOWNSIDE_SHARPE) +
                (norm_pop * self.WEIGHT_POP) +
                (norm_pnl * self.WEIGHT_EXPECTED_PNL)
        )
        return round(raw_score, 4)

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if value <= min_val: return 0.0
        if value >= max_val: return 1.0
        return (value - min_val) / (max_val - min_val)


# --- 4. The Trade Decision Builder (Main Logic) ---
class TradeDecisionBuilder:
    """
    Final decision-making component. Applies deterministic gates
    to produce an immutable TradeDecision[cite: 3, 4].
    """

    def __init__(self):
        self.calculator = ConfidenceCalculator()

    def build_decision(self,
                       expected_pnl: float,
                       prob_profit: float,
                       downside_sharpe: float,
                       cvar: float,
                       premium: float,
                       volatility_consistent: bool,
                       no_macro_events: bool,
                       system_healthy: bool) -> TradeDecision:

        gates_log = []
        metrics = {
            "expected_pnl": expected_pnl,
            "prob_profit": prob_profit,
            "downside_sharpe": downside_sharpe,
            "cvar": cvar,
            "premium": premium
        }

        # --- GATING RULES (Strict Order)  ---

        # 1. Expected P&L > 0 [cite: 20]
        if expected_pnl <= 0:
            return self._reject("Expected P&L <= 0", metrics, gates_log)
        gates_log.append("PASS: Expected P&L > 0")

        # 2. Probability of Profit >= 0.45 [cite: 21]
        if prob_profit < 0.45:
            return self._reject(f"PoP {prob_profit} < 0.45", metrics, gates_log)
        gates_log.append("PASS: PoP >= 0.45")

        # 3. Downside Sharpe >= 0.40 [cite: 22]
        if downside_sharpe < 0.40:
            return self._reject(f"Downside Sharpe {downside_sharpe} < 0.40", metrics, gates_log)
        gates_log.append("PASS: Downside Sharpe >= 0.40")

        # 4. CVaR >= -0.65 * premium [cite: 23]
        # Note: CVaR is usually negative (loss). We want it to be LESS negative than threshold.
        # e.g. CVaR -60 is better than limit -65. (-60 >= -65 is True)
        cvar_limit = -0.65 * premium
        if cvar < cvar_limit:
            return self._reject(f"CVaR {cvar} exceeds limit {cvar_limit}", metrics, gates_log)
        gates_log.append("PASS: CVaR Limit")

        # 5. Volatility Regime Consistency [cite: 24]
        if not volatility_consistent:
            return self._reject("Volatility Regime Inconsistent", metrics, gates_log)
        gates_log.append("PASS: Volatility Regime")

        # 6. No Disallowed Events [cite: 25]
        if not no_macro_events:
            return self._reject("Macro/Earnings Event Detected", metrics, gates_log)
        gates_log.append("PASS: No Events")

        # 7. System Health [cite: 27]
        if not system_healthy:
            return self._reject("System Health/Drawdown Fail", metrics, gates_log)
        gates_log.append("PASS: System Health")

        # --- ALL GATES PASSED ---

        # Calculate Confidence only after passing gates [User Refinement 1]
        score = self.calculator.calculate(prob_profit, downside_sharpe, expected_pnl)

        return TradeDecision(
            decision=DecisionType.TRADE,
            confidence_score=score,
            timestamp=datetime.datetime.now(),
            reason="All deterministic gates passed.",
            metrics_snapshot=metrics,
            gates_log=gates_log
        )

    def _reject(self, reason: str, metrics: dict, gates_log: List[str]) -> TradeDecision:
        """Helper to create a NO_TRADE decision immediately upon failure."""
        gates_log.append(f"FAIL: {reason}")
        return TradeDecision(
            decision=DecisionType.NO_TRADE,
            confidence_score=0.0,  # Zero confidence on reject
            timestamp=datetime.datetime.now(),
            reason=reason,
            metrics_snapshot=metrics,
            gates_log=gates_log
        )


# --- Example Usage ---
if __name__ == "__main__":
    builder = TradeDecisionBuilder()

    # Scenario: A valid trade
    decision = builder.build_decision(
        expected_pnl=150.0,
        prob_profit=0.68,
        downside_sharpe=1.2,
        cvar=-50.0,
        premium=100.0,  # Limit would be -65.0, so -50 is safe
        volatility_consistent=True,
        no_macro_events=True,
        system_healthy=True
    )

    print(f"Decision: {decision.decision}")
    print(f"Reason: {decision.reason}")
    print(f"Score: {decision.confidence_score}")
    print(f"Log: {decision.gates_log}")
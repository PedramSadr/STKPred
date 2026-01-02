from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import datetime


# --- 1. Enums & Constants ---
class DecisionType(Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"
    REJECT = "REJECT"  # Alias for internal rejection logic


# --- 2. The Immutable Trade Decision Object ---
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


# --- 3. The Confidence Calculator ---
class ConfidenceCalculator:
    """
    Deterministic score derived from normalized metrics.
    Never learned or optimized.
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
    to produce an immutable TradeDecision.
    """

    def __init__(self):
        self.calculator = ConfidenceCalculator()

    def build_decision(self,
                       expected_pnl: float,
                       prob_profit: float,
                       downside_sharpe: float,
                       cvar: float,
                       premium: float,
                       volatility_consistent: bool = True,
                       no_macro_events: bool = True,
                       system_healthy: bool = True) -> TradeDecision:

        gates_log = []
        metrics = {
            "expected_pnl": expected_pnl,
            "prob_profit": prob_profit,
            "downside_sharpe": downside_sharpe,
            "cvar": cvar,
            "premium": premium
        }

        # --- GATING RULES (Strict Order)  ---

        # 0. System Checks
        if not system_healthy:
            return self._reject("System Unhealthy", metrics, gates_log)
        if not no_macro_events:
            return self._reject("Macro Event Risk", metrics, gates_log)
        if not volatility_consistent:
            return self._reject("Vol Regime Mismatch", metrics, gates_log)

        # 1. Expected P&L > 0
        if expected_pnl <= 0:
            return self._reject(f"Negative Expectancy: {expected_pnl:.2f}", metrics, gates_log)
        gates_log.append("PASS: Expected P&L > 0")

        # 2. Probability of Profit >= 0.40
        if prob_profit < 0.40:
            return self._reject(f"PoP {prob_profit:.2f} < 0.40", metrics, gates_log)
        gates_log.append("PASS: PoP >= 0.40")

        # 3. Downside Sharpe (Gray Zone Logic)
        # ---------------------------------------------------------
        TARGET_SHARPE = 0.35
        GRAY_ZONE_FLOOR = 0.34

        # Explicit Aux Thresholds for Gray Zone
        GRAY_POP_THRESH = 0.42
        GRAY_PNL_THRESH = 4.5  # Raised to 4.5 per recommendation

        if downside_sharpe < TARGET_SHARPE:
            if downside_sharpe >= GRAY_ZONE_FLOOR:
                # In Gray Zone: Require stronger aux metrics to compensate
                if prob_profit < GRAY_POP_THRESH or expected_pnl < GRAY_PNL_THRESH:
                    reason = (
                        f"Downside Sharpe {downside_sharpe:.3f} in Gray Zone. "
                        f"Override Failed: Needs PoP >= {GRAY_POP_THRESH} (got {prob_profit:.2f}) "
                        f"AND PnL >= {GRAY_PNL_THRESH} (got {expected_pnl:.2f})"
                    )
                    return self._reject(reason, metrics, gates_log)
                else:
                    gates_log.append(f"PASS: Downside Sharpe {downside_sharpe:.3f} (Gray Zone Override)")
            else:
                # Hard Fail below floor
                return self._reject(f"Downside Sharpe {downside_sharpe:.3f} < {GRAY_ZONE_FLOOR} (Hard Floor)", metrics,
                                    gates_log)
        else:
            gates_log.append(f"PASS: Downside Sharpe {downside_sharpe:.3f} >= {TARGET_SHARPE}")

        # 4. CVaR / Tail Risk Check
        if abs(cvar) > 3.0 * premium and premium > 0:
            return self._reject("Extreme Tail Risk (CVaR > 3x Premium)", metrics, gates_log)
        gates_log.append("PASS: CVaR Limit")

        # --- ALL GATES PASSED ---

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
        """Helper to create a NO_TRADE/REJECT decision immediately upon failure."""
        gates_log.append(f"FAIL: {reason}")
        return TradeDecision(
            decision=DecisionType.REJECT,
            confidence_score=0.0,
            timestamp=datetime.datetime.now(),
            reason=reason,
            metrics_snapshot=metrics,
            gates_log=gates_log
        )
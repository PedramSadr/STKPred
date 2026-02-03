import logging
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
import datetime


class DecisionType(Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class TradeDecision:
    decision: DecisionType
    confidence_score: float
    timestamp: datetime.datetime
    reason: str
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    gates_log: List[str] = field(default_factory=list)


class ConfidenceCalculator:
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
        return round((norm_sharpe * self.WEIGHT_DOWNSIDE_SHARPE) + (norm_pop * self.WEIGHT_POP) + (
                    norm_pnl * self.WEIGHT_EXPECTED_PNL), 4)

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if value <= min_val: return 0.0
        if value >= max_val: return 1.0
        return (value - min_val) / (max_val - min_val)


class TradeDecisionBuilder:
    def __init__(self):
        self.calculator = ConfidenceCalculator()

    def build_decision(self, expected_pnl, prob_profit, downside_sharpe, cvar, premium,
                       volatility_consistent=True, no_macro_events=True, system_healthy=True, min_pop_threshold=0.35):

        gates_log = []
        metrics = {"expected_pnl": expected_pnl, "prob_profit": prob_profit, "downside_sharpe": downside_sharpe,
                   "cvar": cvar, "premium": premium}

        # 1. EV > 0
        if expected_pnl <= 0: return self._reject(f"Negative EV: ${expected_pnl:.2f}", metrics, gates_log)
        gates_log.append("PASS: Expected P&L > 0")

        # 2. Edge (Optional, >1% ROI)
        if premium > 0 and (expected_pnl / premium) < 0.01:
            return self._reject(f"Edge {expected_pnl / premium:.1%} < 1.0%", metrics, gates_log)
        gates_log.append("PASS: Edge > 1%")

        # 3. PoP
        if prob_profit < min_pop_threshold: return self._reject(f"PoP {prob_profit:.2f} < {min_pop_threshold}", metrics,
                                                                gates_log)
        gates_log.append(f"PASS: PoP >= {min_pop_threshold}")

        # 4. Downside Sharpe (Debit Friendly Floor: 0.20)
        MIN_DS = 0.20
        if downside_sharpe < MIN_DS: return self._reject(f"Downside Sharpe {downside_sharpe:.2f} < {MIN_DS:.2f}",
                                                         metrics, gates_log)
        gates_log.append(f"PASS: Downside Sharpe >= {MIN_DS:.2f}")

        # 5. CVaR Limit (Max Loss)
        # Use rounding epsilon (0.01) instead of % buffer to fix exact-match failures
        cvar_limit = -1.0 * premium
        if cvar < (cvar_limit - 0.01):
            return self._reject(f"CVaR {cvar:.2f} exceeds limit {cvar_limit:.2f}", metrics, gates_log)
        gates_log.append("PASS: CVaR Limit")

        # 6. System Checks
        if not volatility_consistent: return self._reject("Volatility Regime Inconsistent", metrics, gates_log)
        if not no_macro_events: return self._reject("Macro/Earnings Event Detected", metrics, gates_log)
        if not system_healthy: return self._reject("System Health Fail", metrics, gates_log)

        score = self.calculator.calculate(prob_profit, downside_sharpe, expected_pnl)
        return TradeDecision(DecisionType.TRADE, score, datetime.datetime.now(), "All gates passed.", metrics,
                             gates_log)

    def _reject(self, reason, metrics, gates_log):
        gates_log.append(f"FAIL: {reason}")
        return TradeDecision(DecisionType.NO_TRADE, 0.0, datetime.datetime.now(), reason, metrics, gates_log)
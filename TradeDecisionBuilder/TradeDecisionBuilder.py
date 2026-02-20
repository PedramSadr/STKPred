from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import datetime


# --- 1. Enums & Constants ---
class DecisionType(Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"
    REJECT = "REJECT"


# --- 2. The Immutable Trade Decision Object ---
@dataclass(frozen=True)
class TradeDecision:
    decision: DecisionType
    confidence_score: float
    timestamp: datetime.datetime
    reason: str
    metrics_snapshot: dict = field(default_factory=dict)
    gates_log: List[str] = field(default_factory=list)


# --- 3. The Confidence Calculator ---
class ConfidenceCalculator:
    WEIGHT_DOWNSIDE_SHARPE = 0.50
    WEIGHT_POP = 0.30
    WEIGHT_EXPECTED_PNL = 0.20

    CAP_DOWNSIDE_SHARPE = 3.0
    CAP_POP = 0.90
    CAP_PNL = 500.0  # <--- This is Contract Dollars ($500)

    def calculate(self, prob_profit: float, downside_sharpe: float, ev_dollars: float) -> float:
        norm_sharpe = self._normalize(downside_sharpe, 0.0, self.CAP_DOWNSIDE_SHARPE)
        norm_pop = self._normalize(prob_profit, 0.0, self.CAP_POP)
        # FIX 1: Scoring now uses ev_dollars so it correctly maps against the 500.0 cap
        norm_pnl = self._normalize(ev_dollars, 0.0, self.CAP_PNL)

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
    def __init__(self):
        self.calculator = ConfidenceCalculator()

    def build_decision(self,
                       expected_pnl: float,
                       prob_profit: float,
                       downside_sharpe: float,
                       cvar: float,
                       premium: float,
                       spread_abs: float,
                       downside_sharpe_std: float = 0.0,
                       volatility_consistent: bool = True,
                       no_macro_events: bool = True,
                       system_healthy: bool = True) -> TradeDecision:

        gates_log = []

        # =========================================================================
        # THE ONE CLEAN RULE: NORMALIZE EVERYTHING TO TOTAL CONTRACT DOLLARS (x100)
        # =========================================================================
        # We assume expected_pnl, premium, cvar, and spread_abs arrive as $/share.
        ev_dollars = expected_pnl * 100
        premium_dollars = premium * 100
        cvar_dollars = cvar * 100
        spread_cost_dollars = spread_abs * 100

        # Friction multiplier: Set to 1.5 assuming aggressive market orders or slippage
        min_required_edge = 1.5 * spread_cost_dollars

        metrics = {
            "expected_pnl_share": expected_pnl,
            "cvar_share": cvar,
            "premium_share": premium,
            "spread_abs_share": spread_abs,
            "prob_profit": prob_profit,
            "downside_sharpe": downside_sharpe,
            "downside_sharpe_std": downside_sharpe_std,
            "ev_dollars": ev_dollars,
            "premium_dollars": premium_dollars,
            "cvar_dollars": cvar_dollars,
            "spread_cost_dollars": spread_cost_dollars,
            "min_required_edge": min_required_edge
        }

        # --- GATING RULES ---

        # 0. System Checks
        if not system_healthy: return self._reject("System Unhealthy", metrics, gates_log)
        if not no_macro_events: return self._reject("Macro Event Risk", metrics, gates_log)
        if not volatility_consistent: return self._reject("Vol Regime Mismatch", metrics, gates_log)

        # 1. Base P&L Check
        if ev_dollars <= 0:
            return self._reject(f"Negative Expectancy: ${ev_dollars:.2f}", metrics, gates_log)
        gates_log.append("PASS: Expected P&L > $0")

        # 1.5 THE KILLER GATE: EV vs Spread Cost
        if ev_dollars < min_required_edge:
            return self._reject(
                f"Edge too small: EV (${ev_dollars:.2f}) < 1.5x Spread Cost (${min_required_edge:.2f})",
                metrics, gates_log
            )
        gates_log.append(f"PASS: EV (${ev_dollars:.2f}) >= 1.5x Spread Cost (${min_required_edge:.2f})")

        # 2. PoP >= 0.40
        if prob_profit < 0.40:
            return self._reject(f"PoP {prob_profit:.2f} < 0.40", metrics, gates_log)
        gates_log.append("PASS: PoP >= 0.40")

        # 3. Downside Sharpe (Gray Zone Logic)
        TARGET_SHARPE = 0.35
        GRAY_ZONE_FLOOR = 0.34
        GRAY_POP_THRESH = 0.42
        GRAY_PNL_THRESH = 450.0  # $450 total expected profit per contract
        EPS = 1e-9

        is_gray_zone = False
        is_extended_gray_zone = False

        if downside_sharpe >= TARGET_SHARPE:
            gates_log.append(f"PASS: Downside Sharpe {downside_sharpe:.3f} >= {TARGET_SHARPE}")
        else:
            if downside_sharpe >= GRAY_ZONE_FLOOR:
                is_gray_zone = True
            elif (downside_sharpe + downside_sharpe_std) >= GRAY_ZONE_FLOOR:
                is_extended_gray_zone = True

            if is_gray_zone or is_extended_gray_zone:
                pop_ok = (prob_profit + EPS) >= GRAY_POP_THRESH
                pnl_ok = (ev_dollars + EPS) >= GRAY_PNL_THRESH

                if not (pop_ok and pnl_ok):
                    zone_type = "Extended Gray Zone" if is_extended_gray_zone else "Gray Zone"
                    pop_status = "PASS" if pop_ok else "FAIL"
                    pnl_status = "PASS" if pnl_ok else "FAIL"

                    reason = (
                        f"Downside Sharpe {downside_sharpe:.3f} in {zone_type} (std={downside_sharpe_std:.3f}). "
                        f"Override Failed: PoP: {pop_status}, PnL: {pnl_status} (${ev_dollars:.2f} vs ${GRAY_PNL_THRESH})"
                    )
                    return self._reject(reason, metrics, gates_log)
                else:
                    msg = "Extended Gray Zone Override" if is_extended_gray_zone else "Gray Zone Override"
                    gates_log.append(f"PASS: Downside Sharpe {downside_sharpe:.3f} ({msg})")
            else:
                return self._reject(
                    f"Downside Sharpe {downside_sharpe:.3f} + std {downside_sharpe_std:.3f} < {GRAY_ZONE_FLOOR}",
                    metrics, gates_log
                )

        # 4. Tail Risk (FIX 2: Apples-to-Apples Contract Dollars Comparison)
        if abs(cvar_dollars) > 3.0 * premium_dollars and premium_dollars > 0:
            return self._reject(
                f"Extreme Tail Risk: CVaR (${abs(cvar_dollars):.2f}) > 3x Premium (${premium_dollars:.2f})", metrics,
                gates_log)
        gates_log.append("PASS: CVaR Limit")

        # FIX 1: Pass ev_dollars directly into the confidence calculator
        score = self.calculator.calculate(prob_profit, downside_sharpe, ev_dollars)

        return TradeDecision(DecisionType.TRADE, score, datetime.datetime.now(), "All deterministic gates passed.",
                             metrics, gates_log)

    def _reject(self, reason: str, metrics: dict, gates_log: List[str]) -> TradeDecision:
        gates_log.append(f"FAIL: {reason}")
        return TradeDecision(DecisionType.REJECT, 0.0, datetime.datetime.now(), reason, metrics, gates_log)
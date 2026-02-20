"""
MarketStateAdapter
------------------
Purpose: Translate a CandidateTrade into the market_state dictionary.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class MarketState:
    S: float  # Spot price
    IV: float  # Current implied volatility
    T: float  # Time to expiry (years)
    r: float  # Risk-free rate
    K: float  # Strike
    price: float  # Current option price
    is_call: bool  # Call or Put
    mu: float  # Expected stock drift (AI Prediction)
    sigma: float  # Expected stock volatility (AI Prediction)
    aiv: float  # Expected change in implied vol (AI Prediction)


class MarketStateAdapter:
    def __init__(self, risk_free_rate: float = 0.04):
        self.r = risk_free_rate

    def build_market_state(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        self._validate_candidate(candidate)

        # At this stage, we are guaranteed to have exactly 1 leg
        leg = candidate["legs"][0]

        try:
            S0 = float(candidate["underlying_price"])
            K = float(leg["strike"])
            IV0 = float(leg["iv"])
            entry_price = float(leg["entry_price"])
            dte = int(leg["dte"])
            is_call = (leg["type"].lower() == "call")

            # --- THE FIX: Extract all three AI predictions ---
            mu = float(candidate["mu"])
            sigma = float(candidate["sigma"])
            aiv = float(candidate["aiv"])

        except KeyError as e:
            raise ValueError(f"Candidate violates Canonical Schema. Missing field: {e}")

        market_state = MarketState(
            S=S0,
            IV=IV0,
            T=dte / 365.0,
            r=self.r,
            K=K,
            price=entry_price,
            is_call=is_call,
            mu=mu,
            sigma=sigma,
            aiv=aiv
        )
        return market_state.__dict__

    def _validate_candidate(self, candidate: Dict[str, Any]) -> None:
        if "legs" not in candidate or not candidate["legs"]:
            raise ValueError("Candidate must contain at least one option leg")

        # SAFETY: Protect against multi-leg candidates reaching single-leg logic
        if len(candidate["legs"]) != 1:
            raise NotImplementedError(
                f"Adapter received {len(candidate['legs'])} legs. "
                "Multi-leg valuation is not supported yet."
            )

    def adapt(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        return self.build_market_state(candidate)
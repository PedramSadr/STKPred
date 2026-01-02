import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class ExitRules:
    use_path_exits: bool = False
    hold_days: int = 14

    # exits (expressed as % return on option premium)
    take_profit_pct: float = 0.50  # +50%
    stop_loss_pct: float = 0.30  # -30%
    be_trigger_pct: float = 0.20  # arm BE stop after +20%
    be_exit_buffer: float = 0.00  # 0.00 => true break-even

    # time/vol conventions
    year_days: float = 252.0  # 252 trading days
    aiv_is_10day_change: bool = True

    # NEW: IV noise control + optional IV cap
    iv_daily_std: float = 0.01
    iv_min: float = 0.01
    iv_max: float = 3.00  # set None if you don't want a cap


class MonteCarloEngine:
    """
    Monte Carlo Engine for Option Valuation.
    Supports both Terminal-Only valuation and Path-Dependent simulation with exits.
    """

    def __init__(self, num_paths=50000, exit_config: ExitRules = None):
        self.num_paths = num_paths
        self.cfg = exit_config if exit_config else ExitRules()

    def generate_risk_metrics(self, fusion_output, market_state):
        if self.cfg.use_path_exits:
            return self._simulate_path_exits(fusion_output, market_state)
        else:
            return self._simulate_terminal_only(fusion_output, market_state)

    # -------------------------------
    # 1. PATH-EXIT SIMULATION (PATCHED)
    # -------------------------------
    def _simulate_path_exits(self, fusion_output, market_state):
        """
        Path-dependent simulation with TP/SL/BE.
        Includes steps clamping, configurable IV noise, and exit diagnostics.
        """
        mu = float(fusion_output["mu"])
        sigma = float(fusion_output["sigma"])
        aiv = float(fusion_output["aiv"])

        S0 = float(market_state["S"])
        IV0 = float(market_state["IV"])
        T_exp = float(market_state["T"])
        r = float(market_state["r"])
        K = float(market_state["K"])
        is_call = bool(market_state["is_call"])

        entry_price = float(market_state.get("price", 0.0))
        if entry_price <= 0:
            entry_price = self._black_scholes_single(S0, K, r, T_exp, IV0, is_call)

        steps = int(self.cfg.hold_days)
        if steps <= 0:
            pnl = np.zeros(self.num_paths, dtype=np.float64)
            terminal_vals = np.full(self.num_paths, entry_price, dtype=np.float64)
            return self._compute_stats(pnl, terminal_vals)

        year_days = float(getattr(self.cfg, "year_days", 252.0))
        dt = 1.0 / year_days

        # --- FIX: clamp to expiry safely ---
        if T_exp <= 0.0:
            terminal_val = self._black_scholes_single(S0, K, r, 0.0, IV0, is_call)
            terminal_vals = np.full(self.num_paths, terminal_val, dtype=np.float64)
            pnl = terminal_vals - entry_price
            return self._compute_stats(pnl, terminal_vals)

        max_steps = int(np.floor(T_exp / dt))  # number of full steps before expiry
        if max_steps <= 0:
            # can't even advance one step before expiry
            terminal_val = self._black_scholes_single(S0, K, r, T_exp, IV0, is_call)
            terminal_vals = np.full(self.num_paths, terminal_val, dtype=np.float64)
            pnl = terminal_vals - entry_price
            return self._compute_stats(pnl, terminal_vals)

        steps = min(steps, max_steps)
        if steps <= 0:
            terminal_vals = np.full(self.num_paths, entry_price, dtype=np.float64)
            pnl = terminal_vals - entry_price
            return self._compute_stats(pnl, terminal_vals)

        # --- AIV scaling ---
        aiv_is_10d = bool(getattr(self.cfg, "aiv_is_10day_change", False))
        aiv_h = aiv * (steps / 10.0) if aiv_is_10d else aiv

        # --- 1) Underlying paths ---
        Z = np.random.standard_normal((self.num_paths, steps))
        drift_step = (mu - 0.5 * sigma ** 2) * dt
        diff_step = sigma * np.sqrt(dt) * Z
        log_ret_acc = np.cumsum(drift_step + diff_step, axis=1)
        S_paths = S0 * np.exp(log_ret_acc)

        # --- 2) IV paths: linear drift + RANDOM WALK noise ---
        t_idx = np.arange(1, steps + 1, dtype=np.float64)  # 1..steps
        iv_base = IV0 + aiv_h * (t_idx / steps)  # (steps,)
        iv_daily_std = float(getattr(self.cfg, "iv_daily_std", 0.01))

        iv_shocks = np.random.normal(0.0, iv_daily_std, (self.num_paths, steps))
        iv_cum_noise = np.cumsum(iv_shocks, axis=1)
        IV_paths = iv_base[None, :] + iv_cum_noise

        # Apply min/cap
        iv_min = float(getattr(self.cfg, "iv_min", 0.01))
        iv_max = getattr(self.cfg, "iv_max", 3.0)  # may be None
        if iv_max is None:
            IV_paths = np.maximum(iv_min, IV_paths)
        else:
            IV_paths = np.clip(IV_paths, iv_min, float(iv_max))

        # --- 3) Option prices for all steps ---
        T_rem_vec = np.maximum(T_exp - (t_idx * dt), 0.0)  # (steps,)
        Opt_paths = self._black_scholes_matrix(S_paths, K, r, T_rem_vec, IV_paths, is_call)

        # --- 4) Exits (vectorized scan) + optional diagnostics ---
        ratios = Opt_paths / entry_price
        final_values = Opt_paths[:, -1].copy()

        active = np.ones(self.num_paths, dtype=bool)
        be_armed = np.zeros(self.num_paths, dtype=bool)

        # Diagnostics
        exit_day = np.full(self.num_paths, -1, dtype=np.int16)  # -1 means held to end
        exit_reason = np.zeros(self.num_paths, dtype=np.int8)  # 0=hold, 1=TP, 2=SL, 3=BE

        tp_level = 1.0 + float(self.cfg.take_profit_pct)
        sl_level = 1.0 - float(self.cfg.stop_loss_pct)
        be_arm_level = 1.0 + float(self.cfg.be_trigger_pct)
        be_exit_level = 1.0 + float(self.cfg.be_exit_buffer)

        for t in range(steps):
            if not active.any():
                break

            r_t = ratios[:, t]
            p_t = Opt_paths[:, t]

            # A) Take profit
            tp_hit = (r_t >= tp_level) & active
            if tp_hit.any():
                final_values[tp_hit] = p_t[tp_hit]
                active[tp_hit] = False
                exit_day[tp_hit] = t + 1
                exit_reason[tp_hit] = 1

            # B) Stop loss
            sl_hit = (r_t <= sl_level) & active
            if sl_hit.any():
                final_values[sl_hit] = p_t[sl_hit]
                active[sl_hit] = False
                exit_day[sl_hit] = t + 1
                exit_reason[sl_hit] = 2

            # C) Arm break-even lock
            arm = (r_t >= be_arm_level) & active
            if arm.any():
                be_armed[arm] = True

            # D) Break-even exit
            be_exit = (r_t <= be_exit_level) & be_armed & active
            if be_exit.any():
                final_values[be_exit] = p_t[be_exit]
                active[be_exit] = False
                exit_day[be_exit] = t + 1
                exit_reason[be_exit] = 3

        pnl = final_values - entry_price
        stats = self._compute_stats(pnl, final_values)

        # Add diagnostics stats
        stats["exit_day_mean"] = float(np.mean(np.where(exit_day < 0, steps, exit_day)))
        stats["exit_rate_tp"] = float(np.mean(exit_reason == 1))
        stats["exit_rate_sl"] = float(np.mean(exit_reason == 2))
        stats["exit_rate_be"] = float(np.mean(exit_reason == 3))
        stats["hold_rate"] = float(np.mean(exit_reason == 0))

        return stats

    # -------------------------------
    # 2. TERMINAL-ONLY SIMULATION (UPDATED CONSISTENCY)
    # -------------------------------
    def _simulate_terminal_only(self, fusion_output, market_state):
        """
        Terminal-only simulation.
        Updated to use cfg.iv_daily_std so it matches path physics.
        """
        mu = float(fusion_output["mu"])
        sigma = float(fusion_output["sigma"])
        aiv = float(fusion_output["aiv"])

        S0 = float(market_state["S"])
        IV0 = float(market_state["IV"])
        T_exp = float(market_state["T"])
        r = float(market_state["r"])
        K = float(market_state["K"])
        is_call = bool(market_state["is_call"])

        entry_price = float(market_state.get("price", 0.0))
        if entry_price <= 0:
            entry_price = self._black_scholes_single(S0, K, r, T_exp, IV0, is_call)

        year_days = float(getattr(self.cfg, "year_days", 252.0))
        dt = 1.0 / year_days

        steps = int(self.cfg.hold_days)
        if steps <= 0:
            pnl = np.zeros(self.num_paths, dtype=np.float64)
            terminal_vals = np.full(self.num_paths, entry_price, dtype=np.float64)
            return self._compute_stats(pnl, terminal_vals)

        # Expiry Safety
        if T_exp <= 0.0:
            terminal_vals = np.full(self.num_paths, self._black_scholes_single(S0, K, r, 0.0, IV0, is_call),
                                    dtype=np.float64)
            pnl = terminal_vals - entry_price
            return self._compute_stats(pnl, terminal_vals)

        max_steps = int(np.floor(T_exp / dt))
        if max_steps <= 0:
            terminal_val = self._black_scholes_single(S0, K, r, T_exp, IV0, is_call)
            terminal_vals = np.full(self.num_paths, terminal_val, dtype=np.float64)
            pnl = terminal_vals - entry_price
            return self._compute_stats(pnl, terminal_vals)

        steps_eff = min(steps, max_steps)
        hold_T_eff = steps_eff * dt
        T_rem = max(T_exp - hold_T_eff, 0.0)

        # Underlying at horizon
        Z = np.random.standard_normal(self.num_paths)
        drift = (mu - 0.5 * sigma ** 2) * hold_T_eff
        diffusion = sigma * np.sqrt(hold_T_eff) * Z
        S_h = S0 * np.exp(drift + diffusion)

        # AIV Scaling
        aiv_is_10d = bool(getattr(self.cfg, "aiv_is_10day_change", False))
        aiv_h = aiv * (steps_eff / 10.0) if aiv_is_10d else aiv

        # IV Horizon Noise (Scaled by sqrt(steps) to match Random Walk)
        iv_daily_std = float(getattr(self.cfg, "iv_daily_std", 0.01))
        iv_noise_h = np.random.normal(0.0, iv_daily_std * np.sqrt(steps_eff), self.num_paths)

        IV_h = IV0 + aiv_h + iv_noise_h

        # Apply min/cap
        iv_min = float(getattr(self.cfg, "iv_min", 0.01))
        iv_max = getattr(self.cfg, "iv_max", 3.0)
        if iv_max is None:
            IV_h = np.maximum(iv_min, IV_h)
        else:
            IV_h = np.clip(IV_h, iv_min, float(iv_max))

        # Price
        terminal_vals = self._black_scholes_vectorized(S_h, K, r, T_rem, IV_h, is_call)
        pnl = terminal_vals - entry_price

        return self._compute_stats(pnl, terminal_vals)

    # -------------------------------
    # 3. MATH HELPERS
    # -------------------------------
    def _black_scholes_matrix(self, S, K, r, T_vec, sigma, is_call):
        """Matrix pricing: S (N, steps), T_vec (steps,)"""
        T = T_vec[None, :]  # broadcast
        sigma = np.maximum(sigma, 1e-8)

        intrinsic = np.maximum(0.0, (S - K) if is_call else (K - S))
        near0 = (T <= 1e-8)

        with np.errstate(divide="ignore", invalid="ignore"):
            sqrtT = np.sqrt(np.maximum(T, 1e-16))
            d1 = (np.log(np.maximum(S, 1e-16) / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
            d2 = d1 - sigma * sqrtT

        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        price = np.where(near0, intrinsic, price)
        return price

    def _black_scholes_vectorized(self, S, K, r, T, sigma, is_call):
        """Vectorized pricing: scalar T or N-vector T"""
        if np.isscalar(T) and T <= 1e-8:
            payoff = (S - K) if is_call else (K - S)
            return np.maximum(0.0, payoff)

        sigma = np.maximum(sigma, 1e-8)

        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

        if not np.isscalar(T):
            d1 = np.where(T > 1e-8, d1, 0.0)
            d2 = np.where(T > 1e-8, d2, 0.0)

        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        if not np.isscalar(T):
            intrinsic = np.maximum(0.0, (S - K) if is_call else (K - S))
            price = np.where(T <= 1e-8, intrinsic, price)

        return price

    def _black_scholes_single(self, S, K, r, T, sigma, is_call):
        """Scalar pricing for entry checks"""
        if T <= 1e-8 or sigma <= 1e-8:
            payoff = (S - K) if is_call else (K - S)
            return max(0.0, payoff)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _compute_stats(self, pnl, terminal_vals):
        stats = {
            "expected_option_price": float(np.mean(terminal_vals)),
            "expected_pnl": float(np.mean(pnl)),
            "p10_value": float(np.percentile(terminal_vals, 10)),
            "p90_value": float(np.percentile(terminal_vals, 90)),
            "prob_profit": float(np.mean(pnl > 0)),
            "VaR_95": float(np.percentile(pnl, 5)),
        }
        stats.update(self._calculate_sharpe_metrics(pnl))
        return stats

    def _calculate_sharpe_metrics(self, pnl_array):
        mean_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array)
        if std_pnl < 1e-6:
            return {"mc_sharpe": 0.0, "downside_sharpe": 0.0}

        mc_sharpe = mean_pnl / std_pnl
        negative = pnl_array[pnl_array < 0]
        downside_dev = np.sqrt(np.mean(negative ** 2)) if len(negative) else 1e-6
        downside_sharpe = mean_pnl / downside_dev
        return {"mc_sharpe": float(mc_sharpe), "downside_sharpe": float(downside_sharpe)}
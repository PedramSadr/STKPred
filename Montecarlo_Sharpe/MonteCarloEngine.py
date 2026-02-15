import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class ExitRules:
    use_path_exits: bool = False
    hold_days: int = 14

    # Exits
    take_profit_pct: float = 0.50
    stop_loss_pct: float = 0.30
    be_trigger_pct: float = 0.20
    be_exit_buffer: float = 0.00

    # Friction
    slippage_pct: float = 0.01  # 1% slippage on exit

    # Time/Vol
    year_days: float = 252.0
    aiv_is_10day_change: bool = True

    # IV Physics
    iv_daily_std: float = 0.01
    iv_min: float = 0.01
    iv_max: float = 3.00


class MonteCarloEngine:
    def __init__(self, num_paths=50000, exit_config: ExitRules = None):
        self.num_paths = num_paths
        self.cfg = exit_config if exit_config else ExitRules()

    def generate_risk_metrics(self, fusion_output, market_state):
        if self.cfg.use_path_exits:
            return self._simulate_path_exits(fusion_output, market_state)
        else:
            return self._simulate_terminal_only(fusion_output, market_state)

    # ----------------------------------------------------------------
    # 1. PATH-EXIT SIMULATION (Active Management)
    # ----------------------------------------------------------------
    def _simulate_path_exits(self, fusion_output, market_state):
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
            return self._compute_stats(np.zeros(self.num_paths), np.full(self.num_paths, entry_price))

        year_days = float(getattr(self.cfg, "year_days", 252.0))
        dt = 1.0 / year_days

        # Expiry Safety
        if T_exp <= 0.0:
            terminal_val = self._black_scholes_single(S0, K, r, 0.0, IV0, is_call)
            return self._compute_stats(np.full(self.num_paths, terminal_val - entry_price),
                                       np.full(self.num_paths, terminal_val))

        max_steps = int(np.floor(T_exp / dt))
        steps = min(steps, max_steps) if max_steps > 0 else 0
        if steps <= 0:
            return self._compute_stats(np.zeros(self.num_paths), np.full(self.num_paths, entry_price))

        # --- AIV scaling ---
        aiv_is_10d = bool(getattr(self.cfg, "aiv_is_10day_change", False))
        aiv_h = aiv * (steps / 10.0) if aiv_is_10d else aiv

        # --- 1) Underlying paths ---
        Z = np.random.standard_normal((self.num_paths, steps))
        drift_step = (mu - 0.5 * sigma ** 2) * dt
        diff_step = sigma * np.sqrt(dt) * Z
        log_ret_acc = np.cumsum(drift_step + diff_step, axis=1)
        S_paths = S0 * np.exp(log_ret_acc)

        # --- 2) IV paths ---
        t_idx = np.arange(1, steps + 1, dtype=np.float64)
        iv_base = IV0 + aiv_h * (t_idx / steps)
        iv_daily_std = float(getattr(self.cfg, "iv_daily_std", 0.01))
        iv_shocks = np.random.normal(0.0, iv_daily_std, (self.num_paths, steps))
        iv_cum_noise = np.cumsum(iv_shocks, axis=1)
        IV_paths = iv_base[None, :] + iv_cum_noise

        iv_min = float(getattr(self.cfg, "iv_min", 0.01))
        iv_max = getattr(self.cfg, "iv_max", 3.0)
        if iv_max is None:
            IV_paths = np.maximum(iv_min, IV_paths)
        else:
            IV_paths = np.clip(IV_paths, iv_min, float(iv_max))

        # --- 3) Option Pricing ---
        T_rem_vec = np.maximum(T_exp - (t_idx * dt), 0.0)
        Opt_paths = self._black_scholes_matrix(S_paths, K, r, T_rem_vec, IV_paths, is_call)

        # --- EXCURSIONS (MAE / MFE) ---
        # Calculate max/min excursions along the paths BEFORE exit evaluation
        pnl_paths = Opt_paths - entry_price
        cum_max_pnl = np.maximum.accumulate(pnl_paths, axis=1)
        cum_min_pnl = np.minimum.accumulate(pnl_paths, axis=1)

        # --- 4) Exits with SLIPPAGE ---
        ratios = Opt_paths / entry_price
        final_values = Opt_paths[:, -1].copy()

        active = np.ones(self.num_paths, dtype=bool)
        be_armed = np.zeros(self.num_paths, dtype=bool)

        exit_day = np.full(self.num_paths, -1, dtype=np.int16)
        exit_reason = np.zeros(self.num_paths, dtype=np.int8)

        tp_level = 1.0 + float(self.cfg.take_profit_pct)
        sl_level = 1.0 - float(self.cfg.stop_loss_pct)
        be_arm_level = 1.0 + float(self.cfg.be_trigger_pct)
        be_exit_level = 1.0 + float(self.cfg.be_exit_buffer)

        # Slippage Multiplier (1.0 - 0.01 = 0.99)
        fill_mult = 1.0 - float(self.cfg.slippage_pct)

        for t in range(steps):
            if not active.any():
                break

            r_t = ratios[:, t]
            p_t = Opt_paths[:, t]

            # TP Exit
            tp_hit = (r_t >= tp_level) & active
            if tp_hit.any():
                final_values[tp_hit] = p_t[tp_hit] * fill_mult
                active[tp_hit] = False
                exit_day[tp_hit] = t + 1
                exit_reason[tp_hit] = 1

            # SL Exit
            sl_hit = (r_t <= sl_level) & active
            if sl_hit.any():
                final_values[sl_hit] = p_t[sl_hit] * fill_mult
                active[sl_hit] = False
                exit_day[sl_hit] = t + 1
                exit_reason[sl_hit] = 2

            # BE Arm
            arm = (r_t >= be_arm_level) & active
            if arm.any():
                be_armed[arm] = True

            # BE Exit
            be_exit = (r_t <= be_exit_level) & be_armed & active
            if be_exit.any():
                final_values[be_exit] = p_t[be_exit] * fill_mult
                active[be_exit] = False
                exit_day[be_exit] = t + 1
                exit_reason[be_exit] = 3

        # Capture MAE/MFE on the actual day the path exited
        exit_idx = np.where(exit_day > 0, exit_day - 1, steps - 1)
        path_indices = np.arange(self.num_paths)
        mae_arr = cum_min_pnl[path_indices, exit_idx]
        mfe_arr = cum_max_pnl[path_indices, exit_idx]

        final_values[active] = final_values[active] * fill_mult

        pnl = final_values - entry_price
        stats = self._compute_stats(pnl, final_values)

        # Diagnostics & New Metrics
        stats["exit_day_mean"] = float(np.mean(np.where(exit_day < 0, steps, exit_day)))
        stats["exit_rate_tp"] = float(np.mean(exit_reason == 1))
        stats["exit_rate_sl"] = float(np.mean(exit_reason == 2))
        stats["exit_rate_be"] = float(np.mean(exit_reason == 3))
        stats["hold_rate"] = float(np.mean(exit_reason == 0))

        # Inject Mean MAE / MFE into the output
        stats["mae"] = float(np.mean(mae_arr))
        stats["mfe"] = float(np.mean(mfe_arr))

        return stats

    # ----------------------------------------------------------------
    # 2. TERMINAL-ONLY SIMULATION (Fast Triage)
    # ----------------------------------------------------------------
    def _simulate_terminal_only(self, fusion_output, market_state):
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
            return self._compute_stats(np.zeros(self.num_paths), np.full(self.num_paths, entry_price))

        max_steps = int(np.floor(T_exp / dt))
        steps_eff = min(steps, max_steps) if max_steps > 0 else 0
        if steps_eff <= 0:
            terminal_val = self._black_scholes_single(S0, K, r, max(T_exp, 0.0), IV0, is_call)
            return self._compute_stats(np.full(self.num_paths, terminal_val - entry_price),
                                       np.full(self.num_paths, terminal_val))

        hold_T_eff = steps_eff * dt
        T_rem = max(T_exp - hold_T_eff, 0.0)

        Z = np.random.standard_normal(self.num_paths)
        drift = (mu - 0.5 * sigma ** 2) * hold_T_eff
        diffusion = sigma * np.sqrt(hold_T_eff) * Z
        S_h = S0 * np.exp(drift + diffusion)

        aiv_is_10d = bool(getattr(self.cfg, "aiv_is_10day_change", False))
        aiv_h = aiv * (steps_eff / 10.0) if aiv_is_10d else aiv

        iv_daily_std = float(getattr(self.cfg, "iv_daily_std", 0.01))
        iv_noise_h = np.random.normal(0.0, iv_daily_std * np.sqrt(steps_eff), self.num_paths)

        IV_h = IV0 + aiv_h + iv_noise_h

        iv_min = float(getattr(self.cfg, "iv_min", 0.01))
        iv_max = getattr(self.cfg, "iv_max", 3.0)
        if iv_max is None:
            IV_h = np.maximum(iv_min, IV_h)
        else:
            IV_h = np.clip(IV_h, iv_min, float(iv_max))

        terminal_vals = self._black_scholes_vectorized(S_h, K, r, T_rem, IV_h, is_call)

        fill_mult = 1.0 - float(self.cfg.slippage_pct)
        terminal_vals = terminal_vals * fill_mult

        pnl = terminal_vals - entry_price
        stats = self._compute_stats(pnl, terminal_vals)

        # Fallback MAE/MFE mapping for terminal-only simulations
        stats["mae"] = float(np.mean(np.minimum(pnl, 0.0)))
        stats["mfe"] = float(np.mean(np.maximum(pnl, 0.0)))

        return stats

    # ----------------------------------------------------------------
    # 3. STATS & MATH
    # ----------------------------------------------------------------
    def _compute_stats(self, pnl, terminal_vals):
        stats = {
            "expected_pnl": float(np.mean(pnl)),
            "expected_option_price": float(np.mean(terminal_vals)),
            "prob_profit": float(np.mean(pnl > 0)),
            "VaR_95": float(np.percentile(pnl, 5)),
            "p10_value": float(np.percentile(terminal_vals, 10)),
            "p90_value": float(np.percentile(terminal_vals, 90)),
        }

        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl)

        stats["mc_sharpe"] = float(mean_pnl / std_pnl) if std_pnl > 1e-6 else 0.0

        negative = pnl[pnl < 0]
        downside_dev = np.sqrt(np.mean(negative ** 2)) if len(negative) > 0 else 1e-6
        stats["downside_sharpe"] = float(mean_pnl / downside_dev)

        return stats

    def _black_scholes_matrix(self, S, K, r, T_vec, sigma, is_call):
        T = T_vec[None, :]
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
        return np.where(near0, intrinsic, price)

    def _black_scholes_vectorized(self, S, K, r, T, sigma, is_call):
        if np.isscalar(T) and T <= 1e-8:
            return np.maximum(0.0, (S - K) if is_call else (K - S))
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
            price = np.where(T <= 1e-8, np.maximum(0.0, (S - K) if is_call else (K - S)), price)
        return price

    def _black_scholes_single(self, S, K, r, T, sigma, is_call):
        if T <= 1e-8: return max(0.0, (S - K) if is_call else (K - S))
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call: return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
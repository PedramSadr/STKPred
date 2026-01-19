import pandas as pd
import logging
from math import floor
from pathlib import Path


class PortfolioRiskManager:
    """
    4-Layer Risk Framework (Production Release)

    Safety Features:
    - Blocks trading if Open Positions have 0.0 MaxLoss or CVaR (Corruption Check).
    - Guardrails against negative or fractional quantities in CSV.
    - Strict expiration parsing with NaT checks.
    """

    def __init__(self, config, open_positions_path):
        self.cfg = config
        self.positions_path = open_positions_path

        # --- CONFIGURATION ---
        self.RISK_PER_TRADE_PCT = getattr(config, 'RISK_PER_TRADE_PCT', 0.005)
        self.PORTFOLIO_MAX_LOSS_PCT = getattr(config, 'PORTFOLIO_MAX_LOSS_PCT', 0.02)
        self.PORTFOLIO_CVAR_PCT = getattr(config, 'PORTFOLIO_CVAR_PCT', 0.01)

        # Exposure Caps
        self.MAX_POS_TOTAL = getattr(config, 'MAX_POS_TOTAL', 3)
        self.MAX_POS_PER_SYMBOL = getattr(config, 'MAX_POS_PER_SYMBOL', 1)
        self.MAX_RISK_PER_FACTOR_PCT = getattr(config, 'MAX_RISK_PER_FACTOR_PCT', 0.015)
        self.MAX_RISK_PER_EXPIRY_PCT = getattr(config, 'MAX_RISK_PER_EXPIRY_PCT', 0.015)
        self.MAX_RISK_UNMAPPED_PCT = getattr(config, 'MAX_RISK_UNMAPPED_PCT', 0.005)

        # Factor Map
        self.FACTOR_MAP = {
            'TSLA': 'QQQ', 'NVDA': 'QQQ', 'AMD': 'QQQ', 'MSFT': 'QQQ',
            'SPY': 'SPY', 'IWM': 'RUT'
        }

    def _load_portfolio_state(self):
        """
        Loads current risk state. Returns None if state is corrupted (Blocks Trading).
        """
        if not Path(self.positions_path).exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.positions_path)

            # Filter for OPEN positions
            if 'status' in df.columns:
                df = df[df['status'] == 'OPEN'].copy()

            if df.empty: return df

            # 1. Check Missing Columns FIRST
            cols_to_numeric = ['qty', 'max_loss_per_contract', 'cvar_per_contract']
            missing = [c for c in cols_to_numeric if c not in df.columns]

            if missing:
                logging.error(f"CRITICAL: Positions file missing columns {missing}. Blocking trading.")
                return None

                # 2. Coerce Numeric Types
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # [FIX] Block on Zero Risk (MaxLoss OR CVaR) - Prevents phantom leverage
            if (df['max_loss_per_contract'] <= 0).any():
                logging.error("CRITICAL: Found OPEN positions with <= 0.0 Max Loss. CSV Corrupted. Blocking trading.")
                return None

            if (df['cvar_per_contract'] <= 0).any():
                logging.error("CRITICAL: Found OPEN positions with <= 0.0 CVaR. CSV Corrupted. Blocking trading.")
                return None

            # [FIX] Qty Guardrail: Clip negative numbers, enforce int
            if (df['qty'] < 0).any():
                logging.warning("CORRUPTION: Found negative quantity in positions. Clipping to 0.")
            df['qty'] = df['qty'].clip(lower=0).astype(int)

            # Enrich with Meta-Data
            df['factor'] = df['symbol'].map(self.FACTOR_MAP).fillna('Unmapped')

            # Safe Expiry Parsing
            df['expiry_dt'] = pd.to_datetime(df['expiration'], errors='coerce')
            if df['expiry_dt'].isna().any():
                logging.warning(f"CORRUPTION: Found {df['expiry_dt'].isna().sum()} invalid Expirations.")

            df['expiry_bucket'] = df['expiry_dt'].dt.to_period('M')

            return df

        except Exception as e:
            logging.exception(f"CRITICAL: Failed to load portfolio state: {e}")
            return None

    def allocate(self, candidate, equity=10000.0):
        """
        Calculates safe qty. Returns: (final_qty, details_dict)
        """
        # --- 1. NORMALIZE INPUTS ---
        symbol = candidate.get('symbol', 'TSLA')
        factor = self.FACTOR_MAP.get(symbol, 'Unmapped')
        economics = candidate.get('economics', {})

        max_loss_per_share = economics.get('max_loss', 0.0)
        raw_cvar_per_share = abs(candidate.get('CVaR95', 0.0))

        # Enhanced Unit Guardrail
        cvar_raw_val = raw_cvar_per_share * 100

        if raw_cvar_per_share > (max_loss_per_share * 1.01):
            logging.warning(
                f"Unit Guardrail: Capping CVaR ${raw_cvar_per_share:.2f} to MaxLoss ${max_loss_per_share:.2f}")
            raw_cvar_per_share = max_loss_per_share
            cvar_note = "Capped (MaxLoss)"
        else:
            cvar_note = "MC Calc"

        # Scale to Contract
        max_loss_per_contract = max_loss_per_share * 100
        cvar_per_contract = raw_cvar_per_share * 100

        if max_loss_per_contract <= 0: return self._block_response("Invalid Max Loss")

        # [FIX] Strict Expiry Parsing
        try:
            legs = candidate.get('legs', [])
            # Find first leg with a valid expiration string
            exp_str = next((l.get('expiration') for l in legs if l.get('expiration')), None)

            if not exp_str: raise ValueError("No valid expiration found in legs")

            cand_expiry_dt = pd.to_datetime(exp_str, errors='coerce')
            if pd.isna(cand_expiry_dt): raise ValueError(f"Invalid date format: {exp_str}")

            cand_bucket = cand_expiry_dt.to_period('M')
        except Exception as e:
            return self._block_response(f"Expiry Parse Error: {e}")

        # --- 2. LOAD STATE ---
        df_pos = self._load_portfolio_state()
        if df_pos is None: return self._block_response("Portfolio State Failure")

        # Aggregate Current Risk
        curr_max_loss_total = 0.0
        curr_cvar_total = 0.0
        curr_factor_risk = 0.0
        curr_expiry_risk = 0.0

        if not df_pos.empty:
            curr_max_loss_total = (df_pos['max_loss_per_contract'] * df_pos['qty']).sum()
            curr_cvar_total = (df_pos['cvar_per_contract'] * df_pos['qty']).sum()

            f_mask = df_pos['factor'] == factor
            curr_factor_risk = (df_pos.loc[f_mask, 'cvar_per_contract'] * df_pos.loc[f_mask, 'qty']).sum()

            e_mask = df_pos['expiry_bucket'] == cand_bucket
            curr_expiry_risk = (df_pos.loc[e_mask, 'cvar_per_contract'] * df_pos.loc[e_mask, 'qty']).sum()

        # --- LAYER 1: PER-TRADE HARD CAP ---
        budget_trade = equity * self.RISK_PER_TRADE_PCT
        qty_L1 = floor(budget_trade / max_loss_per_contract)

        # --- LAYER 2: PORTFOLIO MAX LOSS ---
        remaining_loss_budget = (equity * self.PORTFOLIO_MAX_LOSS_PCT) - curr_max_loss_total
        qty_L2 = floor(max(0, remaining_loss_budget) / max_loss_per_contract)

        # --- LAYER 3: TAIL RISK (CVaR) ---
        remaining_cvar_budget = (equity * self.PORTFOLIO_CVAR_PCT) - curr_cvar_total
        denom_cvar = cvar_per_contract if cvar_per_contract >= 1.0 else max_loss_per_contract
        qty_L3 = floor(max(0, remaining_cvar_budget) / denom_cvar)

        # --- LAYER 4: EXPOSURE CAPS ---
        # A. Total Positions
        if len(df_pos) >= self.MAX_POS_TOTAL:
            return self._block_response("Max Total Positions Reached")

        # B. Symbol Count
        symbol_count = len(df_pos[df_pos['symbol'] == symbol]) if not df_pos.empty else 0
        if symbol_count >= self.MAX_POS_PER_SYMBOL:
            return self._block_response(f"Max {symbol} Positions Reached")

        # C. Factor Risk Cap
        limit_pct = self.MAX_RISK_UNMAPPED_PCT if factor == 'Unmapped' else self.MAX_RISK_PER_FACTOR_PCT
        remaining_factor_budget = (equity * limit_pct) - curr_factor_risk
        qty_L4_Factor = floor(max(0, remaining_factor_budget) / denom_cvar)

        # D. Expiry Bucket Cap
        remaining_expiry_budget = (equity * self.MAX_RISK_PER_EXPIRY_PCT) - curr_expiry_risk
        qty_L4_Expiry = floor(max(0, remaining_expiry_budget) / denom_cvar)

        # --- FINAL AGGREGATION ---
        limits = {
            "L1_TradeCap": qty_L1,
            "L2_PortLoss": qty_L2,
            "L3_PortCVaR": qty_L3,
            "L4_Factor": qty_L4_Factor,
            "L4_Expiry": qty_L4_Expiry
        }

        # Deterministic Binding Constraint
        binding_key = min(limits, key=limits.get)
        final_qty = max(0, int(limits[binding_key]))

        snapshot = {
            "Total_MaxLoss": f"${curr_max_loss_total:.0f}/${(equity * self.PORTFOLIO_MAX_LOSS_PCT):.0f}",
            "Total_CVaR": f"${curr_cvar_total:.0f}/${(equity * self.PORTFOLIO_CVAR_PCT):.0f}",
            "Factor_Risk": f"${curr_factor_risk:.0f} ({factor})",
            "Expiry_Risk": f"${curr_expiry_risk:.0f} ({cand_bucket})"
        }

        details = {
            "qty": final_qty,
            "blocked": (final_qty == 0),
            "blocked_by": binding_key,
            "limits": limits,
            "snapshot": snapshot,
            "CVaR_Used_Per_Contract": cvar_per_contract,
            "CVaR_Raw_Per_Contract": cvar_raw_val,
            "CVaR_Note": cvar_note,
            "Factor_Used": factor
        }

        return final_qty, details

    def _block_response(self, reason):
        return 0, {
            "qty": 0,
            "blocked": True,
            "blocked_by": "Safety_Check",
            "reason": reason
        }
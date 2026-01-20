import pandas as pd
import logging
from math import floor
from pathlib import Path


class PortfolioRiskManager:
    """
    4-Layer Risk Framework (Golden Master - Polished)
    Safety: Blocks trading on corruption, zero-risk data, or schema drift.
    """

    def __init__(self, config, open_positions_path):
        self.cfg = config
        self.positions_path = open_positions_path

        # Config Defaults
        self.RISK_PER_TRADE_PCT = getattr(config, 'RISK_PER_TRADE_PCT', 0.005)
        self.PORTFOLIO_MAX_LOSS_PCT = getattr(config, 'PORTFOLIO_MAX_LOSS_PCT', 0.02)
        self.PORTFOLIO_CVAR_PCT = getattr(config, 'PORTFOLIO_CVAR_PCT', 0.01)
        self.MAX_POS_TOTAL = getattr(config, 'MAX_POS_TOTAL', 3)
        self.MAX_POS_PER_SYMBOL = getattr(config, 'MAX_POS_PER_SYMBOL', 1)
        self.MAX_RISK_PER_FACTOR_PCT = getattr(config, 'MAX_RISK_PER_FACTOR_PCT', 0.015)
        self.MAX_RISK_PER_EXPIRY_PCT = getattr(config, 'MAX_RISK_PER_EXPIRY_PCT', 0.015)
        self.MAX_RISK_UNMAPPED_PCT = getattr(config, 'MAX_RISK_UNMAPPED_PCT', 0.005)

        self.FACTOR_MAP = {'TSLA': 'QQQ', 'NVDA': 'QQQ', 'AMD': 'QQQ', 'MSFT': 'QQQ', 'SPY': 'SPY'}

    def _load_portfolio_state(self):
        if not Path(self.positions_path).exists(): return pd.DataFrame()

        try:
            df = pd.read_csv(self.positions_path)
            if 'status' in df.columns: df = df[df['status'] == 'OPEN'].copy()
            if df.empty: return df

            # 1. Missing Columns Check (Fail Fast)
            cols_to_numeric = ['qty', 'max_loss_per_contract', 'cvar_per_contract']
            missing = [c for c in cols_to_numeric if c not in df.columns]
            if missing:
                logging.error(f"CRITICAL: Positions missing {missing}. Blocking.")
                return None

                # 2. Coerce & Corruption Check
            # Un-indented so it runs when schema is VALID
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # [SAFETY] Block on Zero Risk (MaxLoss OR CVaR)
            if (df['max_loss_per_contract'] <= 0).any():
                logging.error("CRITICAL: OPEN positions with <= 0.0 Max Loss. Blocking.")
                return None

            if (df['cvar_per_contract'] <= 0).any():
                logging.error("CRITICAL: OPEN positions with <= 0.0 CVaR. Blocking.")
                return None

            df['qty'] = df['qty'].clip(lower=0).astype(int)
            df['factor'] = df['symbol'].map(self.FACTOR_MAP).fillna('Unmapped')

            df['expiry_dt'] = pd.to_datetime(df['expiration'], errors='coerce')
            df['expiry_bucket'] = df['expiry_dt'].dt.to_period('M')

            return df

        except pd.errors.EmptyDataError:
            # Handle empty CSV gracefully (Day 1 scenario)
            return pd.DataFrame()

        except Exception as e:
            logging.exception(f"CRITICAL: Portfolio Load Failed: {e}")
            return None

    def allocate(self, candidate, equity=10000.0):
        # 1. Normalize Inputs
        symbol = candidate.get('symbol', 'TSLA')
        factor = self.FACTOR_MAP.get(symbol, 'Unmapped')
        economics = candidate.get('economics', {})

        max_loss_per_share = economics.get('max_loss', 0.0)
        raw_cvar_per_share = abs(candidate.get('CVaR95', 0.0))

        # Unit Guardrail
        if raw_cvar_per_share > (max_loss_per_share * 1.01):
            raw_cvar_per_share = max_loss_per_share

        max_loss_per_contract = max_loss_per_share * 100
        cvar_per_contract = raw_cvar_per_share * 100

        if max_loss_per_contract <= 0: return self._block_response("Invalid Max Loss")

        # Expiry Parse
        try:
            legs = candidate.get('legs', [])
            exp_str = next((l.get('expiration') for l in legs if l.get('expiration')), None)
            if not exp_str: raise ValueError("No Expiration")
            cand_bucket = pd.to_datetime(exp_str).to_period('M')
        except Exception as e:
            return self._block_response(f"Expiry Error: {e}")

        # 2. Load State
        df_pos = self._load_portfolio_state()
        if df_pos is None: return self._block_response("Portfolio State Failure")

        # Aggregate Risk
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

        # 3. Layer Calculations
        qty_L1 = floor((equity * self.RISK_PER_TRADE_PCT) / max_loss_per_contract)
        qty_L2 = floor(max(0, (equity * self.PORTFOLIO_MAX_LOSS_PCT) - curr_max_loss_total) / max_loss_per_contract)

        denom_cvar = cvar_per_contract if cvar_per_contract >= 1.0 else max_loss_per_contract
        qty_L3 = floor(max(0, (equity * self.PORTFOLIO_CVAR_PCT) - curr_cvar_total) / denom_cvar)

        # Exposure
        if len(df_pos) >= self.MAX_POS_TOTAL: return self._block_response("Max Total Pos")
        if not df_pos.empty and len(df_pos[df_pos['symbol'] == symbol]) >= self.MAX_POS_PER_SYMBOL:
            return self._block_response(f"Max {symbol} Pos")

        limit_pct = self.MAX_RISK_UNMAPPED_PCT if factor == 'Unmapped' else self.MAX_RISK_PER_FACTOR_PCT
        qty_L4_Factor = floor(max(0, (equity * limit_pct) - curr_factor_risk) / denom_cvar)
        qty_L4_Expiry = floor(max(0, (equity * self.MAX_RISK_PER_EXPIRY_PCT) - curr_expiry_risk) / denom_cvar)

        # 4. Result
        limits = {"L1": qty_L1, "L2": qty_L2, "L3": qty_L3, "L4_F": qty_L4_Factor, "L4_E": qty_L4_Expiry}
        binding_key = min(limits, key=limits.get)
        final_qty = max(0, int(limits[binding_key]))

        details = {"qty": final_qty, "blocked_by": binding_key, "limits": limits}
        return final_qty, details

    def _block_response(self, reason):
        return 0, {"qty": 0, "blocked_by": "Safety", "reason": reason}
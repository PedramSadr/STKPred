import pandas as pd
import os
import numpy as np
from dataclasses import dataclass
from typing import TypedDict, List, Literal, Optional


# --- CANONICAL SCHEMA DEFINITION ---
class OptionLeg(TypedDict):
    row_id: int         # <--- NEW: Traceability ID
    expiration: str
    strike: float
    type: Literal['call', 'put']
    side: Literal['long', 'short']
    dte: int
    iv: float
    entry_price: float  # Canonical Price (opt_price > mid)
    bid: float
    ask: float
    delta: float
    vega: float


class CandidateTrade(TypedDict):
    date: str
    structure: str
    type: str
    underlying_price: float
    legs: List[OptionLeg]


# 1️⃣ CONSTANT: Canonical IV Column Name
IV_COL = "implied_volatility"


@dataclass
class GeneratorConfig:
    min_dte: int = 7
    max_dte: int = 45
    min_vol: int = 1
    min_oi: int = 1
    max_candidates: int = 30


class CandidateGenerator:
    DEFAULT_PATH = r"C:\My Documents\Mics\Logs\TSLA_Options_Contracts.csv"

    # SAFETY: Disable spreads until Monte Carlo supports multi-leg valuation
    ENABLE_SPREADS = False

    def __init__(self, config: GeneratorConfig = None):
        self.cfg = config if config else GeneratorConfig()

    def load_catalog(self, file_path=None):
        path = file_path if file_path else self.DEFAULT_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Catalog not found at: {path}")
        df = pd.read_csv(path)
        for col in ['date', 'expiration']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    def generate(self, catalog_df: pd.DataFrame, trade_date: str = None) -> List[CandidateTrade]:
        # 1. Date Filtering
        if trade_date:
            target_date = pd.to_datetime(trade_date).date()
            df = catalog_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.date == target_date].copy()
        else:
            df = catalog_df.copy()

        if df.empty:
            print("DEBUG: DataFrame is empty after date filter.")
            return []

        # 2. Strict IV & Numeric Coercion
        if IV_COL not in df.columns:
            if 'iv' in df.columns:
                df = df.rename(columns={'iv': IV_COL})
            else:
                raise RuntimeError(f"CRITICAL: Missing required column '{IV_COL}'.")

        df[IV_COL] = pd.to_numeric(df[IV_COL], errors='coerce')

        # IV Sanity Cap: Match validator (0 < IV < 3.0)
        df = df[(df[IV_COL] > 0) & (df[IV_COL] < 3.0)]

        # Core Columns Handling
        core_cols = ['dte', 'volume', 'open_interest', 'strike', 'underlying_price', 'opt_price', 'bid', 'ask']
        for col in core_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with missing core data (including opt_price)
        cols_to_check = ['dte', 'strike', 'underlying_price', 'opt_price']
        df = df.dropna(subset=[c for c in cols_to_check if c in df.columns])

        # Fill Greeks
        for col in ['delta', 'vega', 'gamma']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Apply Constraints
        mask = (
                (df['dte'] >= self.cfg.min_dte) &
                (df['dte'] <= self.cfg.max_dte) &
                (df['volume'] >= self.cfg.min_vol) &
                (df['open_interest'] >= self.cfg.min_oi)
        )
        df = df[mask]
        print(f"DEBUG: {len(df)} rows remaining after filters.")

        candidates: List[CandidateTrade] = []

        # 3. Generate Candidates per Expiration
        for expiry, group in df.groupby('expiration'):
            current_price = group['underlying_price'].iloc[0]

            # Normalize type casing
            group = group.copy()
            if 'type' in group.columns:
                group['type'] = group['type'].astype(str).str.lower()

            calls = group[group['type'] == 'call'].sort_values('strike')
            puts = group[group['type'] == 'put'].sort_values('strike')

            if calls.empty and puts.empty: continue

            # --- SINGLE LEGS (ATM) ---
            if not calls.empty:
                idx = (calls['strike'] - current_price).abs().argsort()[:1]
                atm_call = calls.iloc[idx]
                if not atm_call.empty:
                    self._try_add_candidate(candidates, "ATM Call", "Single", [atm_call.iloc[0]], current_price)

            if not puts.empty:
                idx = (puts['strike'] - current_price).abs().argsort()[:1]
                atm_put = puts.iloc[idx]
                if not atm_put.empty:
                    self._try_add_candidate(candidates, "ATM Put", "Single", [atm_put.iloc[0]], current_price)

            # --- VERTICAL SPREADS (Gated) ---
            if self.ENABLE_SPREADS:
                # Bull Call Spread
                if len(calls) > 2:
                    idx_atm = (calls['strike'] - current_price).abs().argsort()[:1].values[0]
                    idx_short = idx_atm + 2
                    if idx_short < len(calls):
                        long_leg = calls.iloc[idx_atm]
                        short_leg = calls.iloc[idx_short]
                        if short_leg['strike'] > long_leg['strike']:
                            self._try_add_spread(candidates, "Bull Call Spread", "Vertical",
                                                 [long_leg, short_leg], ['long', 'short'], current_price)

                # Bear Put Spread
                if len(puts) > 2:
                    idx_atm = (puts['strike'] - current_price).abs().argsort()[:1].values[0]
                    idx_short = idx_atm - 2
                    if idx_short >= 0:
                        long_leg = puts.iloc[idx_atm]
                        short_leg = puts.iloc[idx_short]
                        if short_leg['strike'] < long_leg['strike']:
                            self._try_add_spread(candidates, "Bear Put Spread", "Vertical",
                                                 [long_leg, short_leg], ['long', 'short'], current_price)

        return candidates[:self.cfg.max_candidates]

    def _try_add_candidate(self, candidates_list, name, struct_type, legs, spot):
        try:
            # FIX: Capture raw date before cleaning
            raw_date = legs[0]['date']

            clean_legs = [self._clean_leg(legs[0], side='long')]

            # Pass raw_date explicitly
            cand = self._make_candidate(name, struct_type, clean_legs, spot, raw_date)
            self._validate_candidate(cand)
            candidates_list.append(cand)
        except ValueError:
            pass

    def _try_add_spread(self, candidates_list, name, struct_type, raw_legs, sides, spot):
        try:
            # FIX: Capture raw date
            raw_date = raw_legs[0]['date']

            clean_legs = []
            for raw_leg, side in zip(raw_legs, sides):
                clean_legs.append(self._clean_leg(raw_leg, side=side))

            cand = self._make_candidate(name, struct_type, clean_legs, spot, raw_date)
            self._validate_candidate(cand)
            candidates_list.append(cand)
        except ValueError:
            pass

    def _make_candidate(self, name, struct_type, legs, underlying_price, date_val) -> CandidateTrade:
        # FIX: Accept date_val argument instead of trying to read from clean legs
        if not isinstance(date_val, pd.Timestamp):
            date_val = pd.to_datetime(date_val)

        return {
            "date": date_val.strftime('%Y-%m-%d'),
            "structure": name,
            "type": struct_type,
            "underlying_price": float(underlying_price),
            "legs": legs
        }

    def _clean_leg(self, row, side: Literal['long', 'short'] = 'long') -> OptionLeg:
        # 1. Normalize IV
        iv = float(row[IV_COL])

        # 2. Canonical Entry Price
        if 'opt_price' in row and not pd.isna(row['opt_price']):
            entry = float(row['opt_price'])
        elif 'bid' in row and 'ask' in row and not pd.isna(row['bid']):
            entry = (float(row['bid']) + float(row['ask'])) / 2.0
        else:
            entry = 0.0  # Caught by validator

        if isinstance(row['expiration'], pd.Timestamp):
            exp_str = row['expiration'].strftime('%Y-%m-%d')
        else:
            exp_str = str(row['expiration'])

        # FIX: Capture Row ID (default to -1 if missing)
        row_id = int(row.get('row_id', -1))

        return {
            "row_id": row_id,  # <--- NEW
            "expiration": exp_str,
            "strike": float(row['strike']),
            "type": row['type'],
            "side": side,
            "dte": int(row['dte']),
            "iv": iv,
            "entry_price": entry,
            "bid": float(row.get('bid', 0)),
            "ask": float(row.get('ask', 0)),
            "delta": float(row.get('delta', 0)),
            "vega": float(row.get('vega', 0))
        }

    def _validate_candidate(self, cand: CandidateTrade):
        """Strict Runtime Validation."""
        if cand['underlying_price'] <= 0:
            raise ValueError(f"Invalid underlying price: {cand['underlying_price']}")

        if not cand['legs']:
            raise ValueError("Candidate has no legs")

        for leg in cand['legs']:
            if leg['iv'] <= 0 or leg['iv'] >= 3.0:
                raise ValueError(f"Invalid IV: {leg['iv']}")

            if leg['entry_price'] <= 0:
                raise ValueError(f"Invalid Entry Price: {leg['entry_price']}")

            if leg['dte'] <= 0:
                raise ValueError(f"Invalid DTE: {leg['dte']}")


if __name__ == "__main__":
    gen = CandidateGenerator()
    try:
        df = gen.load_catalog()
        latest_date = df['date'].max()
        results = gen.generate(df, trade_date=str(latest_date.date()))
        print(f"Generated {len(results)} Candidates.")
    except Exception as e:
        print(f"Error: {e}")
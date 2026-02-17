import pandas as pd
import os
import numpy as np
import logging
from dataclasses import dataclass
from typing import TypedDict, List, Literal, Optional

# --- LOGGING CONFIGURATION ---
LOG_DIR = r"C:\My Documents\Mics\Logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "candidate_generator.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


# --- CANONICAL SCHEMA DEFINITION ---
class OptionLeg(TypedDict):
    row_id: int
    expiration: str
    strike: float
    type: Literal['call', 'put']
    side: Literal['long', 'short']
    dte: int
    iv: float
    entry_price: float
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
    ENABLE_SPREADS = False

    def __init__(self, config: GeneratorConfig = None):
        self.cfg = config if config else GeneratorConfig()
        logging.info("CandidateGenerator initialized.")

    def load_catalog(self, file_path=None):
        path = file_path if file_path else self.DEFAULT_PATH
        logging.info(f"Loading catalog from: {path}")
        if not os.path.exists(path):
            logging.error(f"Catalog not found at: {path}")
            raise FileNotFoundError(f"Catalog not found at: {path}")

        df = pd.read_csv(path)
        for col in ['date', 'expiration']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        logging.info(f"Catalog loaded successfully with {len(df)} rows.")
        return df

    def generate(self, catalog_df: pd.DataFrame, trade_date: str = None) -> List[CandidateTrade]:
        logging.info(f"Starting candidate generation. Trade Date: {trade_date}")

        if trade_date:
            target_date = pd.to_datetime(trade_date).date()
            df = catalog_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'].dt.date == target_date].copy()
        else:
            df = catalog_df.copy()

        if df.empty:
            logging.warning("DataFrame is empty after date filter.")
            return []

        # --- FIX 1: Normalize Columns ---
        column_mapping = {
            'underlying': 'underlying_price',
            'spot': 'underlying_price',
            'current_price': 'underlying_price',
            'close': 'opt_price',
            'last': 'opt_price'
        }
        df = df.rename(columns=column_mapping)

        # --- FIX 2: Calculate Missing DTE ---
        if 'dte' not in df.columns:
            if 'expiration' in df.columns and 'date' in df.columns:
                df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['dte'] = (df['expiration'] - df['date']).dt.days
                df['dte'] = df['dte'].fillna(-1).astype(int)
            else:
                logging.warning("CRITICAL WARNING: 'dte' column missing and cannot be calculated.")
                df['dte'] = -1

        # --- FIX 3: Ensure Underlying Price Exists ---
        if 'underlying_price' not in df.columns:
            logging.warning("CRITICAL WARNING: 'underlying_price' column missing.")
            df['underlying_price'] = np.nan

        if IV_COL not in df.columns:
            if 'iv' in df.columns:
                df = df.rename(columns={'iv': IV_COL})
            else:
                logging.error(f"CRITICAL: Missing required column '{IV_COL}'.")
                return []

        df[IV_COL] = pd.to_numeric(df[IV_COL], errors='coerce')
        df = df[(df[IV_COL] > 0) & (df[IV_COL] < 3.0)]

        core_cols = ['dte', 'volume', 'open_interest', 'strike', 'underlying_price', 'opt_price', 'bid', 'ask']
        for col in core_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        cols_to_check = ['dte', 'strike', 'underlying_price', 'opt_price']
        missing_mask = df[cols_to_check].isna().any(axis=1)
        if missing_mask.any():
            df = df.dropna(subset=cols_to_check)

        for col in ['delta', 'vega', 'gamma']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # =========================================================
        # --- NEW STRUCTURAL & TIERED LIQUIDITY GATES ---
        # =========================================================

        # 1. QUOTE SANITY GATES
        df = df.dropna(subset=["bid", "ask", "underlying_price", "strike"])
        df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]

        # 2. ATM DISTANCE (Tightened to 10% to save compute)
        spot = df["underlying_price"]
        df["rel_distance"] = (df["strike"] - spot).abs() / spot
        df = df[df["rel_distance"] <= 0.10]

        # 3. NO-ARBITRAGE PRICE GATES
        spot = df["underlying_price"]  # <--- CRITICAL FIX: Re-sync spot size after filtering
        K = df["strike"]
        t = df["type"].astype(str).str.lower()
        is_call = t == "call"

        intrinsic = np.where(is_call, np.maximum(spot - K, 0), np.maximum(K - spot, 0))
        upper = np.where(is_call, spot, K)
        df = df[(df["ask"] >= 0.95 * intrinsic) & (df["ask"] <= 1.05 * upper)]

        # 4. TIERED LIQUIDITY GATES
        mid = (df["bid"] + df["ask"]) / 2.0
        spread_abs = df["ask"] - df["bid"]
        spread_pct = np.where(mid > 0, spread_abs / mid, 1.0)

        cond_t1_fail = (mid < 0.50) & (spread_abs > 0.02)
        cond_t2_fail = (mid >= 0.50) & (mid < 1.00) & ((spread_abs > 0.05) | (spread_pct > 0.15))
        cond_t3_fail = (mid >= 1.00) & ((spread_pct > 0.03) | (spread_abs > np.maximum(0.10, 0.02 * mid)))

        reject_mask = cond_t1_fail | cond_t2_fail | cond_t3_fail

        if reject_mask.any():
            reasons = np.select(
                [cond_t1_fail, cond_t2_fail, cond_t3_fail],
                ["T1_CHEAP_FAIL", "T2_NORMAL_FAIL", "T3_EXPENSIVE_FAIL"],
                default="OK"
            )
            reject_counts = pd.Series(reasons[reject_mask]).value_counts().to_dict()
            logging.info(f"Rejected {reject_mask.sum()} rows due to Tiered Liquidity. Breakdown: {reject_counts}")
            df = df[~reject_mask]

        # 5. CONDITIONAL VOLUME/OI
        if "open_interest" in df.columns:
            oi_nonzero = (df["open_interest"].fillna(0) > 0).mean()
            if oi_nonzero < 0.20:
                logging.warning("OI coverage < 20%; skipping OI filter today.")
                self.cfg.min_oi = 0

        if "volume" in df.columns:
            vol_nonzero = (df["volume"].fillna(0) > 0).mean()
            if vol_nonzero < 0.20:
                logging.warning("Volume coverage < 20%; skipping Volume filter today.")
                self.cfg.min_vol = 0
        # =========================================================

        # Apply Final Constraints
        mask = (
                (df['dte'] >= self.cfg.min_dte) &
                (df['dte'] <= self.cfg.max_dte) &
                (df['volume'] >= self.cfg.min_vol) &
                (df['open_interest'] >= self.cfg.min_oi)
        )
        df = df[mask]
        logging.info(f"{len(df)} rows remaining after filters.")

        candidates: List[CandidateTrade] = []

        # Generate Candidates per Expiration
        for expiry, group in df.groupby('expiration'):
            current_price = group['underlying_price'].iloc[0]

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
                if len(calls) > 2:
                    if not (calls['strike'] - current_price).abs().empty:
                        idx_atm = (calls['strike'] - current_price).abs().argsort()[:1].values[0]
                        idx_short = idx_atm + 2
                        if idx_short < len(calls):
                            long_leg = calls.iloc[idx_atm]
                            short_leg = calls.iloc[idx_short]
                            if short_leg['strike'] > long_leg['strike']:
                                self._try_add_spread(candidates, "Bull Call Spread", "Vertical",
                                                     [long_leg, short_leg], ['long', 'short'], current_price)

                if len(puts) > 2:
                    if not (puts['strike'] - current_price).abs().empty:
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
            raw_date = legs[0]['date']
            clean_legs = [self._clean_leg(legs[0], side='long')]
            cand = self._make_candidate(name, struct_type, clean_legs, spot, raw_date)
            self._validate_candidate(cand)
            candidates_list.append(cand)
        except (ValueError, IndexError) as e:
            logging.debug(f"Failed to add candidate {name}: {e}")
            pass

    def _try_add_spread(self, candidates_list, name, struct_type, raw_legs, sides, spot):
        try:
            raw_date = raw_legs[0]['date']
            clean_legs = []
            for raw_leg, side in zip(raw_legs, sides):
                clean_legs.append(self._clean_leg(raw_leg, side=side))
            cand = self._make_candidate(name, struct_type, clean_legs, spot, raw_date)
            self._validate_candidate(cand)
            candidates_list.append(cand)
        except (ValueError, IndexError) as e:
            logging.debug(f"Failed to add spread {name}: {e}")
            pass

    def _make_candidate(self, name, struct_type, legs, underlying_price, date_val) -> CandidateTrade:
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
        iv = float(row[IV_COL])

        if 'opt_price' in row and not pd.isna(row['opt_price']):
            entry = float(row['opt_price'])
        elif 'bid' in row and 'ask' in row and not pd.isna(row['bid']):
            entry = (float(row['bid']) + float(row['ask'])) / 2.0
        else:
            entry = 0.0

        if isinstance(row['expiration'], pd.Timestamp):
            exp_str = row['expiration'].strftime('%Y-%m-%d')
        else:
            exp_str = str(row['expiration'])

        row_id = int(row.get('row_id', -1))

        return {
            "row_id": row_id,
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
                pass


if __name__ == "__main__":
    gen = CandidateGenerator()
    try:
        df = gen.load_catalog()
        latest_date = df['date'].max()
        results = gen.generate(df, trade_date=str(latest_date.date()))
        logging.info(f"Generated {len(results)} Candidates.")
        print(f"Generated {len(results)} Candidates.")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        print(f"Error: {e}")
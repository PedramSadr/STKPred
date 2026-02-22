import pandas as pd
import os
import numpy as np
import logging
from dataclasses import dataclass
from typing import TypedDict, List, Literal, Optional

LOG_DIR = r"C:\My Documents\Mics\Logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "candidate_generator.log")
EVAL_LOG_FILE = os.path.join(LOG_DIR, "candidate_evaluation_log.csv")

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


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
    structure_type: str
    underlying_price: float
    legs: List[OptionLeg]
    decision: str
    block_reason: str


IV_COL = "implied_volatility"


@dataclass
class GeneratorConfig:
    min_dte: int = 7
    max_dte: int = 90
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Catalog not found at: {path}")
        df = pd.read_csv(path)
        for col in ['date', 'expiration']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def generate(self, catalog_df: pd.DataFrame, trade_date: str = None, spot_price: float = None) -> List[
        CandidateTrade]:
        if trade_date:
            target_date = pd.to_datetime(trade_date).date()
            df = catalog_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'].dt.date == target_date].copy()
        else:
            df = catalog_df.copy()

        if df.empty: return []

        column_mapping = {
            'underlying': 'underlying_price', 'spot': 'underlying_price',
            'current_price': 'underlying_price', 'close': 'opt_price', 'last': 'opt_price'
        }
        df = df.rename(columns=column_mapping)

        if spot_price is not None and spot_price > 0:
            if 'underlying_price' not in df.columns:
                df['underlying_price'] = spot_price
            else:
                df['underlying_price'] = df['underlying_price'].fillna(spot_price)

        required_cols = ["expiration", "type", "strike", "bid", "ask"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logging.error(f"CRITICAL: Missing columns: {missing_cols}. Aborting.")
            return []

        # GLOBAL NORMALIZATION: Enforced exactly once at the top to prevent classification mismatch
        df["type"] = df["type"].astype(str).str.lower().str.strip()

        if 'dte' not in df.columns:
            if 'expiration' in df.columns and 'date' in df.columns:
                df['dte'] = (pd.to_datetime(df['expiration'], errors='coerce') - pd.to_datetime(df['date'],
                                                                                                errors='coerce')).dt.days
                df['dte'] = df['dte'].fillna(-1).astype(int)
            else:
                df['dte'] = -1

        if 'underlying_price' not in df.columns: df['underlying_price'] = np.nan
        if IV_COL not in df.columns:
            if 'iv' in df.columns:
                df = df.rename(columns={'iv': IV_COL})
            else:
                return []

        df[IV_COL] = pd.to_numeric(df[IV_COL], errors='coerce')
        df = df[(df[IV_COL] > 0) & (df[IV_COL] < 3.0)]

        for col in ['dte', 'open_interest', 'strike', 'underlying_price', 'opt_price', 'bid', 'ask']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['dte', 'strike'])
        df["decision"] = "TRADE"
        df["block_reason"] = "OK"

        bad_type = ~df["type"].isin(['call', 'put'])
        df.loc[bad_type, "decision"] = "BLOCK"
        df.loc[bad_type, "block_reason"] = "G0_Invalid_Type"

        bad_quotes = df["bid"].isna() | df["ask"].isna() | (df["bid"] <= 0) | (df["ask"] <= 0) | (df["ask"] < df["bid"])
        df.loc[bad_quotes & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
        df.loc[bad_quotes & (df["block_reason"] == "OK"), "block_reason"] = "G1_Bad_Quotes"

        bad_spot = df["underlying_price"].isna() | (df["underlying_price"] <= 0)
        df.loc[bad_spot & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
        df.loc[bad_spot & (df["block_reason"] == "OK"), "block_reason"] = "G2_Invalid_Spot"

        spot = np.where(df["underlying_price"] > 0, df["underlying_price"], np.nan)
        df["rel_distance"] = (df["strike"] - spot).abs() / spot
        MAX_OTM = 0.20

        call_too_far_otm = (df["type"] == "call") & (df["strike"] > spot) & (((df["strike"] - spot) / spot) > MAX_OTM)
        put_too_far_otm = (df["type"] == "put") & (df["strike"] < spot) & (((spot - df["strike"]) / spot) > MAX_OTM)
        df.loc[(call_too_far_otm | put_too_far_otm) & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
        df.loc[(call_too_far_otm | put_too_far_otm) & (df["block_reason"] == "OK"), "block_reason"] = "G3_Too_Far_OTM"

        K = df["strike"]
        is_call = df["type"] == "call"
        intrinsic = np.where(is_call, np.maximum(spot - K, 0), np.maximum(K - spot, 0))
        upper = np.where(is_call, spot, K)
        arb_fail = (df["ask"] < 0.95 * intrinsic) | (df["ask"] > 1.05 * upper)
        df.loc[arb_fail & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
        df.loc[arb_fail & (df["block_reason"] == "OK"), "block_reason"] = "G4_Arb_Bounds_Failed"

        mid = (df["bid"] + df["ask"]) / 2.0
        spread_abs = df["ask"] - df["bid"]
        spread_pct = np.where(mid > 0, spread_abs / mid, 1.0)
        cond_t1_fail = (mid < 0.50) & (spread_abs > 0.02)
        cond_t2_fail = (mid >= 0.50) & (mid < 1.00) & ((spread_abs > 0.05) | (spread_pct > 0.15))
        cond_t3_fail = (mid >= 1.00) & ((spread_pct > 0.03) | (spread_abs > np.maximum(0.10, 0.02 * mid)))
        reject_mask = cond_t1_fail | cond_t2_fail | cond_t3_fail
        reasons = np.select([cond_t1_fail, cond_t2_fail, cond_t3_fail],
                            ["G5_T1_CHEAP", "G5_T2_NORMAL", "G5_T3_EXPENSIVE"], default="OK")
        df.loc[reject_mask & (df["decision"] == "TRADE"), "block_reason"] = reasons[
            reject_mask & (df["decision"] == "TRADE")]
        df.loc[reject_mask & (df["decision"] == "TRADE"), "decision"] = "BLOCK"

        if "open_interest" in df.columns:
            active_min_oi = self.cfg.min_oi
            if (df["open_interest"].fillna(0) > 0).mean() < 0.20:
                active_min_oi = 0
            oi_fail = (df['open_interest'] < active_min_oi)
            df.loc[oi_fail & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
            df.loc[oi_fail & (df["block_reason"] == "OK"), "block_reason"] = "G6_Low_OI"

        if "volume" in df.columns:
            active_min_vol = self.cfg.min_vol
            if (df["volume"].fillna(0) > 0).mean() < 0.20:
                active_min_vol = 0
            vol_fail = (df['volume'] < active_min_vol)
            df.loc[vol_fail & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
            df.loc[vol_fail & (df["block_reason"] == "OK"), "block_reason"] = "G6_Low_Volume"

        dte_fail = (df['dte'] < self.cfg.min_dte) | (df['dte'] > self.cfg.max_dte)
        df.loc[dte_fail & (df["decision"] == "TRADE"), "decision"] = "BLOCK"
        df.loc[dte_fail & (df["block_reason"] == "OK"), "block_reason"] = "G7_DTE_Out_Of_Bounds"

        audit_cols = ["date", "expiration", "type", "strike", "underlying_price", "dte", "bid", "ask", "mid",
                      "spread_abs", "spread_pct", "rel_distance", "volume", "open_interest", "decision", "block_reason"]
        audit_temp = df.copy()
        audit_temp["mid"] = mid
        audit_temp["spread_abs"] = spread_abs
        audit_temp["spread_pct"] = spread_pct
        audit_df = audit_temp.reindex(columns=audit_cols)

        if not os.path.exists(EVAL_LOG_FILE):
            audit_df.to_csv(EVAL_LOG_FILE, index=False)
        else:
            audit_df.to_csv(EVAL_LOG_FILE, mode='a', header=False, index=False)

        df_trades = df[df["decision"] == "TRADE"].copy()
        candidates: List[CandidateTrade] = []

        for expiry, group in df_trades.groupby('expiration'):
            current_price = group['underlying_price'].median()
            group = group.copy()
            calls = group[group['type'] == 'call'].sort_values('strike')
            puts = group[group['type'] == 'put'].sort_values('strike')

            if not calls.empty:
                idx = (calls['strike'] - current_price).abs().idxmin()
                atm_call = calls.loc[idx]
                if isinstance(atm_call, pd.DataFrame): atm_call = atm_call.iloc[0]
                self._try_add_candidate(candidates, "ATM Call", "Single", [atm_call], current_price)

            if not puts.empty:
                idx = (puts['strike'] - current_price).abs().idxmin()
                atm_put = puts.loc[idx]
                if isinstance(atm_put, pd.DataFrame): atm_put = atm_put.iloc[0]
                self._try_add_candidate(candidates, "ATM Put", "Single", [atm_put], current_price)

            if self.ENABLE_SPREADS:
                if len(calls) > 2:
                    idx_atm = (calls['strike'] - current_price).abs().argsort()[:1].values[0]
                    if idx_atm + 2 < len(calls):
                        long_leg = calls.iloc[idx_atm]
                        short_leg = calls.iloc[idx_atm + 2]
                        if short_leg['strike'] > long_leg['strike']:
                            self._try_add_spread(candidates, "Bull Call Spread", "Vertical", [long_leg, short_leg],
                                                 ['long', 'short'], current_price)

                if len(puts) > 2:
                    idx_atm = (puts['strike'] - current_price).abs().argsort()[:1].values[0]
                    if idx_atm - 2 >= 0:
                        long_leg = puts.iloc[idx_atm]
                        short_leg = puts.iloc[idx_atm - 2]
                        if short_leg['strike'] < long_leg['strike']:
                            self._try_add_spread(candidates, "Bear Put Spread", "Vertical", [long_leg, short_leg],
                                                 ['long', 'short'], current_price)

        return candidates[:self.cfg.max_candidates]

    def _try_add_candidate(self, candidates_list, name, struct_type, legs, spot):
        try:
            raw_date = legs[0]['date'] if 'date' in legs[0] else None
            clean_legs = [self._clean_leg(legs[0], side='long')]
            cand = self._make_candidate(name, struct_type, clean_legs, spot, raw_date, legs[0])
            self._validate_candidate(cand)
            candidates_list.append(cand)
        except (ValueError, IndexError, KeyError) as e:
            logging.debug(f"Failed to add candidate {name}: {e}")

    def _try_add_spread(self, candidates_list, name, struct_type, raw_legs, sides, spot):
        try:
            raw_date = raw_legs[0]['date'] if 'date' in raw_legs[0] else None
            clean_legs = [self._clean_leg(r_leg, side=s) for r_leg, s in zip(raw_legs, sides)]
            cand = self._make_candidate(name, struct_type, clean_legs, spot, raw_date, raw_legs[0])
            self._validate_candidate(cand)
            candidates_list.append(cand)
        except (ValueError, IndexError, KeyError) as e:
            logging.debug(f"Failed to add spread {name}: {e}")

    def _make_candidate(self, name, struct_type, legs, underlying_price, date_val, raw_row) -> CandidateTrade:
        if pd.isna(date_val) or date_val is None:
            date_str = ""
        elif not isinstance(date_val, pd.Timestamp):
            date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
        else:
            date_str = date_val.strftime('%Y-%m-%d')

        return {
            "date": date_str,
            "structure": name,
            "structure_type": struct_type,
            "underlying_price": float(underlying_price),
            "decision": raw_row.get("decision", "TRADE"),
            "block_reason": raw_row.get("block_reason", "OK"),
            "legs": legs
        }

    def _clean_leg(self, row, side: Literal['long', 'short'] = 'long') -> OptionLeg:
        iv = float(row[IV_COL])
        if 'bid' in row and 'ask' in row and not pd.isna(row['bid']) and not pd.isna(row['ask']) and float(
                row['ask']) > 0:
            entry = (float(row['bid']) + float(row['ask'])) / 2.0
        elif 'opt_price' in row and not pd.isna(row['opt_price']):
            entry = float(row['opt_price'])
        else:
            entry = 0.0

        exp_str = row['expiration'].strftime('%Y-%m-%d') if isinstance(row['expiration'], pd.Timestamp) else str(
            row['expiration'])
        return {
            "row_id": int(row.get('row_id', -1)),
            "expiration": exp_str, "strike": float(row['strike']), "type": row['type'],
            "side": side, "dte": int(row['dte']), "iv": iv, "entry_price": entry,
            "bid": float(row.get('bid', 0)), "ask": float(row.get('ask', 0)),
            "delta": float(row.get('delta', 0)), "vega": float(row.get('vega', 0))
        }

    def _validate_candidate(self, cand: CandidateTrade):
        if cand['underlying_price'] <= 0: raise ValueError(f"Invalid underlying price: {cand['underlying_price']}")
        if not cand['legs']: raise ValueError("Candidate has no legs")
        for leg in cand['legs']:
            if leg['iv'] <= 0 or leg['iv'] >= 3.0: raise ValueError(f"Invalid IV: {leg['iv']}")
            if leg['entry_price'] <= 0: raise ValueError(f"Invalid Entry Price: {leg['entry_price']}")
            if leg['dte'] <= 0: raise ValueError(f"Invalid DTE: {leg['dte']}")


if __name__ == "__main__":
    gen = CandidateGenerator()
    try:
        df = gen.load_catalog()
        results = gen.generate(df, trade_date=str(df['date'].max().date()))
        print(f"Generated {len(results)} Candidates.")
    except Exception as e:
        print(f"Error: {e}")
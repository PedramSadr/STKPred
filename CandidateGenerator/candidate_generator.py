import pandas as pd
import os
from dataclasses import dataclass

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

    def generate(self, catalog_df: pd.DataFrame, trade_date: str = None) -> list:
        # 5️⃣ Date Filtering
        if trade_date:
            target_date = pd.to_datetime(trade_date).date()

            # Robust Date Comparison
            df = catalog_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            df = df[df['date'].dt.date == target_date].copy()
        else:
            df = catalog_df.copy()

        # --- DEBUG 1: Date Filter ---
        print("DEBUG rows after date filter:", len(df))

        if df.empty:
            return []

        # 1️⃣ IV Handling (HARD FAIL - No Injection)
        if IV_COL not in df.columns:
            if 'iv' in df.columns:
                print(f"DEBUG: Renaming 'iv' column to '{IV_COL}'")
                df = df.rename(columns={'iv': IV_COL})
            else:
                # FIX 1: Hard Fail if IV is missing. Do not guess.
                raise RuntimeError(
                    f"Missing required column '{IV_COL}' in catalog. "
                    "Re-run surface_generator.py and ensure IV is exported."
                )

        df[IV_COL] = pd.to_numeric(df[IV_COL], errors='coerce')

        # FIX 2: IV Sanity Cap (Filter noise > 300% IV)
        # Drop <= 0 (invalid) AND > 3.0 (garbage/deep OTM)
        df = df[(df[IV_COL] > 0) & (df[IV_COL] < 3.0)]

        # --- DEBUG 2: Post-IV Filter ---
        print("DEBUG rows after IV filter (0 < IV < 3.0):", len(df))

        if df.empty:
            return []

        # FIX 3: Smart NaN Handling
        # Core columns: Drop row if missing (Critical data)
        core_cols = ['dte', 'volume', 'open_interest', 'strike', 'underlying_price', 'bid', 'ask']
        for col in core_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where critical info is NaN
        cols_to_check = [c for c in core_cols if c in df.columns]
        df = df.dropna(subset=cols_to_check)

        # Greeks: Fill with 0 if missing (Non-critical for structure, critical for risk)
        greek_cols = ['delta', 'vega', 'gamma']
        for col in greek_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- DEBUG 3: Post-Nan Filter ---
        print("DEBUG rows after NaN cleaning:", len(df))

        # Apply Standard Filters (Combined)
        mask = (
                (df['dte'] >= self.cfg.min_dte) &
                (df['dte'] <= self.cfg.max_dte) &
                (df['volume'] >= self.cfg.min_vol) &
                (df['open_interest'] >= self.cfg.min_oi)
        )
        df = df[mask]
        print("DEBUG rows after Liquidity/DTE filters:", len(df))

        candidates = []

        # Process per Expiration
        for expiry, group in df.groupby('expiration'):
            current_price = group['underlying_price'].iloc[0]

            group = group.copy()
            if 'type' in group.columns:
                group['type'] = group['type'].astype(str).str.lower()

            calls = group[group['type'] == 'call']
            puts = group[group['type'] == 'put']

            if calls.empty and puts.empty:
                continue

            # 3️⃣ ATM Selection (Nearest Neighbor)
            if not calls.empty:
                idx_call = (calls['strike'] - current_price).abs().argsort()[:1]
                atm_call = calls.iloc[idx_call]

                if not atm_call.empty:
                    candidates.append(self._make_candidate(
                        "ATM Call", "Single", [atm_call.iloc[0]], current_price
                    ))

            if not puts.empty:
                idx_put = (puts['strike'] - current_price).abs().argsort()[:1]
                atm_put = puts.iloc[idx_put]

                if not atm_put.empty:
                    candidates.append(self._make_candidate(
                        "ATM Put", "Single", [atm_put.iloc[0]], current_price
                    ))

        return candidates[:self.cfg.max_candidates]

    def _make_candidate(self, name, struct_type, legs, underlying_price):
        # FIX 4: Safety on Date Formatting
        date_val = pd.to_datetime(legs[0]['date'])

        return {
            "date": date_val.strftime('%Y-%m-%d'),
            "structure": name,
            "type": struct_type,
            "underlying_price": underlying_price,
            "legs": [self._clean_leg(l) for l in legs]
        }

    def _clean_leg(self, row):
        iv = float(row[IV_COL])
        return {
            "expiration": row['expiration'].strftime('%Y-%m-%d'),
            "strike": row['strike'],
            "type": row['type'],
            "bid": row['bid'],
            "ask": row['ask'],
            "dte": int(row['dte']),
            "iv": iv,
            "delta": float(row.get('delta', 0)),
            "vega": float(row.get('vega', 0))
        }


if __name__ == "__main__":
    gen = CandidateGenerator()
    try:
        df = gen.load_catalog()
        latest_date = df['date'].max()
        results = gen.generate(df, trade_date=str(latest_date.date()))
        print(f"Generated {len(results)} Candidates.")
        if results:
            c = results[0]
            print(f"Sample: {c['structure']} | Spot: {c['underlying_price']} | IV: {c['legs'][0]['iv']}")
    except Exception as e:
        print(f"Error: {e}")
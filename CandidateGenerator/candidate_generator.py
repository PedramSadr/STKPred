import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from config import Config


# --- CONFIGURATION CLASS ---
class GeneratorConfig:
    def __init__(self):
        self.MIN_DTE = 7
        self.MAX_DTE = 45
        self.MIN_DELTA = 0.20
        self.MAX_DELTA = 0.50
        self.MIN_LIQUIDITY = 100
        self.MAX_SPREAD_PCT = 0.05
        self.SPREAD_WIDTHS = [5, 10]


class CandidateGenerator:
    """
    Generates trade candidates with deterministic IDs and complete economics.
    Now includes robust column loading to prevent crashes on missing data.
    """

    def __init__(self, catalog_path: str, current_price: float, trade_date=None, config: GeneratorConfig = None):
        self.catalog_path = catalog_path
        self.current_price = current_price
        self.config = config if config else GeneratorConfig()

        # Time Consistency
        if trade_date:
            self.trade_date = pd.Timestamp(trade_date)
        else:
            self.trade_date = pd.Timestamp.now()

        # Strategy Gates
        self.enable_spreads = Config.ENABLE_SPREADS
        self.enable_single_legs = Config.ENABLE_SINGLE_LEGS

        self.catalog = None
        self._load_catalog()

    def _load_catalog(self):
        try:
            self.catalog = pd.read_csv(self.catalog_path)

            # --- FIX: ROBUST COLUMN NORMALIZATION ---
            # 1. Strip whitespace from headers
            self.catalog.columns = [c.strip() for c in self.catalog.columns]

            # 2. Normalize common column names (Handle OpenInterest vs open_interest vs openInterest)
            # Map of {Target Name: [Possible Aliases]}
            alias_map = {
                'openInterest': ['open_interest', 'OpenInterest', 'openinterest', 'OPEN_INTEREST', 'oi', 'OI'],
                'volume': ['Volume', 'vol', 'VOL'],
                'ask': ['Ask', 'ASK', 'ask_price'],
                'bid': ['Bid', 'BID', 'bid_price'],
                'strike': ['Strike', 'STRIKE', 'strike_price']
            }

            for target, aliases in alias_map.items():
                if target not in self.catalog.columns:
                    for alias in aliases:
                        if alias in self.catalog.columns:
                            self.catalog.rename(columns={alias: target}, inplace=True)
                            logging.debug(f"Renamed column '{alias}' to '{target}'")
                            break

            # 3. Fill missing columns with 0 instead of crashing
            for col in ['openInterest', 'volume', 'ask', 'bid', 'strike']:
                if col not in self.catalog.columns:
                    logging.warning(f"Column '{col}' missing. Filling with 0 to prevent crash.")
                    self.catalog[col] = 0

            # 4. Normalization & Dedup
            if 'type' in self.catalog.columns:
                self.catalog['type'] = self.catalog['type'].str.upper().str.strip()

            if 'contractID' in self.catalog.columns:
                self.catalog = self.catalog.drop_duplicates(subset=['contractID'])
            else:
                self.catalog = self.catalog.drop_duplicates(subset=['expiration', 'strike', 'type'])

            # 5. Type Conversions
            self.catalog['expiration'] = pd.to_datetime(self.catalog['expiration'])
            cols = ['strike', 'ask', 'bid', 'openInterest', 'volume']
            for c in cols:
                self.catalog[c] = pd.to_numeric(self.catalog[c], errors='coerce')

            # 6. Quality Filters
            self.catalog = self.catalog[
                (self.catalog['bid'] > 0) &
                (self.catalog['ask'] > 0) &
                (self.catalog['ask'] > self.catalog['bid'])
                ].copy()

            self.catalog['mid'] = (self.catalog['ask'] + self.catalog['bid']) / 2
            self.catalog['spread_pct'] = (self.catalog['ask'] - self.catalog['bid']) / self.catalog['mid']

            # 7. Liquidity Gates
            self.catalog = self.catalog[self.catalog['spread_pct'] <= self.config.MAX_SPREAD_PCT]
            self.catalog['openInterest'] = self.catalog['openInterest'].fillna(0)
            self.catalog['volume'] = self.catalog['volume'].fillna(0)

            self.catalog = self.catalog[
                (self.catalog['openInterest'] >= self.config.MIN_LIQUIDITY) |
                (self.catalog['volume'] >= self.config.MIN_LIQUIDITY)
                ]

            logging.info(f"[CandidateGenerator] Loaded {len(self.catalog)} valid contracts.")

        except Exception as e:
            logging.error(f"[CandidateGenerator] Failed to load catalog: {e}")
            self.catalog = pd.DataFrame()

    def generate_candidates(self) -> List[Dict]:
        candidates = []
        if self.catalog.empty: return candidates

        self.catalog['dte'] = (self.catalog['expiration'] - self.trade_date).dt.days
        subset = self.catalog[
            (self.catalog['dte'] >= self.config.MIN_DTE) &
            (self.catalog['dte'] <= self.config.MAX_DTE)
            ].copy()

        # Generate Puts
        puts = subset[subset['type'] == 'PUT']
        puts = puts[
            (puts['strike'] >= self.current_price * 0.85) &
            (puts['strike'] <= self.current_price * 1.10)
            ]

        for _, row in puts.iterrows():
            # 1. Single Legs (Gated)
            if self.enable_single_legs:
                max_gain = row['strike'] - row['ask']

                candidates.append({
                    'strategy': 'SINGLE_PUT',
                    'contractID': row['contractID'],
                    'legs': [row.to_dict()],
                    'economics': {
                        'entry_cost': row['ask'],
                        'max_loss': row['ask'],
                        'max_gain': max_gain,
                        'breakeven': row['strike'] - row['ask']
                    }
                })

            # 2. Spreads (Gated)
            if self.enable_spreads:
                self._generate_spreads_for_leg(row, puts, candidates)

        return candidates

    def _generate_spreads_for_leg(self, long_leg, all_puts, candidates_list):
        for width in self.config.SPREAD_WIDTHS:
            target_short_strike = long_leg['strike'] - width

            matches = all_puts[
                (all_puts['expiration'] == long_leg['expiration']) &
                (np.isclose(all_puts['strike'], target_short_strike, atol=0.5))
                ].copy()

            if not matches.empty:
                matches = matches.sort_values(by=['openInterest', 'volume'], ascending=[False, False])
                short_leg = matches.iloc[0]

                # Economics
                net_debit = long_leg['ask'] - short_leg['bid']
                if net_debit <= 0.01: continue

                max_loss = net_debit
                max_gain = width - net_debit
                breakeven = long_leg['strike'] - net_debit

                # Canonical ID with Expiration Date
                exp_str = long_leg['expiration'].strftime('%Y%m%d')
                composite_id = f"SPREAD_{exp_str}_{long_leg['strike']}_{short_leg['strike']}_PUT_DEBIT"

                spread_candidate = {
                    'strategy': 'VERTICAL_PUT_DEBIT',
                    'contractID': composite_id,
                    'legs': [long_leg.to_dict(), short_leg.to_dict()],
                    'economics': {
                        'entry_cost': round(net_debit, 2),
                        'max_loss': round(max_loss, 2),
                        'max_gain': round(max_gain, 2),
                        'breakeven': round(breakeven, 2),
                        'width': width
                    }
                }
                candidates_list.append(spread_candidate)
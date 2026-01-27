import pandas as pd
import logging


class CandidateGenerator:
    """
    Generates trading candidates.
    FIXED: Includes "Safety Brake" to prevent infinite loops on massive files.
    """

    def __init__(self, catalog_path, current_price, trade_date=None):
        self.catalog_path = catalog_path
        self.current_price = current_price
        self.trade_date = str(trade_date).strip() if trade_date else None
        self.df_chain = pd.DataFrame()
        self._load_data()

    def _load_data(self):
        try:
            logging.info(f"Loading catalog from {self.catalog_path}...")
            # 1. Load the massive file
            self.df_chain = pd.read_csv(self.catalog_path, low_memory=False)

            # 2. Normalize Columns (Force lowercase)
            self.df_chain.columns = [str(c).lower().strip() for c in self.df_chain.columns]

            # Map aliases
            if 'option_symbol' in self.df_chain.columns and 'symbol' not in self.df_chain.columns:
                self.df_chain.rename(columns={'option_symbol': 'symbol'}, inplace=True)

            # Force 'type' to string and lowercase
            if 'type' in self.df_chain.columns:
                self.df_chain['type'] = self.df_chain['type'].astype(str).str.lower()

            initial_count = len(self.df_chain)

            # 3. Date Filtering (Critical for Performance)
            if self.trade_date:
                # Identify date column
                date_col = None
                possible_cols = ['date', 'quote_date', 'timestamp', 'data_date', 'trade_date']
                for col in possible_cols:
                    if col in self.df_chain.columns:
                        date_col = col
                        break

                if date_col:
                    # [PERFORMANCE] Vectorized Filter
                    # Ensure date column is string for comparison
                    self.df_chain[date_col] = self.df_chain[date_col].astype(str)

                    # Filter for exact match or startswith
                    mask = self.df_chain[date_col].str.startswith(self.trade_date)
                    self.df_chain = self.df_chain[mask].copy()

                    logging.info(f"Filtered Chain: {initial_count} -> {len(self.df_chain)} rows for {self.trade_date}")
                else:
                    logging.warning(f"CRITICAL: No date column found. Available: {list(self.df_chain.columns)}")

            # 4. SAFETY BRAKE: If we still have too many rows, something is wrong.
            # A single day chain for TSLA is usually ~2,000 - 5,000 rows.
            # If we have > 50,000, we are processing history unintentionally.
            if len(self.df_chain) > 50000:
                logging.error(f"SAFETY BRAKE: Chain has {len(self.df_chain)} rows. This will hang the app.")
                logging.error("Forcing empty chain to prevent crash. Check trade_date format.")
                self.df_chain = pd.DataFrame()

        except Exception as e:
            logging.error(f"Failed to load catalog {self.catalog_path}: {e}")

    def get_composite_price(self, legs_list, price_type='bid'):
        if self.df_chain.empty or not legs_list: return 0.0
        total_price = 0.0
        try:
            for leg in legs_list:
                mask = (
                        (self.df_chain['expiration'] == leg['expiration']) &
                        (self.df_chain['strike'] == float(leg['strike'])) &
                        (self.df_chain['type'] == leg['type'].lower())
                )
                row = self.df_chain[mask]
                if row.empty: return 0.0

                side = leg.get('side', 'long')
                if price_type == 'ask':  # Entry
                    p = float(row['ask'].iloc[0]) if side == 'long' else -float(row['bid'].iloc[0])
                elif price_type == 'bid':  # Exit
                    p = float(row['bid'].iloc[0]) if side == 'long' else -float(row['ask'].iloc[0])
                total_price += p
            return total_price
        except Exception:
            return 0.0

    def generate_candidates(self):
        """Generates Put Debit Spreads (30-60 DTE)."""
        if self.df_chain.empty:
            logging.warning("Option Chain is empty. Skipping candidate generation.")
            return []

        puts = self.df_chain[self.df_chain['type'] == 'put'].copy()
        if puts.empty: return []

        # DTE Window Filter (30-60 Days)
        if self.trade_date:
            try:
                trade_dt = pd.to_datetime(self.trade_date)
                puts['expiration'] = pd.to_datetime(puts['expiration'])
                puts['dte'] = (puts['expiration'] - trade_dt).dt.days

                # Filter Logic
                puts = puts[(puts['dte'] >= 30) & (puts['dte'] <= 60)]
                puts['expiration'] = puts['expiration'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                logging.error(f"Date parsing error in generator: {e}")
                return []

        if puts.empty:
            logging.warning("No candidates found in 30-60 DTE window.")
            return []

        expirations = puts['expiration'].unique()
        candidates = []

        for exp in expirations:
            chain_exp = puts[puts['expiration'] == exp]
            long_cands = chain_exp[chain_exp['strike'] > self.current_price]
            short_cands = chain_exp[chain_exp['strike'] < self.current_price]

            for _, long_leg in long_cands.iterrows():
                shorts = short_cands[
                    (short_cands['strike'] < long_leg['strike']) &
                    (short_cands['strike'] >= long_leg['strike'] - 10)
                    ]

                for _, short_leg in shorts.iterrows():
                    legs = [
                        {'side': 'long', 'strike': long_leg['strike'], 'type': 'put', 'expiration': exp},
                        {'side': 'short', 'strike': short_leg['strike'], 'type': 'put', 'expiration': exp}
                    ]

                    spread_ask = self.get_composite_price(legs, 'ask')
                    spread_bid = self.get_composite_price(legs, 'bid')

                    if spread_ask <= 0: continue

                    cand = {
                        'contractID': f"SPREAD_PUT_{exp}_{long_leg['strike']}_{short_leg['strike']}",
                        'symbol': 'TSLA',
                        'strategy': 'PUT_DEBIT_SPREAD',
                        'legs': legs,
                        'economics': {
                            'ask': spread_ask,
                            'bid': spread_bid,
                            'entry_cost': spread_ask,
                            'max_loss': spread_ask
                        }
                    }
                    candidates.append(cand)
        return candidates
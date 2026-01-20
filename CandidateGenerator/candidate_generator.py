import pandas as pd
import logging


class CandidateGenerator:
    """
    Generates trading candidates and handles Real-Time Pricing.
    Phase 4: Functional Candidate Loop + Composite Pricing.
    """

    def __init__(self, catalog_path, current_price, trade_date=None):
        self.catalog_path = catalog_path
        self.current_price = current_price
        self.trade_date = trade_date
        self.df_chain = pd.DataFrame()
        self._load_data()

    def _load_data(self):
        try:
            self.df_chain = pd.read_csv(self.catalog_path)
            self.df_chain.columns = [c.lower().strip() for c in self.df_chain.columns]

            if 'option_symbol' in self.df_chain.columns and 'symbol' not in self.df_chain.columns:
                self.df_chain.rename(columns={'option_symbol': 'symbol'}, inplace=True)

            if 'type' in self.df_chain.columns:
                self.df_chain['type'] = self.df_chain['type'].str.lower()

        except Exception as e:
            logging.error(f"Failed to load catalog {self.catalog_path}: {e}")

    def get_composite_price(self, legs_list, price_type='bid'):
        """
        Calculates spread price from legs.
        Entry (Ask): Long Ask - Short Bid
        Exit  (Bid): Long Bid - Short Ask
        """
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

                if price_type == 'ask':  # Entry (Debit)
                    if side == 'long':
                        p = float(row['ask'].iloc[0])
                    else:
                        p = -float(row['bid'].iloc[0])
                elif price_type == 'bid':  # Exit (Credit)
                    if side == 'long':
                        p = float(row['bid'].iloc[0])
                    else:
                        p = -float(row['ask'].iloc[0])

                total_price += p
            return total_price

        except Exception as e:
            logging.error(f"Pricing Error: {e}")
            return 0.0

    def generate_candidates(self):
        """
        Generates Put Debit Spreads (Long ITM / Short OTM).
        """
        candidates = []
        if self.df_chain.empty: return []

        puts = self.df_chain[self.df_chain['type'] == 'put'].copy()
        if puts.empty: return []

        expirations = puts['expiration'].unique()

        for exp in expirations:
            chain_exp = puts[puts['expiration'] == exp]

            long_cands = chain_exp[chain_exp['strike'] > self.current_price]
            short_cands = chain_exp[chain_exp['strike'] < self.current_price]

            for _, long_leg in long_cands.iterrows():
                # Heuristic: Width ~ $5-$15
                shorts = short_cands[
                    (short_cands['strike'] < long_leg['strike']) &
                    (short_cands['strike'] >= long_leg['strike'] - 15)
                    ]

                for _, short_leg in shorts.iterrows():
                    legs = [
                        {'side': 'long', 'strike': long_leg['strike'], 'type': 'put', 'expiration': exp},
                        {'side': 'short', 'strike': short_leg['strike'], 'type': 'put', 'expiration': exp}
                    ]

                    spread_ask = self.get_composite_price(legs, 'ask')
                    spread_bid = self.get_composite_price(legs, 'bid')

                    if spread_ask <= 0: continue

                    max_loss = spread_ask

                    cand = {
                        'contractID': f"SPREAD_PUT_{exp}_{long_leg['strike']}_{short_leg['strike']}",
                        'symbol': 'TSLA',
                        'strategy': 'PUT_DEBIT_SPREAD',
                        'legs': legs,
                        'economics': {
                            'ask': spread_ask,
                            'bid': spread_bid,
                            'entry_cost': spread_ask,
                            'max_loss': max_loss
                        }
                    }
                    candidates.append(cand)

        return candidates
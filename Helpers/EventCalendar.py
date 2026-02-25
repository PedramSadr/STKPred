import json
import os
import numpy as np
import logging


class EventCalendar:
    def __init__(self, macro_json_path: str, earnings_json_path: str):
        self.macro_events = self._load_json(macro_json_path)
        self.earnings_events = self._load_json(earnings_json_path)

        # Standard US Market Holidays for 2026 to ensure accurate T-1 calculations
        self.market_holidays = [
            '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
            '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
            '2026-11-26', '2026-12-25'
        ]

        logging.info(
            f"EventCalendar initialized. Loaded {len(self.macro_events)} macro events and {len(self.earnings_events)} earnings windows.")

    def _load_json(self, path: str):
        if not os.path.exists(path):
            logging.error(f"Calendar file missing: {path}")
            return []
        with open(path, 'r') as f:
            data = json.load(f)
            # Safely unwrap dictionary if JSON is wrapped in {"events": []} or {"records": []}
            if isinstance(data, dict):
                return data.get("events", data.get("records", []))
            return data if isinstance(data, list) else []

    def check_trade_date(self, trade_date: str, ticker: str = "TSLA"):
        t_date = np.datetime64(trade_date, 'D')
        reasons = []

        # 1. Macro Events
        for ev in self.macro_events:
            ev_date = np.datetime64(ev['date'], 'D')
            category = ev.get('category', '')

            if "Derivatives" in category:
                if t_date == ev_date:
                    reasons.append(f"Macro Block (D0): {ev['event_name']}")
            else:
                d_minus_1 = np.busday_offset(ev_date, -1, roll='preceding', holidays=self.market_holidays)
                if t_date == ev_date:
                    reasons.append(f"Macro Block (D0): {ev['event_name']}")
                elif t_date == d_minus_1:
                    reasons.append(f"Macro Block (D-1): {ev['event_name']}")

        # 2. Earnings Events
        for er in self.earnings_events:
            if er.get('ticker') != ticker:
                continue

            status = er.get('status', 'TBD').upper()

            if status == 'TBD':
                start = np.datetime64(er['expected_window_start'], 'D')
                end = np.datetime64(er['expected_window_end'], 'D')
                if start <= t_date <= end:
                    reasons.append(f"Earnings Block (TBD Window): {er['fiscal_quarter']}")
            else:
                conf_date_str = er.get('confirmed_date', '')
                if conf_date_str:
                    conf_date = np.datetime64(conf_date_str, 'D')
                    d_minus_1 = np.busday_offset(conf_date, -1, roll='preceding', holidays=self.market_holidays)
                    d_plus_1 = np.busday_offset(conf_date, 1, roll='forward', holidays=self.market_holidays)

                    if d_minus_1 <= t_date <= d_plus_1:
                        reasons.append(f"Earnings Block (D-1 to D+1): {er['fiscal_quarter']}")

        is_clear = len(reasons) == 0
        return is_clear, reasons
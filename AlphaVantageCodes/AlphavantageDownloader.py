import requests
import os
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Alpha Vantage API URL and key (fallback)
API_KEY = "D3738GH3ZEDDZKSB"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
# Default output CSV/JSON directory and filename template
DEFAULT_OUTPUT_DIR = r"C:\My Documents\Mics\Logs\AlphaVantage\TSLA_Options_Chain_Historical"
FILENAME_TEMPLATE = "tsla_options_chain_hist_{date}.csv"  # date -> YYYYMMDD


def is_json_text(text: str) -> bool:
    text = text.lstrip()
    return text.startswith("{") or text.startswith("[")


def find_rows_in_json(data: Any) -> List[Dict]:
    """Return a list of dict rows if found in the JSON structure, otherwise empty list."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    if isinstance(data, dict):
        candidate_keys = [
            "historical", "data", "options", "option_chain", "optionChains", "results",
            "records", "items"
        ]
        for k in candidate_keys:
            v = data.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        # fallback: check values
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def write_rows_to_csv(rows: List[Dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        # nothing structured to write
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('')
        return
    headers = set()
    for r in rows:
        headers.update(r.keys())
    headers = list(headers)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # write CSV header and rows
        import csv
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            # flatten nested structures
            row = {k: (v if not isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False)) for k, v in r.items()}
            writer.writerow(row)


def save_text_to_file(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def download_alpha_vantage_range(from_date: str, to_date: str, output_dir: str = DEFAULT_OUTPUT_DIR,
                                 api_key: str = API_KEY, sleep_seconds: float = 12.0) -> None:
    """
    Download Alpha Vantage HISTORICAL_OPTIONS for TSLA for each date in [from_date, to_date]
    and save each day's response into a per-day CSV (or .json when response is JSON) named
    tsla_options_chain_hist_YYYYMMDD.csv (or .json).

    Note: Alpha Vantage free rate limits are strict (typically 5 requests/minute). Set
    sleep_seconds appropriately (default 12s -> ~5 requests/minute) to avoid being rate-limited.

    from_date and to_date must be strings in YYYY-MM-DD format.
    """
    # validate dates
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError as ve:
        raise ValueError("from_date and to_date must be in YYYY-MM-DD format") from ve

    if start > end:
        raise ValueError("from_date must be earlier than or equal to to_date")

    current = start
    session = requests.Session()

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        filename_date = current.strftime('%Y%m%d')
        csv_name = FILENAME_TEMPLATE.format(date=filename_date)
        csv_path = os.path.join(output_dir, csv_name)
        json_name = f"tsla_options_chain_hist_{filename_date}.json"
        json_path = os.path.join(output_dir, json_name)

        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': 'TSLA',
            'date': date_str,
            'apikey': api_key,
            'datatype': 'csv'  # request CSV when possible
        }

        try:
            print(f"Requesting date {date_str} ...")
            resp = session.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
            # Save raw response regardless of status so you can inspect it
            content_type = resp.headers.get('Content-Type', '').lower()
            text = resp.text

            # If response appears to be JSON
            if 'application/json' in content_type or is_json_text(text):
                # try to parse JSON
                try:
                    data = resp.json()
                except ValueError:
                    # treat as text
                    save_text_to_file(json_path, text)
                    print(f"Saved raw (non-JSON-parsable) response to {json_path}")
                    current += timedelta(days=1)
                    time.sleep(sleep_seconds)
                    continue

                # if data contains structured rows, convert to CSV
                rows = find_rows_in_json(data)
                if rows:
                    write_rows_to_csv(rows, csv_path)
                    print(f"Wrote structured rows to {csv_path} ({len(rows)} rows)")
                else:
                    # there's JSON but no structured rows; save the JSON for inspection
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"Saved JSON response to {json_path} (no structured rows found)")
            else:
                # treat as CSV or raw text
                save_text_to_file(csv_path, text)
                print(f"Saved CSV/text response to {csv_path}")

        except requests.HTTPError as he:
            print(f"HTTP error for {date_str}: {he}")
        except Exception as e:
            print(f"Error requesting {date_str}: {e}")

        # advance date and sleep to respect rate limits
        current += timedelta(days=1)
        time.sleep(sleep_seconds)

    print("All done.")


if __name__ == "__main__":
    # Example usage: download a short range. Adjust dates as needed.
    # WARNING: Downloading many days will be rate-limited by Alpha Vantage. Use Polygon for bulk historical data.
    example_from = '2020-01-01'
    example_to = '2020-12-31'
    download_alpha_vantage_range(example_from, example_to, output_dir=DEFAULT_OUTPUT_DIR, sleep_seconds=1.0)

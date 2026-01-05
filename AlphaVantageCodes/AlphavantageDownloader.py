import requests
import os
import json
import time
import glob
from datetime import datetime, timedelta, date
from typing import Any, Dict, List

# --- CONFIGURATION ---
API_KEY = "QR4I54YXBGUV6GQA"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
DEFAULT_OUTPUT_DIR = r"C:\My Documents\Mics\Logs\AlphaVantage\TSLA_Options_Chain_Historical"
FILENAME_TEMPLATE = "tsla_options_chain_hist_{date}.csv"  # date -> YYYYMMDD

# If the folder is empty, start from this date (e.g., 2025 or 2023)
DEFAULT_START_DATE = "2025-01-01"


def is_json_text(text: str) -> bool:
    text = text.lstrip()
    return text.startswith("{") or text.startswith("[")


def find_rows_in_json(data: Any) -> List[Dict]:
    """Return a list of dict rows if found in the JSON structure."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    if isinstance(data, dict):
        candidate_keys = ["historical", "data", "options", "option_chain", "optionChains", "results", "records",
                          "items"]
        for k in candidate_keys:
            v = data.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def write_rows_to_csv(rows: List[Dict], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        return
    headers = set().union(*(d.keys() for d in rows))

    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for r in rows:
            row = {k: (v if not isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False)) for k, v in
                   r.items()}
            writer.writerow(row)


def get_last_downloaded_date(output_dir: str) -> date:
    """
    Scans the output directory to find the LATEST date that already exists.
    Returns None if no files are found.
    """
    if not os.path.exists(output_dir):
        return None

    # Find all files matching the pattern
    files = glob.glob(os.path.join(output_dir, "tsla_options_chain_hist_*.csv"))
    if not files:
        return None

    dates = []
    for f in files:
        try:
            # Extract date part: "tsla_options_chain_hist_20251231.csv" -> "20251231"
            date_str = os.path.basename(f).split('_')[-1].replace('.csv', '')
            dt = datetime.strptime(date_str, "%Y%m%d").date()
            dates.append(dt)
        except ValueError:
            continue

    if not dates:
        return None

    return max(dates)


def download_alpha_vantage_range(from_date: date, to_date: date, output_dir: str = DEFAULT_OUTPUT_DIR,
                                 api_key: str = API_KEY) -> None:
    current = from_date
    print(f"Checking range: {current} -> {to_date}")

    # Build list of missing dates (excluding weekends)
    missing_dates = []
    while current <= to_date:
        if current.weekday() < 5:  # Skip weekends
            filename_date = current.strftime('%Y%m%d')
            csv_path = os.path.join(output_dir, FILENAME_TEMPLATE.format(date=filename_date))

            # CHECK: If file exists, skip (Incremental Logic)
            if not os.path.exists(csv_path):
                missing_dates.append(current)

        current += timedelta(days=1)

    print(f"Missing {len(missing_dates)} days to download.")
    if not missing_dates:
        print("✅ All data is up to date.")
        return

    session = requests.Session()

    for i, current_date in enumerate(missing_dates):
        date_str = current_date.strftime('%Y-%m-%d')
        filename_date = current_date.strftime('%Y%m%d')
        csv_path = os.path.join(output_dir, FILENAME_TEMPLATE.format(date=filename_date))
        json_path = os.path.join(output_dir, f"tsla_options_chain_hist_{filename_date}.json")

        print(f"[{i + 1}/{len(missing_dates)}] Downloading {date_str}...", end=" ", flush=True)

        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': 'TSLA',
            'date': date_str,
            'apikey': api_key,
            'datatype': 'csv'
        }

        try:
            resp = session.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
            text = resp.text

            # Handle Errors / Rate Limits
            if "Error Message" in text and "Invalid API" not in text:
                if "rate limit" in text.lower():
                    print("\n❌ RATE LIMIT REACHED.")
                    break

            if is_json_text(text) or "Error" in text:
                # Likely JSON (Holiday or Empty)
                try:
                    data = resp.json()
                    rows = find_rows_in_json(data)
                    if rows:
                        write_rows_to_csv(rows, csv_path)
                        print(f"Saved {len(rows)} rows.")
                    else:
                        # Holiday Marker
                        with open(csv_path, 'w') as f:
                            f.write("status,message\nHoliday,No Data")
                        print("Holiday (Marked).")
                except:
                    with open(csv_path, 'w') as f:
                        f.write("status,message\nHoliday,No Data")
                    print("No Data (Marked).")
            else:
                # Valid CSV
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print("Saved CSV.")

        except Exception as e:
            print(f"Error: {e}")

        # Sleep 12s to stay safe (5 calls/min)
        time.sleep(12.0)


if __name__ == "__main__":
    # 1. Determine End Date: YESTERDAY (Avoids partial 'Today' data)
    yesterday = date.today() - timedelta(days=1)

    # 2. Determine Start Date:
    #    - If files exist, start from Last_Date + 1 Day
    #    - If no files, start from Default
    last_downloaded = get_last_downloaded_date(DEFAULT_OUTPUT_DIR)

    if last_downloaded:
        start_date = last_downloaded + timedelta(days=1)
        print(f"Found existing history ending: {last_downloaded}")
        print(f"Resuming download from:      {start_date}")
    else:
        start_date = datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date()
        print(f"No history found. Starting from default: {start_date}")

    # 3. Run Smart Download
    if start_date <= yesterday:
        download_alpha_vantage_range(start_date, yesterday, output_dir=DEFAULT_OUTPUT_DIR)
    else:
        print("✅ Date range is in the future or already covered. Nothing to do.")
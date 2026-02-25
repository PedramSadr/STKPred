import os
import glob
import argparse
import pandas as pd
import logging

# Setup standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def append_csv_files(input_dir, output_file, pattern="*.csv", dedupe=True, parse_dates=False, skip_first_line=False):
    search_pattern = os.path.join(input_dir, pattern)
    file_list = glob.glob(search_pattern)

    if not file_list:
        logging.warning(f"No files found matching {search_pattern}")
        return

    logging.info(f"Found {len(file_list)} files in {input_dir}")

    dfs = []
    for f in file_list:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if not dfs:
        logging.error("No valid dataframes to concatenate.")
        return

    logging.info(f"Concatenating {len(dfs)} DataFrames...")
    combined = pd.concat(dfs, ignore_index=True)

    if dedupe:
        initial_len = len(combined)
        combined = combined.drop_duplicates()
        logging.info(f"Dropped {initial_len - len(combined)} duplicate rows")

    # THE FIX: Safely convert to numeric, but EXPLICITLY PROTECT strings and dates
    protected_strings = ['date', 'expiration', 'type', 'symbol', 'contractid', 'contract_id']

    for col in combined.columns:
        if combined[col].dtype == 'object' and str(col).lower().strip() not in protected_strings:
            try:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
            except Exception:
                pass

    logging.info(f"✅ Success! Wrote {output_file} ({len(combined)} rows)")
    combined.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Append and merge historical CSV files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the daily CSV files")
    parser.add_argument("--output_file", required=True, help="Path to save the combined master CSV")
    parser.add_argument("--pattern", default="*.csv", help="File matching pattern")
    parser.add_argument("--dedupe", action="store_true", default=True,
                        help="Deduplicate rows across the combined dataset")
    parser.add_argument("--parse_dates", action="store_true", default=False)
    parser.add_argument("--skip_first_line", action="store_true", default=False)

    args = parser.parse_args()

    append_csv_files(
        args.input_dir,
        args.output_file,
        pattern=args.pattern,
        dedupe=args.dedupe,
        parse_dates=args.parse_dates,
        skip_first_line=args.skip_first_line
    )
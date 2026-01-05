import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV file with aggressive header cleaning."""
    try:
        # FIX 1: Use 'utf-8-sig' to strip invisible BOM characters (\ufeff)
        df = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return pd.DataFrame()

    # Normalize Columns: Strip whitespace and lower case
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Drop fully empty rows
    df.dropna(how='all', inplace=True)

    # FIX 2: Aggressive "Hard" Header Removal
    # Instead of comparing to the column name, we look for the literal words
    # that indicate a header row, because column names can be messed up.
    if len(df.columns) > 0:
        # Check specific columns if they exist, otherwise check the first column
        target_col = df.columns[0]
        if 'date' in df.columns:
            target_col = 'date'

        # Identify rows where the content matches known header keywords
        # This catches "date", "Date", "DATE", etc.
        mask_rogue_header = df[target_col].astype(str).str.strip().str.lower().isin(
            ['date', 'strike', 'expiration', 'symbol'])

        if mask_rogue_header.any():
            count = mask_rogue_header.sum()
            logger.debug("🧹 Dropping %d rogue header rows from %s", count, path)
            df = df.loc[~mask_rogue_header]

    return df


def append_csv_files(input_dir: str, output_file: str, pattern: str = "*.csv", dedupe: bool = True,
                     parse_dates: bool = False, skip_first_line: bool = False) -> Path:
    in_path = Path(input_dir)
    if not in_path.exists() or not in_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(in_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching {pattern}")

    logger.info("Found %d files in %s", len(files), input_dir)

    dfs = []
    for idx, f in enumerate(files):
        # We use the safe reader for EVERYTHING now.
        # It is smarter than the 'skiprows=1' logic because it detects headers dynamically.
        df = _read_csv_safely(f)

        if df.empty:
            continue

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No readable CSVs found to concatenate.")

    logger.info("Concatenating %d DataFrames...", len(dfs))
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # FIX 3: Final Sweep on the Combined Data
    # Just in case a header slipped through (e.g. from an unlabelled index column)
    if 'date' in combined.columns:
        mask = combined['date'].astype(str).str.strip().str.lower() == 'date'
        if mask.any():
            logger.info("🧹 Final Sweep: Dropping %d rogue headers from combined data", mask.sum())
            combined = combined[~mask]

    # Deduplicate
    if dedupe:
        before = len(combined)
        combined.drop_duplicates(inplace=True)
        after = len(combined)
        logger.info("Dropped %d duplicate rows", before - after)

    # Type Inference
    for col in combined.columns:
        if col == 'date': continue
        # Try numeric conversion
        combined[col] = pd.to_numeric(combined[col], errors='ignore')

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(out_path, index=False)
    logger.info("✅ Success! Wrote %s (%d rows)", out_path, len(combined))
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Append CSV files safely.")
    p.add_argument('--input-dir', '-i', required=True, help='Directory containing CSV files')
    p.add_argument('--output-file', '-o', required=True, help='Output CSV file path')
    p.add_argument('--pattern', default='*.csv', help='Glob pattern')
    p.add_argument('--no-dedupe', dest='dedupe', action='store_false')
    p.add_argument('--parse-dates', action='store_true')
    p.add_argument('--skip-first-line', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    try:
        append_csv_files(args.input_dir, args.output_file, pattern=args.pattern, dedupe=args.dedupe,
                         parse_dates=args.parse_dates, skip_first_line=args.skip_first_line)
    except Exception as e:
        logger.exception("Failed: %s", e)
        raise
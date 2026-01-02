import argparse
import logging
from pathlib import Path
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV file and try to normalize common formatting issues.

    - Uses low_memory=False to avoid mixed dtype warnings.
    - Converts multi-index column names to flat strings.
    - Drops rows that are duplicated header rows inside file (e.g. a repeated 'Date' header).
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return pd.DataFrame()

    # If pandas read a multi-header, flatten column tuples
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x and str(x) != '']) for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    # Drop fully empty rows
    df.dropna(how='all', inplace=True)

    # Remove rows that look like repeated header rows (e.g. first column equals 'Date')
    if len(df.columns) > 0:
        first_col = df.columns[0]
        mask_header_rows = df[first_col].astype(str).str.strip().str.lower() == str(first_col).strip().lower()
        if mask_header_rows.any():
            logger.debug("Dropping %d repeated header rows from %s", mask_header_rows.sum(), path)
            df = df.loc[~mask_header_rows]

    return df


def append_csv_files(input_dir: str, output_file: str, pattern: str = "*.csv", dedupe: bool = True, parse_dates: bool = False, skip_first_line: bool = False) -> Path:
    """Append/concatenate all CSV files matching pattern in input_dir and write to output_file.

    If skip_first_line is True, the first file is read normally and subsequent files are read with skiprows=1
    and their columns coerced to the first file's column names. This handles files that include an extra header
    row at the top.

    Returns the output file path.
    """
    in_path = Path(input_dir)
    if not in_path.exists() or not in_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(in_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching {pattern}")

    logger.info("Found %d files in %s", len(files), input_dir)

    dfs = []
    first_cols = None
    for idx, f in enumerate(files):
        logger.info("Reading %s", f)
        df = pd.DataFrame()
        if skip_first_line and idx > 0:
            # Read subsequent files skipping the first line and supply names from the first file
            if first_cols is None:
                # if for some reason first_cols not set, fall back to safe reader
                df = _read_csv_safely(f)
            else:
                try:
                    df = pd.read_csv(f, header=None, skiprows=1, names=first_cols, low_memory=False)
                except Exception as e:
                    logger.warning("Failed to read %s with skiprows=1: %s. Falling back to safe reader.", f, e)
                    df = _read_csv_safely(f)
        else:
            df = _read_csv_safely(f)

        if df.empty:
            logger.info("  -> empty or unreadable, skipping")
            continue

        if first_cols is None:
            first_cols = list(df.columns)

        # Optionally parse a Date column if present
        if parse_dates and {'Year', 'Month', 'Day'}.issubset(set(df.columns)):
            try:
                df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
            except Exception:
                pass

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No readable CSVs found to concatenate.")

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # Remove duplicated header rows that sometimes slip through (rows where first column equals its header name)
    if combined.shape[1] > 0:
        first_col = combined.columns[0]
        mask_header_rows = combined[first_col].astype(str).str.strip().str.lower() == str(first_col).strip().lower()
        if mask_header_rows.any():
            logger.info("Dropping %d header-like rows from combined dataframe", mask_header_rows.sum())
            combined = combined.loc[~mask_header_rows].reset_index(drop=True)

    # Drop exact duplicate rows if requested
    if dedupe:
        before = len(combined)
        combined.drop_duplicates(inplace=True)
        after = len(combined)
        logger.info("Dropped %d duplicate rows", before - after)

    # Try to infer a sensible dtype for numeric columns: convert object columns that look numeric
    for col in combined.select_dtypes(include=['object']).columns:
        # skip Date-like columns
        if col.lower() in ('date', 'year', 'month', 'day'):
            continue
        try:
            combined[col] = pd.to_numeric(combined[col], errors='ignore')
        except Exception:
            pass

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write combined CSV
    combined.to_csv(out_path, index=False)
    logger.info("Wrote combined CSV to %s (%d rows, %d columns)", out_path, combined.shape[0], combined.shape[1])
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Append/concatenate all CSV files in a directory into one CSV.")
    p.add_argument('--input-dir', '-i', default=r"C:\My Documents\Mics\Logs\AlphaVantage\TSLA_Options_Chain_Historical",
                   help='Directory containing CSV files to append')
    p.add_argument('--output-file', '-o', default=r"C:\My Documents\Mics\Logs\AlphaVantage\TSLA_Options_Chain_Historical_combined.csv",
                   help='Output CSV file path')
    p.add_argument('--pattern', default='*.csv', help='Glob pattern to match files')
    p.add_argument('--no-dedupe', dest='dedupe', action='store_false', help='Do not drop duplicate rows')
    p.add_argument('--parse-dates', dest='parse_dates', action='store_true', help='Try to parse Year/Month/Day into a Date column')
    p.add_argument('--skip-first-line', dest='skip_first_line', action='store_true', help='Skip the first line of every file except the first (useful for repeated headers)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    try:
        append_csv_files(args.input_dir, args.output_file, pattern=args.pattern, dedupe=args.dedupe, parse_dates=args.parse_dates, skip_first_line=args.skip_first_line)
    except Exception as e:
        logger.exception("Failed to append CSV files: %s", e)
        raise

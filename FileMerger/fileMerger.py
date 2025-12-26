import pandas as pd
import os
import io  # New: Used for in-memory file handling

# --- FILE PATHS (BASED ON YOUR REQUEST) ---
DAILY_DATA_PATH = r'C:\My Documents\Mics\Logs\tsla_daily.csv'
OPTIONS_DATA_PATH = r'C:\My Documents\Mics\Logs\TSLA_Options_Chain_Historical_combined.csv'
OUTPUT_DATA_PATH = r'C:\My Documents\Mics\Logs\TSLA_Options_Chain_With_Indicators.csv'

# --- COLUMNS TO MERGE ---
# These are the technical indicators we want to pull from the daily file.
#This code performs an inner join between two CSV files containing daily stock data and options chain data,
INDICATOR_COLUMNS = ['Date', 'RSI', 'VWAP', 'SMA_25', 'SMA_50', 'SMA_100', 'SMA_200']

# --- MOCK DATA FOR TESTING (Define expected structure) ---
# 1. Daily Data (Source for indicators)
MOCK_DAILY_DATA = """Date,Open,Close,RSI,VWAP,SMA_25,SMA_50,SMA_100,SMA_200
2025-01-01,100.0,101.0,70.5,100.5,98.0,95.0,90.0,85.0
2025-01-02,102.0,103.0,65.2,102.5,99.0,96.0,91.0,86.0
2025-01-03,104.0,105.0,60.1,104.5,100.0,97.0,92.0,87.0
2025-01-04,106.0,107.0,55.0,106.5,101.0,98.0,93.0,88.0
"""

# 2. Options Data (Target to receive indicators)
MOCK_OPTIONS_DATA = """Date,ContractID,Strike,Type,Last,Volume
2025-01-01,C123,105,C,1.50,1000
2025-01-02,P456,95,P,0.50,500
2025-01-02,C789,100,C,2.00,1200
2025-01-04,P012,110,P,1.20,300
"""


def merge_financial_data(daily_path: str, options_path: str, output_path: str, indicator_cols: list):
    """
    Reads two CSV files, ensures date columns are properly formatted,
    performs an inner join, and saves the resulting DataFrame.
    """
    print(f"Starting merge operation...")

    if not os.path.exists(daily_path) or not os.path.exists(options_path):
        print(f"Error: One or both input files not found.")
        print(f"Daily Data Check: {os.path.exists(daily_path)}")
        print(f"Options Data Check: {os.path.exists(options_path)}")
        # Check if we are running in the test environment (using StringIO)
        if daily_path.startswith("mock:") and options_path.startswith("mock:"):
            print("Using mock data from memory for testing.")
            # Create DataFrames directly from strings (used in test_merge)
            df_daily = pd.read_csv(io.StringIO(daily_path.split("mock:")[1]))
            df_options = pd.read_csv(io.StringIO(options_path.split("mock:")[1]))
        else:
            return  # Exit if files are missing and not running mock test

    else:
        # Load from disk if files exist
        try:
            df_daily = pd.read_csv(daily_path)
            df_options = pd.read_csv(options_path)
        except Exception as e:
            print(f"Error reading files from disk: {e}")
            return

    try:
        # Normalize column names to lowercase and strip whitespace (a good practice)
        df_daily.columns = df_daily.columns.str.strip().str.lower()
        df_options.columns = df_options.columns.str.strip().str.lower()

        # Ensure 'date' is present in both dataframes
        if 'date' not in df_daily.columns or 'date' not in df_options.columns:
            print("Error: 'Date' column (lowercase) not found in both files. Check capitalization.")
            return

        # 2. Crucial: Convert 'date' column to datetime objects for reliable joining
        df_daily['date'] = pd.to_datetime(df_daily['date'], errors='coerce').dt.date
        df_options['date'] = pd.to_datetime(df_options['date'], errors='coerce').dt.date

        # 3. Select only the necessary columns from the daily data
        # Ensure the selected columns are lowercase to match the DataFrame columns
        # Filter indicator_cols list to only include columns present in df_daily
        required_daily_cols = [c.lower() for c in indicator_cols if c.lower() in df_daily.columns]

        if 'date' not in required_daily_cols:
            required_daily_cols.append('date')  # Ensure date is always included

        df_indicators = df_daily[required_daily_cols].copy()

        # 4. Perform the Inner Join (Merge)
        # We join where 'date' columns are exactly equal in both dataframes.
        df_merged = pd.merge(
            df_options,
            df_indicators,
            on='date',
            how='inner'
        )

        # 5. Save the result
        # If output_path is 'test_output.csv', we save a real file for inspection.
        # If running a live run, it saves to the user's requested path.
        df_merged.to_csv(output_path, index=False)

        print(f"Successfully merged {len(df_merged)} rows.")
        print(f"Output file saved to: {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")


# --- NEW: Testing Function ---
def test_merge(mock_daily: str, mock_options: str):
    """
    Runs the merge function using in-memory mock data and checks the output integrity.
    """
    print("\n--- Running Unit Test with Mock Data ---")
    TEST_DAILY_PATH = "mock:" + mock_daily
    TEST_OPTIONS_PATH = "mock:" + mock_options
    TEST_OUTPUT_PATH = "test_output.csv"

    # 1. Run the merge function using the mock data paths
    merge_financial_data(TEST_DAILY_PATH, TEST_OPTIONS_PATH, TEST_OUTPUT_PATH, INDICATOR_COLUMNS)

    # 2. Verify the output file was created and load it
    if not os.path.exists(TEST_OUTPUT_PATH):
        print("TEST FAILED: Output file not created.")
        return

    df_result = pd.read_csv(TEST_OUTPUT_PATH)

    # 3. Validation Checks

    # Check A: Verify the number of rows (Inner Join)
    # Mock Options has 4 rows, dates are 2025-01-01, 2025-01-02 (x2), 2025-01-04
    # Mock Daily has dates 2025-01-01, 2025-01-02, 2025-01-03, 2025-01-04
    # Date 2025-01-03 exists in Daily but not Options.
    # Expected rows = 4 (from options: 1, 2, 2, 4)
    expected_rows = 4
    if len(df_result) == expected_rows:
        print(f"TEST PASSED: Correct number of rows merged ({len(df_result)}).")
    else:
        print(f"TEST FAILED: Expected {expected_rows} rows, got {len(df_result)}.")

    # Check B: Verify a merged column (e.g., RSI) exists and has the correct value
    if 'rsi' in df_result.columns:
        # Check the RSI value for the known date 2025-01-01 (should be 70.5)
        test_value = df_result[df_result['date'] == '2025-01-01']['rsi'].iloc[0]
        if test_value == 70.5:
            print(f"TEST PASSED: RSI column merged correctly (RSI=70.5).")
        else:
            print(f"TEST FAILED: RSI value mismatch. Got {test_value}, expected 70.5.")
    else:
        print("TEST FAILED: RSI column not found in output.")

    # 4. Cleanup
    os.remove(TEST_OUTPUT_PATH)
    print("Test cleanup complete.")


if __name__ == "__main__":
    # --- Execute the main file merge (if files exist) ---
    if os.path.exists(DAILY_DATA_PATH) and os.path.exists(OPTIONS_DATA_PATH):
        merge_financial_data(DAILY_DATA_PATH, OPTIONS_DATA_PATH, OUTPUT_DATA_PATH, INDICATOR_COLUMNS)
    else:
        print("\n--- Running LIVE FILE CHECK ---")
        print("Live data files not found. Running mock test instead.")

        # --- Run the included unit test ---
        test_merge(MOCK_DAILY_DATA, MOCK_OPTIONS_DATA)
from datetime import date

import yfinance as yf
import pandas as pd
import pandas_ta as ta


def get_stock_data_with_indicators(ticker, start_date, end_date):
    """
    Downloads stock data and calculates RSI and MACD using pandas_ta.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): Start date for data retrieval (YYYY-MM-DD).
        end_date (str): End date for data retrieval (YYYY-MM-DD).

    Returns:
        pandas.DataFrame: DataFrame with Open, High, Low, Close, RSI, and MACD.
    """

    # Download stock data
    df = yf.download(ticker, start=start_date, end=end_date)

    # Check if df is empty (no data downloaded)
    if df.empty:
        raise ValueError(f"No data downloaded for ticker {ticker} in the specified date range.")

    # --- FIX START ---
     # 1. Flatten MultiIndex Columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        # Join the column names to a single string, e.g., ('Adj Close', 'AAPL') -> 'Adj Close_AAPL'
        # Or, if you only want the last level (most common case for OHLC data from yfinance),
        # use df.columns.get_level_values(-1).
        # Let's try combining levels for robustness, then strip.
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
        print("Flattened MultiIndex columns.")

    for col in df.columns:
        if col.endswith("_TSLA"):
            new_col = col.replace("_TSLA", "")
            df = df.rename(columns={col: new_col})
            print(f"Renamed column '{col}' to '{new_col}'.")

    # 2. Identify and standardize the 'Close' price column
    found_close_col = None
    for col in df.columns:
        # Check for 'Close' (case-insensitive)
        if 'close' in col.lower():
            # Prioritize 'Adj Close' if present, otherwise 'Close'
            if 'adj close' in col.lower():
                found_close_col = col
                break  # Found the most specific close column
            elif 'close' == col.lower():  # Exact match for 'Close'
                found_close_col = col
                # Don't break yet, in case 'Adj Close' comes later

    if found_close_col is None:
        raise ValueError(f"DataFrame must contain a 'Close' price column. Found columns: {df.columns.tolist()}")

    # Rename the identified column to 'Close' if it's not already
    if found_close_col != 'Close':
        df = df.rename(columns={found_close_col: 'Close'})
        print(f"Renamed '{found_close_col}' to 'Close'.")

    # Ensure other OHLC columns are correctly named and capitalized
    # Create a mapping for common variations
    ohlc_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume'
    }
    current_cols = [col.lower() for col in df.columns]

    for lower_col, target_col in ohlc_mapping.items():
        if lower_col in current_cols and lower_col != target_col.lower():
            df = df.rename(columns={df.columns[current_cols.index(lower_col)]: target_col})
            print(f"Standardized column name: '{df.columns[current_cols.index(lower_col)]}' to '{target_col}'.")
            current_cols = [col.lower() for col in df.columns]  # Update after rename

    # --- FIX END ---

    # Calculate RSI and add to DataFrame
    # Now we are certain 'Close' column exists and is correctly named
    df['RSI'] = ta.rsi(df['Close'], length=14)

    df['SMA_25'] = ta.sma(df['Close'], length=25)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_100'] = ta.sma(df['Close'], length=100)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    # Use the df.ta accessor to automatically add the MACD columns
    df.ta.macd(append=True)
    # Calculate VWAP
    df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df = df.reset_index()
    return df


def get_market_close_prices(tickers, start_date, end_date):
    """
    Downloads, processes, and returns a DataFrame of daily close prices for given tickers.

    Args:
        tickers (list): A list of ticker symbols for Yahoo Finance.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing the daily close prices,
                          with cleaned column names.
    """
    # Download the historical data using the provided parameters
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d")
    # Select only the 'Close' price column for all tickers
    close_prices = data.loc[:, 'Close']

    # Define a mapping for cleaner column names
    rename_map = {
        '^VIX': 'VIX',
        'DX-Y.NYB': 'DXY',
        'SPY': 'SPY',
        'GC=F': 'XAU'
    }

    # Rename columns using the mapping
    close_prices = close_prices.rename(columns=rename_map)
    close_prices = close_prices.reset_index()

    return close_prices


if __name__ == '__main__':
    # --- Step 1: Define parameters ---
    ticker = 'TSLA'
    start_date = '2010-01-01'
    end_date = date.today().isoformat()

    # --- Step 2: Get the main stock data with indicators ---
    df_with_indicators = get_stock_data_with_indicators(ticker, start_date, end_date)

    # --- Step 3: Get the market data (using the correct ticker for WTI) ---
    # IMPORTANT: The ticker for WTI is 'CL=F', not 'WTI'
    market_tickers = ['^VIX', 'DX-Y.NYB', 'SPY', 'CL=F']
    market_data_df = get_market_close_prices(market_tickers, start_date, end_date)

    # We also need to rename 'CL=F' to 'WTI' for the final output
    market_data_df = market_data_df.rename(columns={'CL=F': 'WTI'})

    # --- Step 4: MERGE FIRST to combine all data into one DataFrame ---
    df_merged = pd.merge(df_with_indicators, market_data_df, on='Date', how='inner')

    # --- Step 5: NOW define the columns you want and select them ---
    macd_cols = [col for col in df_merged.columns if col.startswith('MACD_')]

    # This is the complete list of columns you want in your final file
    output_cols = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP',
        'SMA_25', 'SMA_50', 'SMA_100', 'SMA_200', 'WTI', 'VIX', 'DXY', 'SPY'
    ] + macd_cols

    # Create the final DataFrame for saving by selecting columns from the MERGED data
    df_to_save = df_merged[output_cols]

    # --- Step 6: Save the final DataFrame to a CSV file ---
    output_path = r'C:\My Documents\Mics\Logs\tsla_daily.csv'
    df_to_save.to_csv(output_path, index=False)

    # Print a confirmation and the tail of the data that was saved
    print(f"Successfully saved data to {output_path}")
    print(df_to_save.tail())
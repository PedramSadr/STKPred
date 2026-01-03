import os
import sys
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import date, datetime, timedelta

# Configuration
OUTPUT_FILE = r'C:\My Documents\Mics\Logs\tsla_daily.csv'
DEFAULT_START_DATE = '2010-01-01'


def get_last_recorded_date(file_path):
    """
    Checks the CSV for the last existing date.
    Returns a datetime object or None if file doesn't exist.
    """
    if not os.path.exists(file_path):
        return None

    try:
        # Read only the 'Date' column to be fast
        df = pd.read_csv(file_path, usecols=['Date'])
        if df.empty:
            return None

        last_date_str = df['Date'].max()
        return pd.to_datetime(last_date_str).date()
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}")
        return None


def download_and_clean(ticker, start_date, end_date):
    """
    Downloads raw data and cleans column names.
    Does NOT calculate indicators yet.
    """
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        return pd.DataFrame()

    # 1. Flatten MultiIndex Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]

    # 2. Rename specific ticker suffixes (e.g., Close_TSLA -> Close)
    for col in df.columns:
        if col.endswith(f"_{ticker}"):
            new_col = col.replace(f"_{ticker}", "")
            df = df.rename(columns={col: new_col})

    # 3. Standardize 'Close' and OHLC
    # Use case-insensitive matching to find 'Close' or 'Adj Close'
    found_close = None
    for col in df.columns:
        if 'adj close' in col.lower():
            found_close = col
            break
        if 'close' == col.lower():
            found_close = col

    if found_close:
        df = df.rename(columns={found_close: 'Close'})

    ohlc_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
    for col in df.columns:
        if col.lower() in ohlc_map:
            df = df.rename(columns={col: ohlc_map[col.lower()]})

    df = df.reset_index()
    return df


def calculate_indicators(df):
    """
    Calculates technical indicators on the full dataset.
    """
    if df.empty: return df

    # Ensure Date is datetime for sorting
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    # Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_25'] = ta.sma(df['Close'], length=25)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_100'] = ta.sma(df['Close'], length=100)
    df['SMA_200'] = ta.sma(df['Close'], length=200)

    # MACD
    df.ta.macd(append=True)

    # VWAP
    if {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
        df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])

    return df


if __name__ == '__main__':
    # 1. Determine Start Date
    last_date = get_last_recorded_date(OUTPUT_FILE)
    today = date.today()

    if last_date:
        # If we have data, only fetch recent history + buffer for indicators
        # Buffer: 300 days ensures SMA_200 is accurate for the new rows
        start_date = (last_date - timedelta(days=300)).isoformat()
        print(f"Existing data found up to {last_date}. Smart-fetching from {start_date}...")
    else:
        start_date = DEFAULT_START_DATE
        print("No existing data. Performing full download...")

    end_date = today.isoformat()

    # 2. Download Data (TSLA + Market)
    tsla_df = download_and_clean('TSLA', start_date, end_date)

    market_tickers = ['^VIX', 'DX-Y.NYB', 'SPY', 'CL=F']
    market_df = yf.download(market_tickers, start=start_date, end=end_date, interval="1d")

    # Clean Market Data
    if 'Close' in market_df.columns and isinstance(market_df.columns, pd.MultiIndex):
        # Handle yfinance multi-index output for multiple tickers
        market_df = market_df['Close'].reset_index()
    elif 'Close' in market_df.columns:
        market_df = market_df[['Date', 'Close']].reset_index(drop=True)

    # Rename Market Cols
    rename_map = {'^VIX': 'VIX', 'DX-Y.NYB': 'DXY', 'SPY': 'SPY', 'CL=F': 'WTI'}
    market_df = market_df.rename(columns=rename_map)

    # 3. Merge & Process
    # We merge on Date to align market data with TSLA data
    if 'Date' in tsla_df.columns and 'Date' in market_df.columns:
        # Ensure date types match
        tsla_df['Date'] = pd.to_datetime(tsla_df['Date'])
        market_df['Date'] = pd.to_datetime(market_df['Date'])

        # Merge
        merged_new = pd.merge(tsla_df, market_df, on='Date', how='left')
    else:
        merged_new = tsla_df

    # 4. Combine with Historical Data (if exists)
    if last_date and os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])

        # Filter new data to only rows AFTER the last recorded date
        # (We downloaded a buffer, but we only append the truly new days)
        # Actually: To be safe with indicators, we recalc on everything, then save.

        # Concatenate: Old Data + New Download
        # We use drop_duplicates to handle the 300-day overlap we fetched
        full_df = pd.concat([existing_df, merged_new])
        full_df = full_df.drop_duplicates(subset=['Date'], keep='last')
    else:
        full_df = merged_new

    # 5. Calculate Indicators (On the full Combined Dataset)
    # This ensures today's SMA_200 considers the data from 200 days ago
    final_df = calculate_indicators(full_df)

    # 6. Save
    # Define Column Order
    macd_cols = [c for c in final_df.columns if c.startswith('MACD_')]
    base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP',
                 'SMA_25', 'SMA_50', 'SMA_100', 'SMA_200', 'WTI', 'VIX', 'DXY', 'SPY']

    # Filter columns that actually exist
    cols_to_save = [c for c in base_cols + macd_cols if c in final_df.columns]

    final_df = final_df[cols_to_save]

    # Sort by date before saving
    final_df = final_df.sort_values('Date')
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Successfully updated {OUTPUT_FILE}")
    print(f"Total Records: {len(final_df)}")
    print("Last 3 Records:\n", final_df.tail(3)[['Date', 'Close', 'SMA_200']])
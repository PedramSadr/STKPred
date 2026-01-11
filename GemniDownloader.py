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
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, usecols=['Date'])
        if df.empty: return None
        # Ensure we parse correctly
        return pd.to_datetime(df['Date'].max()).date()
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}")
        return None


def download_and_clean(ticker, start_date, end_date):
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")

    # Download with auto_adjust to handle splits/dividends
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        return pd.DataFrame()

    # FIX 1: Handle yfinance MultiIndex columns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Keep the price level, drop the ticker level
        df.columns = df.columns.get_level_values(0)

    # Reset index to make 'Date' a column
    df = df.reset_index()

    # FIX 2: STRIP TIMEZONES (Crucial for merging with CSV)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Rename standard columns to Title Case
    rename_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume', 'adj close': 'Close'
    }
    df = df.rename(columns=lambda x: rename_map.get(x.lower(), x))

    return df


def calculate_indicators(df):
    if df.empty: return df

    # Sort by date for indicators
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    # Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_25'] = ta.sma(df['Close'], length=25)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_100'] = ta.sma(df['Close'], length=100)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df.ta.macd(append=True)

    # Robust VWAP
    if {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
        try:
            df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
        except:
            df['VWAP'] = 0.0

    return df


if __name__ == '__main__':
    # 1. Determine Start Date
    last_date = get_last_recorded_date(OUTPUT_FILE)
    today = date.today()

    if last_date:
        # Fetch a buffer (300 days) to ensuring moving averages (SMA200) are accurate
        start_date = (last_date - timedelta(days=300)).isoformat()
        print(f"Existing data found up to {last_date}. Smart-fetching from {start_date}...")
    else:
        start_date = DEFAULT_START_DATE
        print("No existing data. Performing full download...")

    end_date = today.isoformat()

    # 2. Download TSLA
    tsla_df = download_and_clean('TSLA', start_date, end_date)

    # 3. Download Market Data
    market_tickers = ['^VIX', 'DX-Y.NYB', 'SPY', 'CL=F']
    print("Fetching Market Data...")
    market_df = yf.download(market_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Clean Market Data Structure
    if isinstance(market_df.columns, pd.MultiIndex):
        # Extract just the 'Close' prices
        try:
            market_df = market_df['Close']
        except KeyError:
            pass  # Fallback if structure differs

    market_df = market_df.reset_index()

    # FIX 3: Strip Timezones from Market Data too
    if 'Date' in market_df.columns:
        market_df['Date'] = pd.to_datetime(market_df['Date']).dt.tz_localize(None)

    market_df = market_df.rename(columns={'^VIX': 'VIX', 'DX-Y.NYB': 'DXY', 'SPY': 'SPY', 'CL=F': 'WTI'})

    # 4. Merge TSLA + Market
    if 'Date' in tsla_df.columns and 'Date' in market_df.columns:
        merged_new = pd.merge(tsla_df, market_df, on='Date', how='left')
    else:
        merged_new = tsla_df

    # 5. Combine with Historical CSV
    if last_date and os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])

        full_df = pd.concat([existing_df, merged_new])
        # Drop duplicates based on Date (now safe because timezones are gone)
        full_df = full_df.drop_duplicates(subset=['Date'], keep='last')
    else:
        full_df = merged_new

    # 6. Save
    final_df = calculate_indicators(full_df)
    final_df = final_df.dropna(subset=['Date']).sort_values('Date')

    # Define Column Order
    macd_cols = [c for c in final_df.columns if c.startswith('MACD_')]
    base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP',
                 'SMA_25', 'SMA_50', 'SMA_100', 'SMA_200', 'WTI', 'VIX', 'DXY', 'SPY']

    cols_to_save = [c for c in base_cols + macd_cols if c in final_df.columns]

    final_df[cols_to_save].to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully updated {OUTPUT_FILE}")
    print(f"Newest Record: {final_df['Date'].max().date()}")
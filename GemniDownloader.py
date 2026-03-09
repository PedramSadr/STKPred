import os
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import date, datetime, timedelta

# Configuration - Updated Path for TFT
OUTPUT_FILE = r'C:\My Documents\Mics\Logs\tft\stock_daily.csv'
DEFAULT_START_DATE = '2010-01-01'


def get_last_recorded_date(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, usecols=['Date'])
        if df.empty: return None
        return pd.to_datetime(df['Date'].max()).date()
    except Exception as e:
        print(f"Warning: Could not read existing file: {e}")
        return None


def download_and_clean(ticker, start_date, end_date):
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    rename_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume', 'adj close': 'Close'
    }
    df = df.rename(columns=lambda x: rename_map.get(x.lower(), x))

    return df


def calculate_zscore(series, window=60):
    """Helper to calculate rolling Z-score with divide-by-zero protection."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    # Replace 0 std with NaN to prevent infinity, then fill resulting NaNs with 0
    z = (series - mean) / std.replace(0, np.nan)
    return z.fillna(0)


def calculate_indicators(df):
    if df.empty: return df

    if 'Date' in df.columns:
        df = df.drop_duplicates(subset=['Date'], keep='last')
        df = df.sort_values('Date')
    df.reset_index(drop=True, inplace=True)

    # 1. Base Technical Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_25'] = ta.sma(df['Close'], length=25)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_100'] = ta.sma(df['Close'], length=100)
    df['SMA_200'] = ta.sma(df['Close'], length=200)

    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd_df is not None and 'MACD_12_26_9' in macd_df.columns:
        df['MACD_12_26_9'] = macd_df['MACD_12_26_9']
    else:
        df['MACD_12_26_9'] = np.nan

    # 2. VWAP
    if {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
        try:
            df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
        except:
            df['VWAP'] = 0.0

    # 3. ATR & NATR
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['NATR_14'] = df['ATR_14'] / df['Close']
    df['NATR_14_CHG_1D'] = df['NATR_14'].diff()
    df['NATR_14_Z60'] = calculate_zscore(df['NATR_14'], 60)

    # 4. Realized Volatility (Annualized)
    tsla_log_ret = np.log(df['Close']).diff()
    df['RV_10D'] = tsla_log_ret.rolling(10).std() * np.sqrt(252)
    df['RV_20D'] = tsla_log_ret.rolling(20).std() * np.sqrt(252)
    df['RV_60D'] = tsla_log_ret.rolling(60).std() * np.sqrt(252)

    # ---> NEW: Realized Volatility Momentum (Regime Sensor) <---
    df['RV_REGIME'] = df['RV_10D'] / df['RV_60D']
    df['RV_REGIME_Z60'] = calculate_zscore(df['RV_REGIME'], 60)

    # ==========================================
    # 5. VOLATILITY RISK PREMIUM (VRP) & TERM STRUCTURE
    # ==========================================
    if 'VIX' in df.columns and 'SPY' in df.columns:
        # Scale VIX to a decimal
        vix_decimal = df['VIX'] / 100.0

        # Calculate SPY RV
        spy_log_ret = np.log(df['SPY']).diff()
        df['SPY_RV_20D'] = spy_log_ret.rolling(20).std() * np.sqrt(252)

        # MACRO VRP
        df['MACRO_VRP_20D'] = vix_decimal - df['SPY_RV_20D']
        df['MACRO_VRP_20D_Z60'] = calculate_zscore(df['MACRO_VRP_20D'], 60)

        # TSLA Market Vol Spread
        df['TSLA_MARKET_VOL_SPREAD_20D'] = vix_decimal - df['RV_20D']
        df['TSLA_MARKET_VOL_SPREAD_20D_Z60'] = calculate_zscore(df['TSLA_MARKET_VOL_SPREAD_20D'], 60)

    # Macro Volatility Term Structure
    if 'VIX' in df.columns and 'VIX3M' in df.columns:
        df['VIX_TERM_RATIO'] = df['VIX'] / df['VIX3M']
        df['VIX_TERM_RATIO_Z60'] = calculate_zscore(df['VIX_TERM_RATIO'], 60)

    # 6. SPY & DXY & WTI Derived Columns
    if 'SPY' in df.columns:
        df['SPY_RET_1D'] = df['SPY'].pct_change()
        df['SPY_RET_5D'] = df['SPY'].pct_change(5)
        df['SPY_Z60'] = calculate_zscore(df['SPY'], 60)

    if 'DXY' in df.columns:
        df['DXY_CHG_1D'] = df['DXY'].pct_change()
        df['DXY_Z60'] = calculate_zscore(df['DXY_CHG_1D'], 60)

    if 'WTI' in df.columns:
        df['WTI_RET_1D'] = df['WTI'].pct_change()
        df['WTI_RET_5D'] = df['WTI'].pct_change(5)
        df['WTI_Z60'] = calculate_zscore(df['WTI'], 60)

    if 'VIX' in df.columns:
        df['VIX_CHG_1D'] = df['VIX'].diff()
        df['VIX_Z60'] = calculate_zscore(df['VIX'], 60)

    # 7. Credit Spread Proxy (HYG / LQD)
    if 'HYG' in df.columns and 'LQD' in df.columns:
        df['CREDIT_RATIO'] = np.log(df['HYG'] / df['LQD'])
        df['CREDIT_RATIO_CHG_1D'] = df['CREDIT_RATIO'].diff()
        df['CREDIT_RATIO_Z60'] = calculate_zscore(df['CREDIT_RATIO'], 60)

    # 8. Yield Curve Data (TNX & US5Y)
    if 'TNX' in df.columns:
        df['TNX_CHG_1D'] = df['TNX'].diff()
        df['TNX_Z60'] = calculate_zscore(df['TNX'], 60)

    if 'US5Y' in df.columns:
        df['US5Y_CHG_1D'] = df['US5Y'].diff()
        df['US5Y_Z60'] = calculate_zscore(df['US5Y'], 60)

    if 'TNX' in df.columns and 'US5Y' in df.columns:
        df['CURVE_5S10S'] = df['TNX'] - df['US5Y']

    # 9. Volatility Matrix (Base metrics)
    if 'VIX3M' in df.columns:
        df['VIX3M_CHG_1D'] = df['VIX3M'].diff()
        df['VIX3M_Z60'] = calculate_zscore(df['VIX3M'], 60)

    if 'VVIX' in df.columns:
        df['VVIX_CHG_1D'] = df['VVIX'].diff()
        df['VVIX_Z60'] = calculate_zscore(df['VVIX'], 60)

    if 'SKEW' in df.columns:
        df['SKEW_CHG_1D'] = df['SKEW'].diff()
        df['SKEW_Z60'] = calculate_zscore(df['SKEW'], 60)

    # =====================================================================
    # 10. SHIFT T-1 FEATURES TO PREVENT 9:50 AM DATA LEAKAGE
    # =====================================================================
    shift_cols = [
        'RSI', 'SMA_25', 'SMA_50', 'SMA_100', 'SMA_200', 'MACD_12_26_9',
        'ATR_14', 'NATR_14', 'NATR_14_CHG_1D', 'NATR_14_Z60',
        'WTI', 'VIX', 'DXY', 'SPY',
        'SPY_RET_1D', 'SPY_RET_5D', 'SPY_Z60',
        'DXY_CHG_1D', 'DXY_Z60',
        'WTI_RET_1D', 'WTI_RET_5D', 'WTI_Z60',
        'VIX_CHG_1D', 'VIX_Z60',
        'HYG', 'LQD', 'CREDIT_RATIO', 'CREDIT_RATIO_CHG_1D', 'CREDIT_RATIO_Z60',
        'TNX', 'TNX_CHG_1D', 'TNX_Z60',
        'US5Y', 'US5Y_CHG_1D', 'US5Y_Z60', 'CURVE_5S10S',
        'VIX3M', 'VIX3M_CHG_1D', 'VIX3M_Z60',
        'VVIX', 'VVIX_CHG_1D', 'VVIX_Z60',
        'SKEW', 'SKEW_CHG_1D', 'SKEW_Z60',
        'RV_10D', 'RV_20D', 'RV_60D',
        'RV_REGIME', 'RV_REGIME_Z60',  # <-- Added Vol Momentum Shift
        'SPY_RV_20D', 'MACRO_VRP_20D', 'MACRO_VRP_20D_Z60',
        'TSLA_MARKET_VOL_SPREAD_20D', 'TSLA_MARKET_VOL_SPREAD_20D_Z60',
        'VIX_TERM_RATIO', 'VIX_TERM_RATIO_Z60'
    ]

    for c in shift_cols:
        if c in df.columns:
            df[c] = df[c].shift(1)

    return df


if __name__ == '__main__':
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Determine Start Date
    last_date = get_last_recorded_date(OUTPUT_FILE)
    today = date.today()

    if last_date:
        start_date = (last_date - timedelta(days=300)).isoformat()
        print(f"Existing data found up to {last_date}. Smart-fetching from {start_date}...")
    else:
        start_date = DEFAULT_START_DATE
        print("No existing data. Performing full download...")

    end_date = today.isoformat()

    # 2. Download TSLA
    tsla_df = download_and_clean('TSLA', start_date, end_date)

    # 3. Download Market Data
    market_tickers = ['^VIX', 'DX-Y.NYB', 'SPY', 'CL=F', 'HYG', 'LQD', '^TNX', '^FVX', '^VIX3M', '^VVIX', '^SKEW']
    print("Fetching Market Data...")
    market_df = yf.download(market_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Forward Fill Fix
    market_df = market_df.ffill()

    if isinstance(market_df.columns, pd.MultiIndex):
        try:
            market_df = market_df['Close']
        except KeyError:
            pass

    market_df = market_df.reset_index()

    if 'Date' in market_df.columns:
        market_df['Date'] = pd.to_datetime(market_df['Date']).dt.tz_localize(None)

    # Map Yahoo tickers to our PDF column names
    rename_market_map = {
        '^VIX': 'VIX',
        'DX-Y.NYB': 'DXY',
        'SPY': 'SPY',
        'CL=F': 'WTI',
        'HYG': 'HYG',
        'LQD': 'LQD',
        '^TNX': 'TNX',
        '^FVX': 'US5Y',
        '^VIX3M': 'VIX3M',
        '^VVIX': 'VVIX',
        '^SKEW': 'SKEW'
    }
    market_df = market_df.rename(columns=rename_market_map)

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
        full_df = full_df.drop_duplicates(subset=['Date'], keep='last')
    else:
        full_df = merged_new

    # 6. Calculate all features and cleanly sort
    final_df = calculate_indicators(full_df)
    final_df = final_df.dropna(subset=['Date']).sort_values('Date')

    # 7. Strictly enforce the layout order
    final_columns_order = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'VWAP',
        'SMA_25', 'SMA_50', 'SMA_100', 'SMA_200',
        'WTI', 'VIX', 'DXY', 'SPY', 'MACD_12_26_9',
        'ATR_14', 'NATR_14', 'NATR_14_CHG_1D', 'NATR_14_Z60',
        'SPY_RET_1D', 'SPY_RET_5D', 'SPY_Z60',
        'DXY_CHG_1D', 'DXY_Z60',
        'WTI_RET_1D', 'WTI_RET_5D', 'WTI_Z60',
        'VIX_CHG_1D', 'VIX_Z60',
        'HYG', 'LQD', 'CREDIT_RATIO', 'CREDIT_RATIO_CHG_1D', 'CREDIT_RATIO_Z60',
        'TNX', 'TNX_CHG_1D', 'TNX_Z60',
        'US5Y', 'US5Y_CHG_1D', 'US5Y_Z60', 'CURVE_5S10S',
        'VIX3M', 'VIX3M_CHG_1D', 'VIX3M_Z60',
        'VVIX', 'VVIX_CHG_1D', 'VVIX_Z60',
        'SKEW', 'SKEW_CHG_1D', 'SKEW_Z60',
        'RV_10D', 'RV_20D', 'RV_60D',
        'RV_REGIME', 'RV_REGIME_Z60',  # <-- Added Vol Momentum Export
        'SPY_RV_20D', 'MACRO_VRP_20D', 'MACRO_VRP_20D_Z60',
        'TSLA_MARKET_VOL_SPREAD_20D', 'TSLA_MARKET_VOL_SPREAD_20D_Z60',
        'VIX_TERM_RATIO', 'VIX_TERM_RATIO_Z60'
    ]

    # Save only the columns that exist, ordered correctly
    cols_to_save = [c for c in final_columns_order if c in final_df.columns]

    # Global sanity check (Replaces infs and NaNs)
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.fillna(0)

    final_df[cols_to_save].to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated full Dataset -> {OUTPUT_FILE}")
    print(f"Newest Record: {final_df['Date'].max().date()}")
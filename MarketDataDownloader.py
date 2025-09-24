import yfinance as yf
import pandas as pd

# Define the tickers and their Yahoo Finance symbols
# VIX: ^VIX, DXY: DX-Y.NYB, SPY: SPY, US2Y: ^UST2Y, Gold: GC=F
# tickers = ['^VIX', 'DX-Y.NYB', 'SPY', '^UST2Y', 'GC=F']
tickers = ['^VIX']

# Define the date range
start_date = '2010-07-01'
end_date = '2025-09-15'

# Download the historical data
# The result is a DataFrame with MultiIndex columns (e.g., ('Close', 'SPY'))
data = yf.download(tickers, start=start_date, end=end_date)

# Select only the 'Close' price for all tickers
close_prices = data.loc[:, 'Close']

# Rename columns for better readability
close_prices = close_prices.rename(columns={
    '^VIX': 'VIX',
    'DX-Y.NYB': 'DXY',
    'SPY': 'SPY',
    '^UST2Y': 'US2Y',
    'GC=F': 'XAU'
})
close_prices = close_prices.reset_index()
print("--- First 5 Rows ---")
print(close_prices.head())
print("\n--- Last 5 Rows ---")
print(close_prices.tail())

close_prices.to_csv(r'C:\My Documents\Mics\Logs\market_data.csv', index=False)
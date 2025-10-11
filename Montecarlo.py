import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Read CSV file and prepare data
df = pd.read_csv(r'C:\My Documents\Mics\Logs\tsla_monte.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')

# 2. Set up simulation parameters
target_date = pd.to_datetime('2025-10-24')
start_date = df.index[-1]
n_days = np.busday_count(start_date.date(), target_date.date())
sims = 20000
start_price = df['Close'][-1]

# 3. Calculate historical log returns for the model
log_returns = np.log(1 + df['Close'].pct_change())
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var)
stdev = log_returns.std()

# 4. Run the NumPy-based Monte Carlo simulation
# Create random values
np.random.seed(42)
Z = np.random.standard_normal((n_days, sims))
# Calculate daily returns
daily_returns = np.exp(drift + stdev * Z)

# Create an array to hold the price paths
price_paths = np.zeros_like(daily_returns)
price_paths[0] = start_price

# Generate the price paths
for t in range(1, n_days):
    price_paths[t] = price_paths[t - 1] * daily_returns[t]

# Convert to a DataFrame for easy analysis and plotting
simulated_prices = pd.DataFrame(price_paths)

# 5. Analyze and plot the results
last_day_prices = simulated_prices.iloc[-1]
predicted_mean = last_day_prices.mean()
print(f"Predicted TSLA close price for {target_date.date()}: ${predicted_mean:.2f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(simulated_prices, color='gray', alpha=0.1)
plt.title(f'Monte Carlo Simulation for TSLA Close Price to {target_date.date()}')
plt.ylabel('Simulated Close Price')
plt.xlabel('Days Ahead')
plt.axhline(predicted_mean, color='red', linestyle='--', label=f'Predicted Mean: ${predicted_mean:.2f}')
plt.legend()
plt.show()
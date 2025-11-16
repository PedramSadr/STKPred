import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os

csv_path = r'C:\My Documents\Mics\Logs\tsla_daily.csv'
plots_dir = r'C:\My Documents\Mics\Logs\Plots'
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(csv_path, parse_dates=['Date'])

for idx, row in df.iterrows():
    single_df = pd.DataFrame({
        'Open': [row['Open']],
        'High': [row['High']],
        'Low': [row['Low']],
        'Close': [row['Close']]
    }, index=[row['Date']])
    single_df.index = pd.to_datetime(single_df.index)
    date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
    plot_path = os.path.join(plots_dir, f'{date_str}.png')

    fig, axlist = mpf.plot(
        single_df, type='candle', style='charles', title=date_str,
        ylabel='Price', returnfig=True
    )
    ax = axlist[0]
    ax.set_xticklabels([])
    fig.savefig(plot_path)
    plt.close(fig)
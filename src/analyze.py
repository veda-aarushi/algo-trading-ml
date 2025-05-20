# src/analyze.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Files
CLUSTER_CSV   = os.path.join(DATA_DIR, 'backtest_cluster0_daily.csv')
SENTIM_CSV    = os.path.join(DATA_DIR, 'backtest_sentiment_daily.csv')
INTRA_CSV     = os.path.join(DATA_DIR, 'backtest_intraday.csv')

# Load data
cluster = pd.read_csv(CLUSTER_CSV, index_col=0, parse_dates=True)
cluster.index.name = 'Date'
sent    = pd.read_csv(SENTIM_CSV, index_col=0, parse_dates=True)
sent.index.name = 'Date'
intra   = pd.read_csv(INTRA_CSV, index_col=0, parse_dates=True)
intra.index.name = 'Date'

# Standardize column names (in case they differ)
cluster_return_col = [c for c in cluster.columns if 'cluster' in c or 'return' in c][0]
sentiment_col      = [c for c in sent.columns if 'sentiment_return' in c or 'return' in c][0]
intraday_col       = [c for c in intra.columns if 'strategy_ret' in c or 'return' in c][0]

# Compute cumulative returns
cluster['cum_ret'] = (1 + cluster[cluster_return_col]).cumprod() - 1
sent['cum_ret']    = (1 + sent[sentiment_col]).cumprod() - 1
intra['cum_ret']   = (1 + intra[intraday_col]).cumprod() - 1

# Sharpe ratios (daily)
sharpe = lambda x: x.mean() / x.std() * np.sqrt(252)
stats = {
    'Unsupervised': sharpe(cluster[cluster_return_col]),
    'Sentiment':    sharpe(sent[sentiment_col]),
    'Intraday':     sharpe(intra[intraday_col]),
}

# Plot
plt.figure(figsize=(10,6))
plt.plot(cluster.index, cluster['cum_ret'], label='Cluster Strategy')
plt.plot(sent.index,    sent['cum_ret'],    label='Sentiment Strategy')
plt.plot(intra.index,   intra['cum_ret'],   label='Intraday Strategy')
plt.legend()
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
out_png = os.path.join(DATA_DIR, 'strategies_cumulative.png')
plt.savefig(out_png)
print(f"Saved performance plot to {out_png}")

# Print stats
print("Sharpe Ratios:")
for name, val in stats.items():
    print(f"  {name}: {val:.2f}")

if __name__ == '__main__':
    plt.show()

# src/backtest.py

import os
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

BASE_DIR   = os.path.dirname(__file__)
RAW_CSV    = os.path.join(BASE_DIR, os.pardir, "data", "raw", "sp500_prices.csv")
CLUSTERED  = os.path.join(BASE_DIR, os.pardir, "data", "processed", "features_clustered.csv")
OUT_DIR    = os.path.join(BASE_DIR, os.pardir, "data", "processed")
CLUSTER_ID = 0  # change to target different cluster


def backtest_cluster(cluster_id=CLUSTER_ID):
    # Load prices and cluster assignments
    raw = pd.read_csv(RAW_CSV, index_col=["Date","Ticker"], parse_dates=["Date"])
    feats = pd.read_csv(CLUSTERED, index_col=["Date","Ticker"], parse_dates=["Date"])

    # Choose adjusted close if available, else close
    price_col = 'adj close' if 'adj close' in raw.columns else 'close'
    price = raw[price_col].unstack('Ticker')

    # Compute daily returns
    rets = price.pct_change().dropna(how='all')

    # Collect daily portfolio returns
    port_returns = []

    # Rebalance each month
    for date, group in feats.groupby(level='Date'):
        tickers = group[group['cluster'] == cluster_id].index.get_level_values('Ticker').tolist()
        if not tickers:
            continue

        train_prices = price.loc[:date, tickers]
        mu = expected_returns.mean_historical_return(train_prices, frequency=252)
        S  = risk_models.sample_cov(train_prices, frequency=252)

        # Optimize for maximum Sharpe, fallback to equal weights if error
        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
        except ValueError:
            print(f"  Warning: no expected return > risk-free on {date.date()}, using equal weights")
            weights = {t: 1/len(tickers) for t in tickers}

        # Simulate next month's daily returns
        start = date + pd.offsets.MonthBegin(1)
        end   = date + pd.offsets.MonthEnd(1)
        mask  = (rets.index >= start) & (rets.index <= end)
        rets_month = rets.loc[mask, tickers]
        port_ret = rets_month.dot(pd.Series(weights))

        df_temp = pd.DataFrame({
            'Date':    port_ret.index,
            'return':  port_ret.values,
            'cluster': cluster_id
        })
        port_returns.append(df_temp)

        print(f"Backtested month {date.date()}: {len(tickers)} tickers -> {port_ret.shape[0]} days of returns")

    result = pd.concat(port_returns).set_index('Date')
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Running backtest for cluster {CLUSTER_ID}…")
    df = backtest_cluster()
    out_file = os.path.join(OUT_DIR, f"backtest_cluster{CLUSTER_ID}_daily.csv")
    df.to_csv(out_file)
    print(f"Saved backtest results to {out_file}")


if __name__ == "__main__":
    main()

# src/twitter_sentiment.py

import os
import pandas as pd
import numpy as np

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
# Primary location for your custom CSV
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "sentiment_data.csv")
# Fallback if file lives under /mnt/data
if not os.path.exists(RAW_CSV):
    alt = "/mnt/data/sentiment_data.csv"
    if os.path.exists(alt):
        RAW_CSV = alt
    else:
        raise FileNotFoundError(f"Could not find sentiment_data.csv at {RAW_CSV} or {alt}")

PRICE_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "sp500_prices.csv")
OUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")


def load_sentiment_data(path=RAW_CSV) -> pd.DataFrame:
    """
    Load the custom Twitter dataset with columns:
      date, symbol, twitterPosts, twitterComments, twitterLikes, twitterImpressions, etc.
    Renames to Date, Ticker, and maps interactions to engagement.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    # Rename columns
    df = df.rename(columns={
        'date': 'Date',
        'symbol': 'Ticker',
        'twitterLikes': 'likes',
        'twitterPosts': 'retweets',
        'twitterComments': 'replies',
        'twitterImpressions': 'quotes'
    })
    # Compute total engagement
    df['engagement'] = df[['likes', 'retweets', 'replies', 'quotes']].sum(axis=1)
    return df[['Date', 'Ticker', 'engagement']]


def compute_monthly_engagement(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate engagement to month-end by ticker.
    Returns a Series indexed by (MonthEnd, Ticker).
    """
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp('M')
    monthly = (
        df
        .groupby(['Month', 'Ticker'])['engagement']
        .mean()
        .rename('engagement')
    )
    return monthly


def backtest_sentiment(monthly: pd.Series):
    """
    For each month:
      - Select top 20 tickers by average engagement
      - Form equal-weight portfolio
      - Compute daily returns for next month
    Save all daily returns to CSV.
    """
    # Load price returns
    px = pd.read_csv(
        PRICE_CSV,
        index_col=['Date', 'Ticker'],
        parse_dates=['Date']
    )
    prices = px['close'].unstack('Ticker')
    rets = prices.pct_change().dropna(how='all')

    results = []
    for month, grp in monthly.groupby(level=0):
        # Top 20 by engagement (MultiIndex: Month, Ticker)
        sorted_ = grp.sort_values(ascending=False)
        # Extract tickers and filter to available price columns
        cand = sorted_.head(20).index.get_level_values(1).tolist()
        longs = [t for t in cand if t in rets.columns]
        if not longs:
            print(f"[WARN] No valid tickers for {month.date()} after filtering vs price data.")
            continue

        # Next-month date range
        start = month + pd.offsets.MonthBegin(1)
        end   = month + pd.offsets.MonthEnd(1)
        mask  = (rets.index >= start) & (rets.index <= end)
        rets_m = rets.loc[mask, longs]

        # Equal weights
        w = np.repeat(1/len(longs), len(longs))
        port_ret = rets_m.dot(w)

        dfm = pd.DataFrame({'Date': port_ret.index, 'sentiment_return': port_ret.values})
        results.append(dfm)
        print(f"Sentiment backtest {month.date()}: {len(longs)} tickers → {port_ret.shape[0]} days")

    allr = pd.concat(results).set_index('Date')
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, 'backtest_sentiment_daily.csv')
    allr.to_csv(out_csv)
    print(f"Saved sentiment backtest to {out_csv}")


def main():
    df = load_sentiment_data()
    monthly = compute_monthly_engagement(df)
    backtest_sentiment(monthly)


if __name__ == '__main__':
    main()

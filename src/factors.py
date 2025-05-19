# src/factors.py

import os
import pandas as pd
import datetime as dt
from pandas_datareader import data as web

# Paths
BASE_DIR     = os.path.dirname(__file__)
RAW_PRICES   = os.path.join(BASE_DIR, os.pardir, "data", "raw", "sp500_prices.csv")
FEATURES_CSV = os.path.join(BASE_DIR, os.pardir, "data", "processed", "features_monthly.csv")
OUT_DIR      = os.path.join(BASE_DIR, os.pardir, "data", "processed")
WINDOW       = 12  # 12-month rolling window for beta

def get_ff_factors(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Download Fama–French 3 factors (Mkt–RF, SMB, HML, RF).
    Returns a DataFrame indexed by month-end timestamps, in decimals.
    """
    ff = web.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)[0]
    # Convert index to month-end Timestamps
    ff.index = ff.index.to_timestamp()
    # Convert percentages to decimals
    return ff.div(100)

def compute_rolling_betas(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    prices: daily DataFrame with MultiIndex [Date, Ticker], with 'close'.
    factors: monthly DataFrame with columns ['Mkt-RF','SMB','HML','RF'].
    Returns: DataFrame of rolling betas, indexed by [Date, Ticker].
    """
    # 1) Build monthly returns
    monthly_close   = prices["close"].unstack("Ticker").resample("M").last()
    monthly_ret     = monthly_close.pct_change().dropna()

    # 2) Align factors to returns
    ff   = factors.reindex(monthly_ret.index).dropna()
    mkt  = ff["Mkt-RF"]
    rf   = ff["RF"]

    # 3) Excess returns
    excess_ret = monthly_ret.sub(rf, axis=0)

    # 4) Rolling covariance / variance → beta
    # cov(excess_ret, Mkt-RF) / var(Mkt-RF)
    cov = excess_ret.rolling(WINDOW).cov(mkt)
    var = mkt.rolling(WINDOW).var()
    betas = cov.div(var, axis=0)

    # 5) Melt to long form
    beta_long = betas.stack().rename("beta").reset_index()
    # beta_long columns: ['Date','Ticker','beta']
    return beta_long.set_index(["Date","Ticker"])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load inputs
    print("Loading raw prices…")
    prices = pd.read_csv(
        RAW_PRICES,
        index_col=["Date","Ticker"],
        parse_dates=["Date"]
    )

    print("Loading monthly features…")
    feats = pd.read_csv(
        FEATURES_CSV,
        index_col=["Date","Ticker"],
        parse_dates=["Date"]
    )

    # Download FF factors
    start = feats.index.get_level_values("Date").min()
    end   = feats.index.get_level_values("Date").max()
    print(f"Downloading Fama–French factors from {start.date()} to {end.date()}…")
    ff = get_ff_factors(start, end)

    # Compute betas
    print("Computing 12-month rolling betas…")
    beta_df = compute_rolling_betas(prices, ff)

    # Merge into features
    print("Merging betas into features…")
    feats = feats.reset_index().merge(
        beta_df.reset_index(),
        on=["Date","Ticker"],
        how="left"
    ).set_index(["Date","Ticker"])

    # Save
    out_path = os.path.join(OUT_DIR, "features_with_betas.csv")
    feats.to_csv(out_path)
    print(f"Saved features+betas to {out_path}")

if __name__ == "__main__":
    main()

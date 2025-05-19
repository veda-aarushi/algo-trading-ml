# src/features.py

import os
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD

BASE_DIR = os.path.dirname(__file__)
RAW_CSV = os.path.join(BASE_DIR, os.pardir, "data", "raw", "sp500_prices.csv")
OUT_DIR = os.path.join(BASE_DIR, os.pardir, "data", "processed")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily prices with MultiIndex [Date, Ticker].
    Output: monthly feature DataFrame of top 50 by dollar volume.
    """
    # 1) Garman–Klass volatility
    df["log_hl"] = np.log(df["high"] / df["low"])
    df["log_co"] = np.log(df["close"] / df["open"])
    df["gk_vol"] = 0.5 * df["log_hl"]**2 - (2 * np.log(2) - 1) * df["log_co"]**2

    # 2) RSI (14) — use transform to keep alignment
    df["rsi"] = (
        df
        .groupby(level="Ticker")["close"]
        .transform(lambda x: RSIIndicator(x, window=14).rsi())
    )

    # 3) Bollinger Bands (20, 2)
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()

    # 4) ATR (14)
    df["atr"] = (
        df
        .groupby(level="Ticker")
        .apply(lambda x: AverageTrueRange(
            high=x["high"], low=x["low"], close=x["close"], window=14
        ).average_true_range())
        .reset_index(level=0, drop=True)
    )

    # 5) MACD diff (12, 26, 9)
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_diff"] = macd.macd_diff()

    # 6) Dollar volume
    df["dollar_vol"] = df["close"] * df["volume"]

    # --- Monthly aggregation & top-50 filter ---
    df = df.reset_index()
    monthly = (
        df
        .groupby([pd.Grouper(key="Date", freq="M"), "Ticker"])
        .agg(
            gk_vol=("gk_vol", "mean"),
            rsi=("rsi", "last"),
            bb_mavg=("bb_mavg", "last"),
            bb_hband=("bb_hband", "last"),
            bb_lband=("bb_lband", "last"),
            atr=("atr", "mean"),
            macd_diff=("macd_diff", "last"),
            dollar_vol=("dollar_vol", "mean"),
        )
        .sort_values(["Date", "dollar_vol"], ascending=[True, False])
        .groupby(level=0)
        .head(50)
    )
    return monthly

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading raw prices…")
    prices = pd.read_csv(
        RAW_CSV,
        index_col=["Date", "Ticker"],
        parse_dates=["Date"]
    )
    print("Computing features…")
    feats = compute_features(prices)
    out_path = os.path.join(OUT_DIR, "features_monthly.csv")
    feats.to_csv(out_path)
    print(f"Saved features to {out_path}")

if __name__ == "__main__":
    main()

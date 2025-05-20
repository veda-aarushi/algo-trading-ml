# src/intraday.py

import os
import pandas as pd
from arch import arch_model
import numpy as np

# Paths
BASE_DIR      = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
RAW_DAILY     = os.path.join(PROJECT_ROOT, "data", "raw", "simulated_daily_data.csv")
RAW_INTRADAY  = os.path.join(PROJECT_ROOT, "data", "raw", "simulated_5min_data.csv")
OUT_DIR       = os.path.join(PROJECT_ROOT, "data", "processed")

# GARCH settings
ROLL_WINDOW   = 252  # days for rolling estimation
VOL_PRED_DAYS = 1    # forecast horizon (days)


def load_data():
    """Load daily and intraday CSVs into DataFrames."""
    # Daily data
    daily = pd.read_csv(
        RAW_DAILY,
        parse_dates=["Date"],
        index_col="Date"
    )
    # Intraday data: column is 'datetime'
    intraday = pd.read_csv(
        RAW_INTRADAY,
        parse_dates=["datetime"],
        index_col="datetime"
    )
    return daily, intraday


def predict_daily_volatility(daily: pd.DataFrame) -> pd.Series:
    """
    Fit a rolling GARCH(1,1) to percent returns and forecast next-day volatility.
    Returns a Series of predicted volatilities indexed by forecast date.
    """
    ret = daily["Close"].pct_change().dropna() * 100
    vol_preds = []
    dates = []

    for end_date in ret.index[ROLL_WINDOW:]:
        window = ret.loc[:end_date].iloc[-ROLL_WINDOW:]
        am = arch_model(window, vol="Garch", p=1, o=0, q=1, dist="normal")
        res = am.fit(disp="off")
        f = res.forecast(horizon=VOL_PRED_DAYS)
        vol = np.sqrt(f.variance.values[-1, -1])
        vol_preds.append(vol)
        dates.append(end_date + pd.Timedelta(days=1))

    return pd.Series(vol_preds, index=dates, name="pred_vol")


def compute_intraday_signal(intraday: pd.DataFrame) -> pd.Series:
    """
    Compute intraday momentum: last bar close / first bar open - 1.
    Returns a Series indexed by each date.
    """
    df = intraday.copy()
    df["Date"] = df.index.date
    # columns are lowercase 'open' and 'close'
    first = df.groupby("Date")["open"].first()
    last = df.groupby("Date")["close"].last()
    signal = (last / first - 1).rename("intraday_mom")
    return signal


def backtest(daily: pd.DataFrame, intraday: pd.DataFrame):
    """
    Merge daily vol forecasts and intraday signal, generate positions,
    and compute strategy returns.
    """
    # Forecast volatility
    vol_pred = predict_daily_volatility(daily)
    # Intraday momentum
    intr_sig = compute_intraday_signal(intraday)
    # Align on date index
    df = pd.concat([vol_pred, intr_sig], axis=1).dropna()

    # Position: long if intraday_mom > 0 and pred_vol > median
    median_vol = df["pred_vol"].median()
    df["pos"] = np.where((df["intraday_mom"] > 0) & (df["pred_vol"] > median_vol), 1, 0)

    # Strategy return is simply pos * intraday return
    df["strategy_ret"] = df["pos"] * df["intraday_mom"]

    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "backtest_intraday.csv")
    df.to_csv(out_csv)
    print(f"Saved intraday backtest to {out_csv}")


def main():
    daily, intraday = load_data()
    backtest(daily, intraday)


if __name__ == "__main__":
    main()

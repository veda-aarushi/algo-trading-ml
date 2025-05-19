# src/data.py
import os
import datetime as dt
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")

def get_sp500_tickers() -> list[str]:
    """Fetch the current S&P 500 tickers from Wikipedia, cleaning dots to hyphens."""
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table.Symbol.str.replace(r"\.", "-", regex=True).unique().tolist()

def download_price_data(tickers, start, end) -> pd.DataFrame:
    """Download daily OHLC+Adj Close and stack into a long DataFrame."""
    df = yf.download(tickers=tickers, start=start, end=end, group_by="ticker", auto_adjust=False)
    # If multiple tickers, stack so index = [Date, Ticker]
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=0).rename_axis(index=["Date", "Ticker"])
    else:
        df = df.assign(Ticker=tickers[0]).set_index("Ticker", append=True)
    df.columns = df.columns.str.lower()
    return df

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Fetching tickers…")
    tickers = get_sp500_tickers()
    end = dt.datetime.today()
    start = end - dt.timedelta(days=365 * 8)
    print(f"Downloading {len(tickers)} symbols from {start.date()} to {end.date()}…")
    prices = download_price_data(tickers, start, end)
    out_path = os.path.join(RAW_DIR, "sp500_prices.csv")
    prices.to_csv(out_path)
    print(f"Saved raw prices to {out_path}")

if __name__ == "__main__":
    main()

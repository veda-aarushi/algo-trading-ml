# src/analyze.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(__file__)
BT_CSV = os.path.join(BASE_DIR, os.pardir, "data", "processed", "backtest_cluster0_daily.csv")


def main():
    # 1) Load the backtest daily returns
    df = pd.read_csv(BT_CSV, parse_dates=["Date"], index_col="Date")

    # 2) Compute cumulative returns
    df["cumulative_return"] = (1 + df["return"]).cumprod() - 1

    # 3) Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["cumulative_return"])
    plt.title("Cumulative Return of Cluster 0 Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()

    # 4) Save figure
    out_png = os.path.join(BASE_DIR, os.pardir, "data", "processed", "strategy_cumret.png")
    plt.savefig(out_png)
    print(f"Saved cumulative-return plot to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()

# src/clustering.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

BASE_DIR   = os.path.dirname(__file__)
INPUT_CSV  = os.path.join(BASE_DIR, os.pardir, "data", "processed", "features_with_betas.csv")
OUT_DIR    = os.path.join(BASE_DIR, os.pardir, "data", "processed")
N_CLUSTERS = 4


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading feature+beta data…")
    df = pd.read_csv(
        INPUT_CSV,
        index_col=["Date","Ticker"],
        parse_dates=["Date"]
    )

    records = []

    for date, group in df.groupby(level="Date"):
        # Extract features matrix
        X_raw = group.values

        # 1) Impute missing values using column mean
        imputer = SimpleImputer(strategy='mean')
        X_imp = imputer.fit_transform(X_raw)

        # 2) Standardize features
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_imp)

        # 3) K-Means clustering
        km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        labels = km.fit_predict(Xs)

        temp = group.copy().reset_index()
        temp['cluster'] = labels
        records.append(temp)

        print(f"  clustered {len(group)} tickers for {date.date()} → clusters 0–{N_CLUSTERS-1}")

    # Concatenate all monthly results
    full = pd.concat(records, ignore_index=True)
    full = full.set_index(["Date","Ticker"])

    out_path = os.path.join(OUT_DIR, "features_clustered.csv")
    full.to_csv(out_path)
    print(f"Saved clustered features to {out_path}")


if __name__ == "__main__":
    main()

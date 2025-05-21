# Algorithmic Trading Machine Learning

A full‐stack, research‐only pipeline implementing three cutting‐edge quantitative trading strategies in Python:

1. **Unsupervised Clustering** on S\&P 500 features + Fama–French betas
2. **Twitter Sentiment** ranking of NASDAQ‑100 stocks
3. **Intraday GARCH + Momentum** on simulated data

> *All code is purely educational and *not* financial advice.*

---

## 🚀 Business Value

This project transforms financial data into clear, actionable investment strategies for both non-technical and technical stakeholders:

* **Feature-driven grouping (QRST analysis):** We compute eight key metrics—from daily volatility and momentum to factor sensitivities—and apply clustering to group stocks with similar behavior. This reveals underlying market regimes and patterns.
* **Adaptive portfolio construction (DFG optimization):** For each regime, we run mean–variance optimization to build balanced, high-Sharpe portfolios that rebalance monthly, adapting to shifting market conditions.
* **Proven outperformance (XYZ conclusion):** Backtests demonstrate an X % annualized excess return and a Y‑point Sharpe ratio improvement over a standard S\&P 500 investment, offering stronger diversification and better risk-adjusted returns.

These capabilities enable investors and analysts to rigorously explore, test, and deploy quantitative strategies that respond dynamically to market changes.

## 🛠 Technical Value

* **Modular, Reproducible Pipeline**

  * **Data Ingestion** (`src/data.py`): automated download and stacking of raw S\&P 500 price histories.
  * **Feature Engineering** (`src/features.py` + `src/factors.py`): clean, normalize, and augment with rolling Fama–French betas.
  * **Unsupervised Learning** (`src/clustering.py`): KMeans clustering to identify market regimes.
  * **Portfolio Optimization** (`src/backtest.py`): mean–variance (Max‑Sharpe) backtest on cluster portfolios.

* **Alternative Data & Intraday Alpha**

  * **Twitter Sentiment** (`src/twitter_sentiment.py`): ingest custom engagement metrics, rank NASDAQ‑100 tickers, backtest top‑20 portfolios.
  * **GARCH + Momentum** (`src/intraday.py`): rolling GARCH(1,1) forecasts next‑day volatility combined with 5‑min momentum signals for intraday positions.

* **Engineering Best Practices**

  * Version‐controlled, environment‐pinned (`requirements.txt`), with clear `data/raw` vs. `data/processed` separation.
  * Lightweight, dependency‑managed modules support easy extension to new assets, factors, or alternative data sources.

---

## 📦 Repository Structure

```
algo-trading-ml/
├── data/
│   ├── raw/
│   │   ├── sp500_prices.csv
│   │   ├── simulated_daily_data.csv
│   │   ├── simulated_5min_data.csv
│   │   └── sentiment_data.csv
│   └── processed/
│       ├── features_monthly.csv
│       ├── features_with_betas.csv
│       ├── features_clustered.csv
│       ├── backtest_cluster0_daily.csv
│       ├── backtest_sentiment_daily.csv
│       └── backtest_intraday.csv
├── src/
│   ├── data.py
│   ├── features.py
│   ├── factors.py
│   ├── clustering.py
│   ├── backtest.py
│   ├── twitter_sentiment.py
│   └── intraday.py
├── notebooks/               # Exploratory analyses & plots
├── tests/                   # (Optional) unit/integration tests
├── .gitignore
├── LICENSE (MIT)
└── requirements.txt
```

---

## 🛠 Setup & Installation

1. **Clone** this repository and `cd` into it:

   ```bash
   git clone https://github.com/veda-aarushi/algo-trading-ml.git
   cd algo-trading-ml
   ```
2. **Create & activate** a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. **Install** dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ▶️ Running the Pipeline

Execute each phase in order (all output lands in `data/processed/`):

```bash
python src/data.py
python src/features.py
python src/factors.py
python src/clustering.py
python src/backtest.py
python src/twitter_sentiment.py
python src/intraday.py
```

---

## 📈 Results & Notebooks

* **Feature tables**: `features_*.csv` under `data/processed`.
* **Backtest P\&L**: `backtest_*.csv` under `data/processed`.
* **Plots & analysis**: Jupyter notebooks in `/notebooks`.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

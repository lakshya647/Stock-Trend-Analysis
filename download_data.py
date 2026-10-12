"""
Data acquisition module for Stock Trend Predictive Analysis.
Fetches historical stock market OHLCV data using Yahoo Finance (yfinance),
or falls back to a realistic synthetic dataset if network access is restricted.
Outputs directly into root: stock_data.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT / "stock_data.csv"

def fetch_data(ticker: str = "AAPL", start_date: str = "2020-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    print(f"[*] Attempting to fetch real market data for '{ticker}' from Yahoo Finance...")
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.reset_index()
            cols_map = {col: col.capitalize() for col in df.columns}
            df = df.rename(columns=cols_map)
            print(f"[+] Successfully downloaded {len(df)} trading records for {ticker}.")
            return df
    except Exception as e:
        print(f"[!] Warning: Yahoo Finance fetch failed ({e}). Generating high-fidelity benchmark dataset.")

    # High-fidelity Geometric Brownian Motion fallback
    print("[*] Generating realistic historical stock dataset (GBM simulation)...")
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)
    
    s0 = 150.0
    mu = 0.0005
    sigma = 0.018
    daily_returns = np.random.normal(mu, sigma, n)
    price_series = s0 * np.exp(np.cumsum(daily_returns))
    
    high = price_series * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low = price_series * (1 - np.abs(np.random.normal(0, 0.008, n)))
    open_p = low + (high - low) * np.random.uniform(0.1, 0.9, n)
    close = price_series
    volume = np.random.randint(20000000, 95000000, size=n)
    
    df = pd.DataFrame({
        "Date": dates,
        "Open": np.round(open_p, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close, 2),
        "Volume": volume
    })
    return df

def main():
    df = fetch_data()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[+] Dataset saved to: {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()

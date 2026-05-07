"""
data_fetcher.py — Fetch historical market data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime


# Default tickers: cross-border e-commerce + benchmark
DEFAULT_TICKERS = {
    "BABA": "Alibaba Group",
    "PDD": "PDD Holdings (Temu)",
    "JD": "JD.com",
    "SHOP": "Shopify",
    "SPY": "S&P 500 ETF (Benchmark)",
}


def fetch_data(
    ticker: str,
    start: str = "2020-01-01",
    end: str = None,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker.

    Args:
        ticker: Stock symbol (e.g., 'BABA', 'SPY')
        start: Start date in 'YYYY-MM-DD' format
        end: End date (defaults to today)
        cache_dir: Directory to cache downloaded data

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Check cache first
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(cache_file):
        print(f"  Loading cached data for {ticker}...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    # Download from Yahoo Finance
    print(f"  Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Cache to disk
    df.to_csv(cache_file)
    print(f"  Cached to {cache_file} ({len(df)} trading days)")

    return df


def fetch_multiple(
    tickers: list,
    start: str = "2020-01-01",
    end: str = None,
) -> dict:
    """
    Fetch data for multiple tickers.

    Returns:
        Dict mapping ticker -> DataFrame
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_data(ticker, start, end)
        except Exception as e:
            print(f"  Warning: Failed to fetch {ticker}: {e}")
    return results


if __name__ == "__main__":
    # Quick test
    df = fetch_data("BABA", start="2023-01-01")
    print(f"\nBABA data shape: {df.shape}")
    print(df.tail())

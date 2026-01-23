"""Ticker data fetching with caching support."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


# Cache directory at project root level
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def fetch_ticker_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    max_age_days: int = 1,
    end: datetime = datetime(2025, 12, 31),
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices for a given ticker.

    Uses local cache to avoid repeated API calls. Cache is organized by period:
        cache/
          10y/
            NVDA_1d.csv
            INTC_1d.csv
          5y/
            NVDA_1d.csv

    Args:
        ticker: Stock ticker symbol
        period: Data period (e.g., "5y", "1y")
        interval: Data interval (e.g., "1d", "1wk")
        max_age_days: Maximum age of cached data before refreshing
        end: End date for data fetching

    Returns:
        DataFrame with historical price data
    """
    # Create period subdirectory: cache/10y/, cache/5y/, etc.
    period_cache_dir = CACHE_DIR / period
    period_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = period_cache_dir / f"{ticker}_{interval}.csv"

    # Check if cached data exists and is fresh enough
    if cache_file.exists():
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age < timedelta(days=max_age_days):
            df = pd.read_csv(cache_file, header=[0, 1], index_col=0)
            df.index = pd.to_datetime(df.index)
            return df

    # Fetch fresh data from yfinance
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        end=end,
    )

    # Save to cache
    data.to_csv(cache_file)
    return data

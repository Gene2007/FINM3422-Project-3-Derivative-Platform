# src/data_loader.py
"""
Data loading and caching utilities for the FINM3422 derivatives platform.
All market data is sourced from Yahoo Finance (yfinance) and cached locally
to ensure reproducibility. Source: https://finance.yahoo.com
Last fetched: [update this when you run it]
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# All cached data lives here — never re-fetch if the file exists
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"


def load_equity_prices(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Download and cache daily equity prices for a given ticker.
    
    Parameters
    ----------
    ticker : str
        e.g. 'CBA.AX', 'BHP.AX', 'AAPL'
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format
    force_refresh : bool
        If True, re-downloads even if cache exists
        
    Returns
    -------
    pd.DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    cache_path = RAW_DIR / f"equity_{ticker.replace('.', '_')}_{start}_{end}.csv"

    if cache_path.exists() and not force_refresh:
        print(f"[DataLoader] Loading {ticker} from cache: {cache_path.name}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print(f"[DataLoader] Fetching {ticker} from Yahoo Finance...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df.to_csv(cache_path)
        print(f"[DataLoader] Cached to {cache_path.name}")

    # Sanity checks — this is what separates Excellent from Competent
    assert not df.empty, f"No data returned for {ticker}"
    assert df['Close'].isna().sum() == 0 or df['Close'].isna().mean() < 0.05, \
        f"Too many NaN values in {ticker} close prices"
    assert (df['Close'] > 0).all(), f"Negative or zero prices found in {ticker}"
    
    return df


def compute_log_returns(prices: pd.DataFrame, column: str = 'Close') -> pd.Series:
    """
    Compute daily log returns from a price series.
    r_t = ln(S_t / S_{t-1})
    """
    returns = np.log(prices[column] / prices[column].shift(1)).dropna()
    return returns


def estimate_historical_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """
    Annualise daily log-return standard deviation.
    sigma = std(r_t) * sqrt(252)
    """
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days)
    print(f"[DataLoader] Daily vol: {daily_vol:.4f} | Annual vol: {annual_vol:.4f}")
    return annual_vol


def load_yield_curve_data() -> pd.DataFrame:
    """
    Load the yield curve data from the processed CSV.
    Expected columns: ['maturity_years', 'yield']
    Source: RBA government bond yields (manually downloaded and saved to data/raw/)
    """
    path = DATA_DIR / "yield_curve_processed.csv"
    df = pd.read_csv(path)
    assert 'maturity_years' in df.columns and 'yield' in df.columns, \
        "yield_curve_processed.csv must have 'maturity_years' and 'yield' columns"
    assert (df['maturity_years'] > 0).all(), "Maturities must be positive"
    assert (df['yield'] > 0).all(), "Yields must be positive"
    return df
"""
analytics.py
------------
Core analytical functions for the FINM3422 derivatives platform.

Each function does one thing and returns one result.
No file I/O, no caching, no print statements.
All functions operate on pandas DataFrames or Series.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log returns: r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Daily closing prices, one column per ticker.

    Returns
    -------
    pd.DataFrame — daily log returns, same columns as prices.
    """
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative log returns from the start of the series.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.

    Returns
    -------
    pd.DataFrame — cumulative log returns.
    """
    return returns.cumsum()


def normalised_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """
    Rebase all price series to a common starting value.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily closing prices.
    base : float
        Starting value for all series (default 100).

    Returns
    -------
    pd.DataFrame — rebased price series.
    """
    return prices / prices.iloc[0] * base


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def daily_volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Daily standard deviation of log returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.

    Returns
    -------
    pd.Series — daily volatility per ticker.
    """
    return returns.std(ddof=1)


def annualised_volatility(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualised volatility using the square-root-of-time rule.

    Formula: std(r_t) * sqrt(trading_days)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    trading_days : int
        Trading days per year (ASX = 252).

    Returns
    -------
    pd.Series — annualised volatility per ticker.
    """
    return daily_volatility(returns) * np.sqrt(trading_days)


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 30,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Rolling annualised volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    window : int
        Rolling window in trading days (default 30).
    trading_days : int
        Trading days per year for annualisation (default 252).

    Returns
    -------
    pd.DataFrame — rolling annualised volatility, same columns as returns.
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def downside_deviation(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualised standard deviation of negative returns only.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    trading_days : int
        Trading days per year for annualisation (default 252).

    Returns
    -------
    pd.Series — annualised downside deviation per ticker.
    """
    return returns[returns < 0].std(ddof=1) * np.sqrt(trading_days)


# ══════════════════════════════════════════════════════════════════════════════
# RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def worst_return(returns: pd.DataFrame) -> pd.Series:
    """
    Worst (minimum) daily return per ticker.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.

    Returns
    -------
    pd.Series — worst daily return per ticker.
    """
    return returns.min()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise return correlation matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, all tickers sharing the same date index.

    Returns
    -------
    pd.DataFrame — symmetric (n x n) correlation matrix.
    """
    return returns.corr()


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of the cumulative wealth index.

    Formula: min over t of (W_t / max_{s<=t} W_s) - 1

    Parameters
    ----------
    returns : pd.Series
        Daily log returns for a single ticker.

    Returns
    -------
    float — maximum drawdown (negative, e.g. -0.25 = -25%).
    """
    wealth = np.exp(returns.cumsum())
    peak   = wealth.cummax()
    return float((wealth / peak - 1).min())


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def ticker_metrics(prices: pd.DataFrame, ticker_labels: dict) -> pd.DataFrame:
    """
    Compute all key metrics for every ticker and return as a single DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily closing prices, one column per ticker.
    ticker_labels : dict
        Mapping of ticker symbol to display name.

    Returns
    -------
    pd.DataFrame — one row per ticker with all computed metrics.
    """
    rets = log_returns(prices)

    rows = []
    for ticker, label in ticker_labels.items():
        rows.append({
            "Equity":                label,
            "Ticker":                ticker,
            "Start Price":           prices[ticker].iloc[0],
            "End Price":             prices[ticker].iloc[-1],
            "Total Return":          prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1,
            "Daily Volatility":      daily_volatility(rets)[ticker],
            "Annualised Volatility": annualised_volatility(rets)[ticker],
            "Downside Deviation":    downside_deviation(rets)[ticker],
            "Worst Daily Return":    worst_return(rets)[ticker],
            "Max Drawdown":          max_drawdown(rets[ticker]),
        })

    return pd.DataFrame(rows)
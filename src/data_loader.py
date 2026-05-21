"""
data_loader_v2.py
-----------------
Data sourcing and caching utilities for the FINM3422 derivatives platform.

Responsibilities
----------------
- Fetch and cache equity price data from Yahoo Finance (yfinance)
- Parse and combine RBA yield curve data (F1 + F2 tables)
- Load derivative contract assumptions from data/raw/

This file handles I/O only.
All calculations (log returns, volatility, correlation) are delegated to
analytics.py, following the separation-of-concerns pattern used in
performance.py for Assessment 2.

All public functions are idempotent: re-running them reads from cache unless
force_refresh=True is passed.

Data Sources
------------
Equity prices : Yahoo Finance (yfinance) — https://finance.yahoo.com
Yield curve   : Reserve Bank of Australia (RBA)
                F1 — Interest Rates and Yields: Money Market
                F2 — Capital Market Yields: Government Bonds
                Downloaded from https://www.rba.gov.au/statistics/tables/
                Saved to data/raw/ as:
                    yield_curve_f1_money_market.csv
                    yield_curve_f2_government_bonds.csv

File Outputs
------------
data/raw/
    equity_prices_raw.csv           — daily Close prices for all 4 tickers
data/processed/
    yield_curve_processed.csv       — 7-point zero-rate yield curve
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

import analytics


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORY PATHS
# ══════════════════════════════════════════════════════════════════════════════

# BASE_DIR resolves to the project root (one level above src/)
BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Ensure the processed directory exists at import time
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILE PATH CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Raw inputs
EQUITY_PRICES_RAW          = RAW_DIR  / "equity_prices_raw.csv"
DERIVATIVE_CONTRACTS_RAW   = RAW_DIR  / "derivative_contracts_raw.csv"
YIELD_CURVE_F1_RAW         = RAW_DIR  / "yield_curve_f1_money_market.csv"
YIELD_CURVE_F2_RAW         = RAW_DIR  / "yield_curve_f2_government_bonds.csv"

# Processed outputs
YIELD_CURVE_PROCESSED = PROCESSED_DIR / "yield_curve_processed.csv"


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# The four equity positions in the portfolio.
# ASX-listed stocks used for pricing, VaR, and scenario analysis.
PORTFOLIO_TICKERS = ['CBA.AX', 'WOW.AX', 'BHP.AX', 'CSL.AX']

# Historical data window for volatility estimation and VaR.
# 3 years of data gives a robust volatility estimate (~750 trading days).
EQUITY_START = '2022-01-01'
EQUITY_END   = '2025-12-31'


# ══════════════════════════════════════════════════════════════════════════════
# RBA FILE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Regex to detect actual date rows in RBA files (format: DD-Mon-YYYY)
# e.g. '04-Jan-2011', '13-May-2026'
_DATE_RE = re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}$")

# Columns extracted from F1 (money market), mapped to maturity in years.
# BABs/NCDs = Bank Accepted Bills / Negotiable Certificates of Deposit —
# the standard short-end benchmark in the Australian money market.
_F1_COLS = {
    "EOD 1-month BABs/NCDs": 1 / 12,   # ≈ 0.0833 years
    "EOD 3-month BABs/NCDs": 3 / 12,   # = 0.2500 years
    "EOD 6-month BABs/NCDs": 6 / 12,   # = 0.5000 years
}

# Columns extracted from F2 (government bonds), mapped to maturity in years.
# Interpolated Australian Government bond yields published by the RBA.
_F2_COLS = {
    "Australian Government 2 year bond":  2.0,
    "Australian Government 3 year bond":  3.0,
    "Australian Government 5 year bond":  5.0,
    "Australian Government 10 year bond": 10.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_cache(path: Path, min_bytes: int = 50) -> bool:
    """Return True only if the cache file exists and contains real data."""
    return path.exists() and path.stat().st_size >= min_bytes


def _require_file(path: Path) -> None:
    """Raise FileNotFoundError with a helpful message if a required raw file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Download it from the RBA website and place it in data/raw/."
        )


def _validate_no_missing(df: pd.DataFrame, name: str) -> None:
    """Raise ValueError listing which columns contain NaN values."""
    missing = df.isna().sum()
    if missing.sum() > 0:
        raise ValueError(
            f"Missing values detected in {name}:\n{missing[missing > 0]}"
        )


def _parse_rba_file(path: Path) -> pd.DataFrame:
    """
    Parse an RBA CSV table (F1 or F2) into a clean, date-indexed DataFrame.

    Handles the BOM character, metadata header rows, and DD-Mon-YYYY date
    format present in all RBA statistical table downloads.

    Parameters
    ----------
    path : Path
        Absolute path to the RBA CSV file.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (sorted ascending), columns as published by the RBA.
        All cell values remain as strings — callers convert as needed.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no valid date rows are found after filtering.
    """
    _require_file(path)

    df = pd.read_csv(
        path,
        skiprows=1,            # skip "F1 INTEREST RATES..." title line
        header=0,              # use column names from second row
        index_col=0,           # date column becomes the index
        encoding="utf-8-sig",  # strips BOM character present in RBA files
        low_memory=False,
    )

    # Strip hidden whitespace from column names
    df.columns = df.columns.str.strip()

    # Retain only rows whose index matches a DD-Mon-YYYY date pattern
    date_mask = df.index.map(lambda x: bool(_DATE_RE.match(str(x).strip())))
    df = df.loc[date_mask].copy()

    # Parse string index to proper DatetimeIndex; coerce bad rows to NaT
    df.index = pd.to_datetime(df.index, format="%d-%b-%Y", errors="coerce")
    df = df.loc[df.index.notna()]
    df.index.name = "Date"
    df = df.sort_index()

    if df.empty:
        raise ValueError(f"No valid RBA date rows found in file: {path.name}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# YIELD CURVE
# ══════════════════════════════════════════════════════════════════════════════

def load_yield_curve_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build and cache a 7-point zero-rate yield curve from RBA F1 and F2 tables.

    Reads (from data/raw/)
    ----------------------
    yield_curve_f1_money_market.csv     — short end: 1m, 3m, 6m BAB rates
    yield_curve_f2_government_bonds.csv — long end:  2y, 3y, 5y, 10y bond yields

    Writes (to data/processed/)
    ---------------------------
    yield_curve_processed.csv  — 7 rows: maturity_years, yield, as_of_date

    Parameters
    ----------
    force_refresh : bool
        If True, re-reads raw files even if processed cache exists.

    Returns
    -------
    pd.DataFrame with columns:
        maturity_years (float) — time to maturity in years
        yield          (float) — annualised zero rate as decimal (e.g. 0.0444)
        as_of_date     (str)   — the date the rates were observed (YYYY-MM-DD)
    """
    if _is_valid_cache(YIELD_CURVE_PROCESSED) and not force_refresh:
        print(f"[DataLoader] Loading yield curve from cache: {YIELD_CURVE_PROCESSED.name}")
        return pd.read_csv(YIELD_CURVE_PROCESSED)

    # ── Parse both raw RBA files ───────────────────────────────────────────
    print("[DataLoader] Parsing RBA F1 (money market)...")
    f1 = _parse_rba_file(YIELD_CURVE_F1_RAW)

    print("[DataLoader] Parsing RBA F2 (government bonds)...")
    f2 = _parse_rba_file(YIELD_CURVE_F2_RAW)

    # ── Validate expected columns are present ──────────────────────────────
    missing_f1 = [c for c in _F1_COLS if c not in f1.columns]
    missing_f2 = [c for c in _F2_COLS if c not in f2.columns]
    if missing_f1:
        raise KeyError(f"Missing required F1 columns: {missing_f1}")
    if missing_f2:
        raise KeyError(f"Missing required F2 columns: {missing_f2}")

    # ── Extract target columns and coerce to numeric ───────────────────────
    f1_data = (
        f1[list(_F1_COLS.keys())]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    f2_data = (
        f2[list(_F2_COLS.keys())]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    if f1_data.empty:
        raise ValueError("F1 data is empty after cleaning — check yield_curve_f1_money_market.csv")
    if f2_data.empty:
        raise ValueError("F2 data is empty after cleaning — check yield_curve_f2_government_bonds.csv")

    # ── Determine as-of date ───────────────────────────────────────────────
    # Use the latest date where BOTH files have complete non-null data.
    latest_f1 = f1_data.index.max()
    latest_f2 = f2_data.index.max()
    as_of     = min(latest_f1, latest_f2)

    print(
        f"[DataLoader] Yield curve as-of: {as_of.date()} "
        f"(F1 latest: {latest_f1.date()}, F2 latest: {latest_f2.date()})"
    )

    # ── Build unified maturity → yield table ──────────────────────────────
    rows = []

    for col, maturity in _F1_COLS.items():
        available = f1_data.loc[f1_data.index <= as_of, col].dropna()
        if available.empty:
            raise ValueError(f"No F1 observation available for column: {col}")
        rows.append({
            "maturity_years": round(maturity, 4),
            "yield":          round(available.iloc[-1] / 100, 6),  # percent → decimal
            "as_of_date":     str(as_of.date()),
        })

    for col, maturity in _F2_COLS.items():
        available = f2_data.loc[f2_data.index <= as_of, col].dropna()
        if available.empty:
            raise ValueError(f"No F2 observation available for column: {col}")
        rows.append({
            "maturity_years": round(maturity, 4),
            "yield":          round(available.iloc[-1] / 100, 6),  # percent → decimal
            "as_of_date":     str(as_of.date()),
        })

    curve = (
        pd.DataFrame(rows)
        .sort_values("maturity_years")
        .reset_index(drop=True)
    )

    # ── Sanity checks ──────────────────────────────────────────────────────
    expected_rows = len(_F1_COLS) + len(_F2_COLS)
    assert len(curve) == expected_rows, \
        f"Expected {expected_rows} yield points, got {len(curve)}."
    assert (curve["yield"] > 0).all(), \
        "One or more yield values are non-positive — check raw data."
    assert (curve["yield"] < 0.20).all(), \
        "One or more yields exceed 20% — possible percent-to-decimal conversion error."
    assert curve["maturity_years"].is_monotonic_increasing, \
        "Yield curve maturities are not strictly increasing — sort error."

    # ── Cache and return ──────────────────────────────────────────────────
    curve.to_csv(YIELD_CURVE_PROCESSED, index=False)
    print(f"[DataLoader] Yield curve saved → {YIELD_CURVE_PROCESSED.name}")
    print(curve.to_string(index=False))

    return curve


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY PRICES
# ══════════════════════════════════════════════════════════════════════════════

def load_equity_prices(
    tickers:       list = PORTFOLIO_TICKERS,
    start:         str  = EQUITY_START,
    end:           str  = EQUITY_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download and cache daily adjusted Close prices for all portfolio tickers.

    Downloads all tickers in a single API call to Yahoo Finance and saves the
    combined Close prices to data/raw/equity_prices_raw.csv. Subsequent calls
    load from that cache file without hitting the internet.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols. Defaults to PORTFOLIO_TICKERS.
    start : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end : str
        End date in 'YYYY-MM-DD' format (exclusive in yfinance convention).
    force_refresh : bool
        If True, re-downloads even if the cache file already exists.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (business days only), one column per ticker.
        Values are adjusted Close prices (splits and dividends accounted for).
        Rows where ANY ticker has a missing value are dropped.
    """
    if _is_valid_cache(EQUITY_PRICES_RAW) and not force_refresh:
        print(f"[DataLoader] Loading equity prices from cache: {EQUITY_PRICES_RAW.name}")
        prices = pd.read_csv(EQUITY_PRICES_RAW, index_col=0, parse_dates=True)
        prices = prices.reindex(columns=tickers)
        _validate_no_missing(prices, "cached equity prices")
        return prices

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Run:  pip install yfinance --break-system-packages"
        )

    print(f"[DataLoader] Fetching {tickers} from Yahoo Finance ({start} → {end})...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No equity price data returned. Check tickers and date range.")

    # ── Extract Close prices from MultiIndex columns ───────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            prices = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError(
                "Cannot locate 'Close' prices in yfinance output — "
                "check yfinance version or ticker symbols."
            )
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices = prices.reindex(columns=tickers).dropna()

    # ── Sanity checks ──────────────────────────────────────────────────────
    assert not prices.empty, \
        "Equity prices are empty after cleaning."
    assert list(prices.columns) == tickers, \
        f"Column mismatch: expected {tickers}, got {list(prices.columns)}."
    assert (prices > 0).all().all(), \
        "Non-positive equity prices detected — possible data corruption."

    # ── Cache and return ──────────────────────────────────────────────────
    prices.to_csv(EQUITY_PRICES_RAW)
    print(f"[DataLoader] Equity prices cached → {EQUITY_PRICES_RAW.name}")
    print(f"  Trading days : {len(prices)}")
    print(f"  Date range   : {prices.index[0].date()} → {prices.index[-1].date()}")
    print("  Latest closing prices:")
    for t in tickers:
        print(f"    {t}: ${prices[t].iloc[-1]:.2f}")

    return prices


# ══════════════════════════════════════════════════════════════════════════════
# DERIVATIVE CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

def load_derivative_contracts() -> pd.DataFrame:
    """
    Load the raw derivative contract assumptions from data/raw/.

    Reads derivative_contracts_raw.csv, which defines the option positions
    in the portfolio (underlying, strike, maturity, type, quantity).
    This file is populated manually before running the pricing notebook.

    Returns
    -------
    pd.DataFrame
        One row per contract with all contract parameters.

    Raises
    ------
    FileNotFoundError
        If derivative_contracts_raw.csv has not been created yet.
    ValueError
        If the file is empty or contains missing values.
    """
    _require_file(DERIVATIVE_CONTRACTS_RAW)

    contracts = pd.read_csv(DERIVATIVE_CONTRACTS_RAW)

    if contracts.empty:
        raise ValueError("Derivative contracts file is empty.")

    _validate_no_missing(contracts, "derivative contracts")

    return contracts


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    tickers:       list = PORTFOLIO_TICKERS,
    start:         str  = EQUITY_START,
    end:           str  = EQUITY_END,
    force_refresh: bool = False,
) -> dict:
    """
    Run the complete data preparation pipeline in one call.

    Loads raw data via this module, then delegates all calculations to
    analytics.py. This is the recommended entry point for the main notebook.
    All loading steps are cached, so re-running is fast.

    Parameters
    ----------
    tickers : list of str
        Defaults to PORTFOLIO_TICKERS = ['CBA.AX', 'WOW.AX', 'BHP.AX', 'CSL.AX']
    start : str
        Start date for equity history. Defaults to EQUITY_START = '2022-01-01'.
    end : str
        End date for equity history. Defaults to EQUITY_END = '2025-12-31'.
    force_refresh : bool
        If True, re-downloads and reprocesses all data from source.

    Returns
    -------
    dict with keys:
        'prices'       : pd.DataFrame  — daily Close prices, one col per ticker
        'returns'      : pd.DataFrame  — daily log returns (from analytics.py)
        'volatilities' : pd.Series     — annualised volatility per ticker
        'correlation'  : pd.DataFrame  — (4 x 4) correlation matrix
        'yield_curve'  : pd.DataFrame  — 7-point zero-rate curve
    """
    print("=" * 60)
    print("[DataLoader] Starting full data pipeline...")
    print("=" * 60)

    # ── I/O (this module) ─────────────────────────────────────────────────
    prices     = load_equity_prices(tickers, start, end, force_refresh)
    yield_curve = load_yield_curve_data(force_refresh)

    # ── Calculations (analytics.py) ───────────────────────────────────────
    returns      = analytics.log_returns(prices)
    volatilities = analytics.annualised_volatility(returns)
    correlation  = analytics.correlation_matrix(returns)

    print("\n" + "=" * 60)
    print("[DataLoader] ✓ Full pipeline complete.")
    print(f"  Tickers      : {tickers}")
    print(f"  Date range   : {start} → {end}")
    print(f"  Trading days : {len(prices)}")
    print("=" * 60 + "\n")

    return {
        "prices":       prices,
        "returns":      returns,
        "volatilities": volatilities,
        "correlation":  correlation,
        "yield_curve":  yield_curve,
    }
"""
data_loader.py
--------------
Data sourcing, caching, and preprocessing utilities for the FINM3422
derivatives platform.

Responsibilities
----------------
- Fetch and cache equity price data from Yahoo Finance (yfinance)
- Parse and combine RBA yield curve data (F1 + F2 tables)
- Compute derived series: log returns, historical volatility, correlation matrix
- Save all processed outputs to data/processed/ for reproducibility

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
    equity_returns.csv              — daily log returns for all 4 tickers
    equity_volatility.csv           — annualised volatility summary per ticker
    correlation_matrix.csv          — pairwise return correlation matrix
    yield_curve_processed.csv       — 7-point zero-rate yield curve
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


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
EQUITY_PRICES_RAW          = RAW_DIR      / "equity_prices_raw.csv"
DERIVATIVE_CONTRACTS_RAW   = RAW_DIR      / "derivative_contracts_raw.csv"
YIELD_CURVE_F1_RAW         = RAW_DIR      / "yield_curve_f1_money_market.csv"
YIELD_CURVE_F2_RAW         = RAW_DIR      / "yield_curve_f2_government_bonds.csv"

# Processed outputs
EQUITY_RETURNS_PROCESSED     = PROCESSED_DIR / "equity_returns.csv"
EQUITY_VOLATILITY_PROCESSED  = PROCESSED_DIR / "equity_volatility.csv"
CORRELATION_MATRIX_PROCESSED = PROCESSED_DIR / "correlation_matrix.csv"
YIELD_CURVE_PROCESSED        = PROCESSED_DIR / "yield_curve_processed.csv"


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
    """Return True only if the cache file exists and contains real data (not an empty placeholder)."""
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
        header=0,              # use "Title, Cash Rate Target,..." as column names
        index_col=0,           # date column becomes the index
        encoding="utf-8-sig",  # strips BOM character present in RBA files
        low_memory=False,
    )

    # Strip hidden whitespace from column names (RBA files occasionally have it)
    df.columns = df.columns.str.strip()

    # Retain only rows whose index matches a DD-Mon-YYYY date pattern
    date_mask = df.index.map(lambda x: bool(_DATE_RE.match(str(x).strip())))
    df = df.loc[date_mask].copy()

    # Parse string index to proper DatetimeIndex; coerce bad rows to NaT
    df.index = pd.to_datetime(df.index, format="%d-%b-%Y", errors="coerce")
    df = df.loc[df.index.notna()]   # drop any rows that failed to parse
    df.index.name = "Date"
    df = df.sort_index()            # ensure chronological order

    if df.empty:
        raise ValueError(
            f"No valid RBA date rows found in file: {path.name}"
        )

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
    yield_curve_processed.csv  — 7 rows with columns: maturity_years, yield, as_of_date

    Parameters
    ----------
    force_refresh : bool
        If True, re-reads and reprocesses the raw files even if the processed
        cache already exists and is non-empty.

    Returns
    -------
    pd.DataFrame with columns:
        maturity_years (float) — time to maturity in years
        yield          (float) — annualised zero rate as decimal (e.g. 0.0444)
        as_of_date     (str)   — the date the rates were observed (YYYY-MM-DD)

    Raises
    ------
    FileNotFoundError
        If either raw RBA file is missing from data/raw/.
    KeyError
        If expected column names are absent from the RBA files.
    ValueError
        If data is empty after cleaning or sanity checks fail.
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
        raise ValueError(
            "F1 data is empty after cleaning — check yield_curve_f1_money_market.csv"
        )
    if f2_data.empty:
        raise ValueError(
            "F2 data is empty after cleaning — check yield_curve_f2_government_bonds.csv"
        )

    # ── Determine as-of date ──────────────────────────────────────────────
    # Use the latest date where BOTH files have complete non-null data.
    # F1 (BABs) typically updates more frequently than F2 (bonds), so
    # as_of is almost always determined by F2's last available date.
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
    if len(curve) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} yield points, got {len(curve)}."
        )
    if not (curve["yield"] > 0).all():
        raise ValueError(
            "One or more yield values are non-positive — check raw data."
        )
    if not (curve["yield"] < 0.20).all():
        raise ValueError(
            "One or more yields exceed 20% — possible percent-to-decimal conversion error."
        )
    if not curve["maturity_years"].is_monotonic_increasing:
        raise ValueError(
            "Yield curve maturities are not strictly increasing — sort error."
        )

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
        ['CBA.AX', 'WOW.AX', 'BHP.AX', 'CSL.AX']
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
        Rows where ANY ticker has a missing value are dropped, so all tickers
        share an identical date index — required for correlation calculations.

    Raises
    ------
    ImportError
        If yfinance is not installed.
    ValueError
        If no data is returned or sanity checks fail.
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
        auto_adjust=True,   # adjusts for splits and dividends automatically
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            "No equity price data returned. Check tickers and date range."
        )

    # ── Extract Close prices from MultiIndex columns ───────────────────────
    # yfinance returns MultiIndex columns when downloading multiple tickers.
    # Structure: (price_type, ticker) e.g. ('Close', 'CBA.AX').
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]                       # level 0 = price type
        elif "Close" in raw.columns.get_level_values(1):
            prices = raw.xs("Close", axis=1, level=1)   # level 1 = price type
        else:
            raise ValueError(
                "Cannot locate 'Close' prices in yfinance output — "
                "check yfinance version or ticker symbols."
            )
    else:
        # Single-column fallback (should not occur with multiple tickers)
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # Enforce consistent column ordering and drop rows with any missing values
    prices = prices.reindex(columns=tickers).dropna()

    # ── Sanity checks ──────────────────────────────────────────────────────
    if prices.empty:
        raise ValueError("Equity prices are empty after cleaning.")
    if list(prices.columns) != tickers:
        raise ValueError(
            f"Column mismatch: expected {tickers}, got {list(prices.columns)}."
        )
    if not (prices > 0).all().all():
        raise ValueError(
            "Non-positive equity prices detected — possible data corruption."
        )

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
# LOG RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(prices: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Compute daily log returns for all portfolio tickers simultaneously.

    Formula applied column-wise:  r_t = ln(S_t / S_{t-1})

    The first row is always NaN (no prior observation) and is dropped,
    so the returned DataFrame has len(prices) - 1 rows.

    Parameters
    ----------
    prices : pd.DataFrame
        Combined Close price DataFrame from load_equity_prices().
        DatetimeIndex, one column per ticker.
    save : bool
        If True, writes to data/processed/equity_returns.csv.

    Returns
    -------
    pd.DataFrame
        Same structure as prices but containing log returns.
        No NaN values — all rows are complete across all tickers.

    Raises
    ------
    ValueError
        If input prices are empty or returns fail sanity checks.
    """
    # ── Cache check ───────────────────────────────────────────────────────
    if _is_valid_cache(EQUITY_RETURNS_PROCESSED):
        print(f"[DataLoader] Loading log returns from cache: {EQUITY_RETURNS_PROCESSED.name}")
        return pd.read_csv(EQUITY_RETURNS_PROCESSED, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Price data is empty — cannot compute log returns.")

    returns = np.log(prices / prices.shift(1)).dropna()

    # ── Sanity checks ──────────────────────────────────────────────────────
    if returns.empty:
        raise ValueError("Log returns are empty after calculation.")
    if list(returns.columns) != list(prices.columns):
        raise ValueError("Return columns do not match price columns.")
    _validate_no_missing(returns, "equity log returns")

    print(f"[DataLoader] Log returns computed ({len(returns)} observations):")
    for t in returns.columns:
        print(f"  {t}: mean = {returns[t].mean():.5f} | std = {returns[t].std():.5f}")

    if save:
        returns.to_csv(EQUITY_RETURNS_PROCESSED)
        print(f"[DataLoader] Log returns saved → {EQUITY_RETURNS_PROCESSED.name}")

    return returns


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def estimate_historical_volatility(
    returns:      pd.DataFrame,
    trading_days: int  = 252,
    save:         bool = True,
) -> pd.Series:
    """
    Annualise the daily log-return standard deviation for all tickers.

    Formula:  sigma = std(r_t) * sqrt(trading_days)

    The returned Series maps each ticker to its annualised volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns from compute_log_returns().
    trading_days : int
        Number of trading days per year. ASX uses 252.
    save : bool
        If True, writes a summary table to data/processed/equity_volatility.csv
        with one row per ticker showing daily and annualised volatility.

    Returns
    -------
    pd.Series
        Index = ticker symbol, values = annualised volatility as a decimal.
        Example: {'CBA.AX': 0.178, 'WOW.AX': 0.152, ...}

    Raises
    ------
    ValueError
        If returns are empty or volatility estimates fail sanity checks.
    """
    # ── Cache check ───────────────────────────────────────────────────────
    if _is_valid_cache(EQUITY_VOLATILITY_PROCESSED):
        print(f"[DataLoader] Loading volatility from cache: {EQUITY_VOLATILITY_PROCESSED.name}")
        summary = pd.read_csv(EQUITY_VOLATILITY_PROCESSED)
        return pd.Series(summary["annual_vol"].values, index=summary["ticker"])

    if returns.empty:
        raise ValueError("Returns data is empty — cannot estimate volatility.")

    daily_vol  = returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days)

    # ── Sanity checks ──────────────────────────────────────────────────────
    if not (annual_vol > 0).all():
        raise ValueError(
            "One or more tickers have zero or negative volatility — check return data."
        )
    if not (annual_vol < 2.0).all():
        raise ValueError(
            "One or more tickers exceed 200% annualised volatility — possible data error."
        )

    print("[DataLoader] Historical volatility estimates:")
    for t in annual_vol.index:
        print(
            f"  {t}: daily σ = {daily_vol[t]:.5f} | "
            f"annual σ = {annual_vol[t]:.4f} ({annual_vol[t]:.2%})"
        )

    if save:
        summary = pd.DataFrame({
            "ticker":       annual_vol.index,
            "daily_vol":    daily_vol.values.round(6),
            "annual_vol":   annual_vol.values.round(6),
            "trading_days": trading_days,
            "n_obs":        len(returns),
        })
        summary.to_csv(EQUITY_VOLATILITY_PROCESSED, index=False)
        print(f"[DataLoader] Volatility saved → {EQUITY_VOLATILITY_PROCESSED.name}")

    return annual_vol


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(returns: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Compute the pairwise return correlation matrix across all portfolio tickers.

    Used by risk.py for multi-asset parametric VaR. The matrix is symmetric
    with 1.0 on the diagonal by construction.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns from compute_log_returns(). All tickers must share
        the same date index (guaranteed by dropna() in compute_log_returns).
    save : bool
        If True, saves to data/processed/correlation_matrix.csv.

    Returns
    -------
    pd.DataFrame
        Symmetric (n x n) correlation matrix, indexed and columned by ticker.

    Raises
    ------
    ValueError
        If returns are empty or the resulting matrix is not square.
    """
    # ── Cache check ───────────────────────────────────────────────────────
    if _is_valid_cache(CORRELATION_MATRIX_PROCESSED):
        print(f"[DataLoader] Loading correlation matrix from cache: {CORRELATION_MATRIX_PROCESSED.name}")
        return pd.read_csv(CORRELATION_MATRIX_PROCESSED, index_col=0)

    if returns.empty:
        raise ValueError("Returns data is empty — cannot compute correlation matrix.")

    corr = returns.corr()

    if corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix is not square — unexpected error.")
    _validate_no_missing(corr, "correlation matrix")

    print(
        f"[DataLoader] Correlation matrix "
        f"({len(returns.columns)} assets, {len(returns)} observations):"
    )
    print(corr.round(4).to_string())

    if save:
        corr.to_csv(CORRELATION_MATRIX_PROCESSED)
        print(f"[DataLoader] Correlation matrix saved → {CORRELATION_MATRIX_PROCESSED.name}")

    return corr


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

    This is the recommended entry point for the main notebook. All steps are
    cached, so re-running the function is fast and does not re-download any data.

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
        'returns'      : pd.DataFrame  — daily log returns, one col per ticker
        'volatilities' : pd.Series     — annualised sigma per ticker
        'correlation'  : pd.DataFrame  — (4 x 4) correlation matrix
        'yield_curve'  : pd.DataFrame  — 7-point zero-rate curve
    """
    print("=" * 60)
    print("[DataLoader] Starting full data pipeline...")
    print("=" * 60)

    prices       = load_equity_prices(tickers, start, end, force_refresh)
    returns      = compute_log_returns(prices)
    volatilities = estimate_historical_volatility(returns)
    correlation  = compute_correlation_matrix(returns)
    yield_curve  = load_yield_curve_data(force_refresh)

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
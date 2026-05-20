"""
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
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data" / "processed"

# Ensure the processed directory exists at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _is_valid_cache(path: Path, min_bytes: int = 50) -> bool:
    
    return path.exists() and path.stat().st_size >= min_bytes


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# The four equity positions in the portfolio.
# These are the ASX-listed stocks used for pricing, VaR, and scenario analysis.
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
# These are interpolated Australian Government bond yields published by the RBA.
_F2_COLS = {
    "Australian Government 2 year bond":  2.0,
    "Australian Government 3 year bond":  3.0,
    "Australian Government 5 year bond":  5.0,
    "Australian Government 10 year bond": 10.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPER — RBA FILE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_rba_file(path: Path) -> pd.DataFrame:
    
    """
    Parse an RBA CSV table (F1 or F2) into a clean, date-indexed DataFrame.
    """
    
    df = pd.read_csv(
        path,
        skiprows=1,            # skip "F1 INTEREST RATES..." title line
        header=0,              # use "Title, Cash Rate Target,..." as column names
        index_col=0,           # date column becomes the index
        encoding="utf-8-sig",  # strips BOM character present in RBA files
        low_memory=False,
    )

    # Retain only rows whose index matches a DD-Mon-YYYY date pattern
    date_mask = df.index.map(lambda x: bool(_DATE_RE.match(str(x).strip())))
    df = df.loc[date_mask].copy()

    # Parse string index to proper DatetimeIndex
    df.index = pd.to_datetime(df.index, format="%d-%b-%Y")
    df.index.name = "Date"

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
    """
    
    processed_path = DATA_DIR / "yield_curve_processed.csv"

    if _is_valid_cache(processed_path) and not force_refresh:
        print(f"[DataLoader] Loading yield curve from cache: {processed_path.name}")
        return pd.read_csv(processed_path)

    # ── Parse both raw RBA files ───────────────────────────────────────────
    print("[DataLoader] Parsing RBA F1 (money market)...")
    f1 = _parse_rba_file(RAW_DIR / "yield_curve_f1_money_market.csv")

    print("[DataLoader] Parsing RBA F2 (government bonds)...")
    f2 = _parse_rba_file(RAW_DIR / "yield_curve_f2_government_bonds.csv")

    # ── Extract target columns and drop rows with any missing values ───────
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

    # Guard against completely empty data (e.g. wrong file placed in raw/)
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
        # Take the last available observation on or before the as-of date
        val = f1_data.loc[f1_data.index <= as_of, col].iloc[-1]
        rows.append({
            "maturity_years": round(maturity, 4),
            "yield":          round(val / 100, 6),   # percent → decimal
        })

    for col, maturity in _F2_COLS.items():
        val = f2_data.loc[f2_data.index <= as_of, col].iloc[-1]
        rows.append({
            "maturity_years": maturity,
            "yield":          round(val / 100, 6),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("maturity_years")
        .reset_index(drop=True)
    )
    df["as_of_date"] = str(as_of.date())

    # ── Sanity checks ──────────────────────────────────────────────────────
    assert len(df) == len(_F1_COLS) + len(_F2_COLS), \
        f"Expected {len(_F1_COLS) + len(_F2_COLS)} yield curve points, got {len(df)}"
    assert (df["yield"] > 0).all(), \
        "One or more yields are non-positive — check raw data"
    assert (df["yield"] < 0.20).all(), \
        "One or more yields exceed 20% — possible parsing error (values still in percent?)"
    assert df["maturity_years"].is_monotonic_increasing, \
        "Maturities are not strictly increasing — sort error"

    # ── Cache and return ──────────────────────────────────────────────────
    df.to_csv(processed_path, index=False)
    print(f"[DataLoader] Yield curve saved → {processed_path.name}")
    print(df.to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# EQUITY PRICES
# ══════════════════════════════════════════════════════════════════════════════

def load_equity_prices(
    tickers: list = PORTFOLIO_TICKERS,
    start:   str  = EQUITY_START,
    end:     str  = EQUITY_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    
    """
    Download and cache daily Close prices for all portfolio tickers.
    """
    
    cache_path = RAW_DIR / "equity_prices_raw.csv"

    if _is_valid_cache(cache_path) and not force_refresh:
        print(f"[DataLoader] Loading equity prices from cache: {cache_path.name}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Reorder columns to match PORTFOLIO_TICKERS in case CSV order differs
        df = df.reindex(columns=tickers)
        return df

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed.  Run:  pip install yfinance"
        )

    print(
        f"[DataLoader] Fetching {tickers} from Yahoo Finance "
        f"({start} → {end})..."
    )
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits and dividends automatically
        progress=False,
    )

    # ── Extract Close prices from MultiIndex columns ───────────────────────
    # yfinance returns MultiIndex columns when downloading multiple tickers.
    # The structure is (price_type, ticker): e.g. ('Close', 'CBA.AX').
    # We select level 0 == 'Close' to get a plain DataFrame of Close prices.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]                           # level 0 = price type
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)      # level 1 = price type
        else:
            raise ValueError(
                "Cannot locate 'Close' prices in yfinance output — "
                "check yfinance version or ticker symbols."
            )
    else:
        # Single-column fallback (should not occur with multiple tickers)
        close = raw[["Close"]] if "Close" in raw.columns else raw

    # Enforce consistent column ordering matching PORTFOLIO_TICKERS
    close = close.reindex(columns=tickers)

    # Drop any rows where one or more tickers have no data (e.g. ASX holidays,
    # stock halts). This ensures all tickers share the same date index,
    # which is required for log return and correlation calculations.
    close = close.dropna()

    # ── Sanity checks ──────────────────────────────────────────────────────
    assert not close.empty, \
        "No equity data returned — check ticker symbols and date range"
    assert list(close.columns) == tickers, \
        f"Column mismatch: expected {tickers}, got {list(close.columns)}"
    assert close.isna().sum().sum() == 0, \
        "NaN values remain after dropna() — unexpected missing data"
    assert (close > 0).all().all(), \
        "Negative or zero prices detected — data corruption"

    # ── Cache and return ──────────────────────────────────────────────────
    close.to_csv(cache_path)
    print(f"[DataLoader] Equity prices cached → {cache_path.name}")
    print(f"  Trading days: {len(close)} | Date range: "
          f"{close.index[0].date()} → {close.index[-1].date()}")
    print("  Latest closing prices:")
    for t in tickers:
        print(f"    {t}: ${close[t].iloc[-1]:.2f}")

    return close


# ══════════════════════════════════════════════════════════════════════════════
# LOG RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(
    prices: pd.DataFrame,
    save:   bool = True,
) -> pd.DataFrame:
    
    """
    Compute daily log returns for all portfolio tickers simultaneously.

    Formula applied column-wise:  r_t = ln(S_t / S_{t-1})

    The first row is always NaN (no prior observation) and is dropped,
    so the returned DataFrame has len(prices) - 1 rows.
    """
    
    returns = np.log(prices / prices.shift(1)).dropna()

    # ── Sanity checks ──────────────────────────────────────────────────────
    assert not returns.empty, \
        "Log returns DataFrame is empty — check input price data"
    assert returns.notna().all().all(), \
        "NaN values remain in log returns after dropna()"
    assert list(returns.columns) == list(prices.columns), \
        "Returns columns do not match price columns"

    print(f"[DataLoader] Log returns computed ({len(returns)} observations):")
    for t in returns.columns:
        print(
            f"  {t}: mean = {returns[t].mean():.5f} | "
            f"std = {returns[t].std():.5f}"
        )

    if save:
        path = DATA_DIR / "equity_returns.csv"
        returns.to_csv(path)
        print(f"[DataLoader] Log returns saved → {path.name}")

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
    """
    daily_vol  = returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days)

    # ── Sanity checks ──────────────────────────────────────────────────────
    assert (annual_vol > 0).all(), \
        "One or more tickers have zero or negative volatility — check return data"
    assert (annual_vol < 2.0).all(), \
        "One or more tickers exceed 200% annualised volatility — possible data error"

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
        path = DATA_DIR / "equity_volatility.csv"
        summary.to_csv(path, index=False)
        print(f"[DataLoader] Volatility saved → {path.name}")

    return annual_vol


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(
    returns: pd.DataFrame,
    save:    bool = True,
) -> pd.DataFrame:
    """
    Compute the pairwise return correlation matrix across all portfolio tickers.

    Used by risk.py for multi-asset parametric VaR. The matrix is symmetric
    with 1.0 on the diagonal by construction.
    """
    corr = returns.corr()

    print(
        f"[DataLoader] Correlation matrix "
        f"({len(returns.columns)} assets, {len(returns)} observations):"
    )
    print(corr.round(4).to_string())

    if save:
        path = DATA_DIR / "correlation_matrix.csv"
        corr.to_csv(path)
        print(f"[DataLoader] Correlation matrix saved → {path.name}")

    return corr


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    tickers: list = PORTFOLIO_TICKERS,
    start:   str  = EQUITY_START,
    end:     str  = EQUITY_END,
    force_refresh: bool = False,
) -> dict:
    """
    Run the complete data preparation pipeline in one call.
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
    print(f"  Tickers    : {tickers}")
    print(f"  Date range : {start} → {end}")
    print(f"  Trading days: {len(prices)}")
    print("=" * 60 + "\n")

    return {
        "prices":       prices,
        "returns":      returns,
        "volatilities": volatilities,
        "correlation":  correlation,
        "yield_curve":  yield_curve,
    }
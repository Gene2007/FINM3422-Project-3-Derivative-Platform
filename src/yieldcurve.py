"""
yieldcurve.py
-------------
This section consists of the yield curve construction and interpoation for 
the FINM3422 derivatives platform.

Summary of Code:
Reads yield_curve_processed.csv (built by data_loader.py) and fits a cubic spline 
through the 7 RBA data points. Interpolates the data points and provides continuous 
zero rates and discount factors at any maturity, to be used by derivataive.py and
risk.py for discounting and pricing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

# ══════════════════════════════════════════════════════════════════════════════
# PATH
# ══════════════════════════════════════════════════════════════════════════════

_BASE_DIR              = Path(__file__).resolve().parent.parent
_YIELD_CURVE_PROCESSED = _BASE_DIR / "data" / "processed" / "yield_curve_processed.csv"



# ══════════════════════════════════════════════════════════════════════════════
# YIELD CURVE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class YieldCurve:
    """
    Cubic spline yield curve built from RBA zero rates.
 
    Loads the 7-point curve from data/processed/yield_curve_processed.csv,
    fits a CubicSpline, and exposes vectorised methods for zero rates,
    discount factors, and forward rates at any maturity.
 
    Parameters
    ----------
    csv_path : Path or str, optional
        Path to yield_curve_processed.csv. Defaults to the standard project path.
 
    Attributes
    ----------
    as_of_date : str
        The observation date of the underlying RBA data.
    maturities : np.ndarray
        The 7 raw maturity points in years.
    yields : np.ndarray
        The 7 raw zero rates as decimals (e.g. 0.0444).
    """
 
    __slots__ = ("as_of_date", "maturities", "yields", "_spline")
 
    def __init__(self, csv_path: Path = _YIELD_CURVE_PROCESSED) -> None:
        """Load curve data and fit cubic spline."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Yield curve data not found: {csv_path}\n"
                "Run data_loader.run_full_pipeline() first."
            )
 
        df = pd.read_csv(csv_path)
 
        # Validate required columns
        required = {"maturity_years", "yield", "as_of_date"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns. Expected: {required}")
 
        # Store raw data as numpy arrays for fast vectorised access
        self.maturities : np.ndarray = df["maturity_years"].to_numpy(dtype=float)
        self.yields     : np.ndarray = df["yield"].to_numpy(dtype=float)
        self.as_of_date : str        = df["as_of_date"].iloc[0]
 
        # Fit cubic spline — natural boundary conditions (second derivative = 0
        # at endpoints) prevent unrealistic oscillations outside the data range
        self._spline = CubicSpline(self.maturities, self.yields, bc_type="natural")
 
        print(
            f"[YieldCurve] Curve loaded as-of {self.as_of_date} "
            f"| {len(self.maturities)} knot points "
            f"| range: {self.maturities[0]:.4f}–{self.maturities[-1]:.0f} yrs"
        )

# ── Core methods ──────────────────────────────────────────────────────────
 
    def get_zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """
        Return the continuously compounded zero rate at the given maturity.
 
        Vectorised — accepts a scalar or any array-like of maturities.
 
        Parameters
        ----------
        maturity : float or array-like
            Time to maturity in years. Must be within [min_maturity, max_maturity].
 
        Returns
        -------
        float or np.ndarray
            Annualised zero rate as a decimal (e.g. 0.0472).
        """
        t = np.asarray(maturity, dtype=float)
        self._check_bounds(t)
        rate = self._spline(t)
        return float(rate) if rate.ndim == 0 else rate
 
    def get_discount_factor(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """
        Return the discount factor D(T) = exp(-r(T) * T).
 
        Vectorised — accepts a scalar or any array-like of maturities.
 
        Parameters
        ----------
        maturity : float or array-like
            Time to maturity in years.
 
        Returns
        -------
        float or np.ndarray
            Present value of $1 received at maturity (between 0 and 1).
        """
        t    = np.asarray(maturity, dtype=float)
        self._check_bounds(t)
        rate = self._spline(t)
        df   = np.exp(-rate * t)
        return float(df) if df.ndim == 0 else df
 
    def get_forward_rate(
        self,
        t1: float | np.ndarray,
        t2: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Return the implied forward rate between two maturities.
 
        Derived from: f(t1, t2) = (r(t2)*t2 - r(t1)*t1) / (t2 - t1)
 
        Parameters
        ----------
        t1 : float or array-like
            Start of the forward period in years.
        t2 : float or array-like
            End of the forward period in years. Must be greater than t1.
 
        Returns
        -------
        float or np.ndarray
            Continuously compounded forward rate as a decimal.
        """
        t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        if np.any(t2 <= t1):
            raise ValueError("t2 must be strictly greater than t1.")
        self._check_bounds(t1)
        self._check_bounds(t2)
 
        r1  = self._spline(t1)
        r2  = self._spline(t2)
        fwd = (r2 * t2 - r1 * t1) / (t2 - t1)
        return float(fwd) if fwd.ndim == 0 else fwd
 
    # ── Internal helpers ──────────────────────────────────────────────────────
 
    def _check_bounds(self, t: np.ndarray) -> None:
        """Raise ValueError if any maturity falls outside the curve's data range."""
        lo, hi = self.maturities[0], self.maturities[-1]
        if np.any(t < lo) or np.any(t > hi):
            raise ValueError(
                f"Maturity out of range [{lo:.4f}, {hi:.0f}] years. "
                "Extrapolation beyond RBA data is not supported."
            )
 
    def __repr__(self) -> str:
        return (
            f"YieldCurve(as_of='{self.as_of_date}', "
            f"knots={len(self.maturities)}, "
            f"range=[{self.maturities[0]:.4f}, {self.maturities[-1]:.0f}] yrs)"
        )
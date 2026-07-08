"""Shared price-transform helpers used by more than one router."""

import numpy as np
import pandas as pd


def inverse_log_return(eur: np.ndarray, log_returns: np.ndarray) -> np.ndarray:
    """Convert predicted log_return_7d values to absolute EUR prices.

    Applies the inverse of log1p: expm1(log1p(eur) + log_return).
    Returns NaN for elements where eur is NaN.
    """
    not_null = pd.notna(eur)
    log_eur = np.where(not_null, np.log1p(np.where(not_null, eur, 0.0)), np.nan)
    return np.where(not_null, np.expm1(log_eur + log_returns), np.nan)

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataValidationError


def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    if return_type not in {"log", "simple"}:
        raise ValueError("return_type must be 'log' or 'simple'.")

    if return_type == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change(fill_method=None)

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all")
    returns = returns.dropna(axis=1, how="all")
    if returns.shape[1] < 2:
        raise DataValidationError("Need at least two tickers with valid returns.")
    return returns


def correlation_distance(returns: pd.DataFrame) -> pd.DataFrame:
    corr = returns.corr(min_periods=2)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    common = corr.index.intersection(corr.columns)
    corr = corr.loc[common, common]

    if corr.shape[0] < 2:
        raise DataValidationError("Need at least two tickers with pairwise correlations.")

    distance = (1.0 - corr).clip(lower=0.0, upper=2.0).copy()
    for ticker in distance.index:
        distance.loc[ticker, ticker] = 0.0

    if distance.isna().any().any():
        bad = distance.columns[distance.isna().any()].tolist()
        raise DataValidationError(
            f"Distance matrix contains NaN values for tickers: {bad}. "
            "Increase data coverage or lower missingness before clustering."
        )

    return distance

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ExperimentConfig


class DataValidationError(ValueError):
    """Raised when stock input data cannot support the experiment contract."""


def load_stock_data(path: Path | str, config: ExperimentConfig) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Stock input file not found: {path}. Provide a CSV with columns "
            f"{config.date_col!r}, {config.ticker_col!r}, and {config.price_col!r}."
        )

    df = pd.read_csv(path)
    required = {config.date_col, config.ticker_col, config.price_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required stock columns: {missing}")

    df = df.copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df[config.price_col] = pd.to_numeric(df[config.price_col], errors="coerce")

    if df[config.date_col].isna().any():
        raise DataValidationError("Some stock rows have invalid dates.")
    if df[config.price_col].isna().any():
        raise DataValidationError("Some stock rows have non-numeric close prices.")
    if (df[config.price_col] <= 0).any():
        raise DataValidationError("Close prices must be strictly positive.")

    if config.volume_col and config.volume_col in df.columns:
        df[config.volume_col] = pd.to_numeric(df[config.volume_col], errors="coerce")

    duplicated = df.duplicated([config.date_col, config.ticker_col]).sum()
    if duplicated:
        raise DataValidationError(
            f"Found {duplicated} duplicate date/ticker rows. Deduplicate input first."
        )

    return df.sort_values([config.date_col, config.ticker_col]).reset_index(drop=True)


def prepare_price_matrix(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    work = df.copy()

    if config.start_date:
        work = work[work[config.date_col] >= pd.Timestamp(config.start_date)]
    if config.end_date:
        work = work[work[config.date_col] <= pd.Timestamp(config.end_date)]

    if work.empty:
        raise DataValidationError("No stock rows remain after date filtering.")

    if (
        config.min_average_volume is not None
        and config.volume_col
        and config.volume_col in work.columns
    ):
        avg_volume = work.groupby(config.ticker_col)[config.volume_col].mean()
        keep = avg_volume[avg_volume >= config.min_average_volume].index
        work = work[work[config.ticker_col].isin(keep)]

    prices = work.pivot(
        index=config.date_col,
        columns=config.ticker_col,
        values=config.price_col,
    ).sort_index()

    if prices.empty or prices.shape[1] < 2:
        raise DataValidationError("Need at least two tickers with valid prices.")

    observation_ratio = prices.notna().mean()
    keep_cols = observation_ratio[
        observation_ratio >= config.min_observation_ratio
    ].index.tolist()
    prices = prices[keep_cols]

    if config.forward_fill_limit > 0:
        prices = prices.ffill(limit=config.forward_fill_limit)

    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if prices.shape[1] < 2:
        raise DataValidationError(
            "Need at least two tickers after observation and liquidity filters."
        )
    if len(prices) < 3:
        raise DataValidationError("Need at least three dates to compute returns.")

    return prices


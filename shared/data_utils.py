"""Shared data utilities: date parsing, resampling, alignment, and splitting."""

import pandas as pd


def parse_monthly_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Convert a date column to a monthly DatetimeIndex (Month Start)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index = df.index.to_period("M").to_timestamp("MS")
    return df


def resample_to_monthly(series: pd.Series, agg: str = "last") -> pd.Series:
    """Resample a series to monthly frequency."""
    return getattr(series.resample("MS"), agg)()


def align_series(*series: pd.Series, how: str = "inner") -> pd.DataFrame:
    """Align multiple series into a DataFrame by date with a configurable join."""
    return pd.concat(series, axis=1, join=how).dropna(how="all")


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / val / test by cut-off dates."""
    train = df.loc[:train_end]
    val   = df.loc[train_end:val_end].iloc[1:]   # exclude the cut-off point
    test  = df.loc[val_end:].iloc[1:]
    return train, val, test


def freeze_snapshot(df: pd.DataFrame, path: str, tag: str) -> None:
    """Save a versioned parquet snapshot of a DataFrame."""
    import os
    os.makedirs(path, exist_ok=True)
    df.to_parquet(f"{path}/{tag}.parquet")

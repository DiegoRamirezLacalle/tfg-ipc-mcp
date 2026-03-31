"""
Utilidades de datos compartidas: parsers de fechas, resampling, alineación.
"""

import pandas as pd


def parse_monthly_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Convierte la columna de fecha a DatetimeIndex mensual (Month Start)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index = df.index.to_period("M").to_timestamp("MS")
    return df


def resample_to_monthly(series: pd.Series, agg: str = "last") -> pd.Series:
    """Resamplea una serie a frecuencia mensual."""
    return getattr(series.resample("MS"), agg)()


def align_series(*series: pd.Series, how: str = "inner") -> pd.DataFrame:
    """Alinea múltiples series en un DataFrame por fecha, con join configurable."""
    return pd.concat(series, axis=1, join=how).dropna(how="all")


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en train / val / test por fechas de corte."""
    train = df.loc[:train_end]
    val   = df.loc[train_end:val_end].iloc[1:]   # excluye el punto de corte
    test  = df.loc[val_end:].iloc[1:]
    return train, val, test


def freeze_snapshot(df: pd.DataFrame, path: str, tag: str) -> None:
    """Guarda un snapshot versionado del DataFrame (parquet + tag)."""
    import os
    os.makedirs(path, exist_ok=True)
    df.to_parquet(f"{path}/{tag}.parquet")

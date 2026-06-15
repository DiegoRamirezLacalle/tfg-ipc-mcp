"""MCP exogenous feature builder and Ridge residual corrector.

Two public functions:

  build_mcp_exog(series_index, future_signals)
      Merges pre-computed historical MCP signals from parquet with the
      future signals retrieved from MongoDB for the forecast horizon.
      Returns a DataFrame aligned to series_index, or None when the
      parquet file is not mounted in the container.

  apply_ridge_correction(base_preds, y_train, full_exog, test_index)
      Post-hoc correction for foundation models that lack native exog
      support (TimesFM, Chronos-2).  Fits Ridge(MCP signals -> target)
      on the training period and applies a damped level correction to
      the zero-shot base predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

_PARQUET_PATH = Path("/app/tfg-forecasting/data/processed/mcp_signals_global.parquet")
_SIGNAL_CACHE: pd.DataFrame | None = None


def _load_signal_history() -> pd.DataFrame | None:
    global _SIGNAL_CACHE
    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE
    if not _PARQUET_PATH.exists():
        log.warning("mcp_parquet_missing", path=str(_PARQUET_PATH))
        return None
    df = pd.read_parquet(_PARQUET_PATH)
    df.index = pd.to_datetime(df.index)
    _SIGNAL_CACHE = df
    return _SIGNAL_CACHE


def load_signal_history() -> pd.DataFrame | None:
    """Public accessor for the cached MCP signal history parquet.

    Returns the raw monthly signal DataFrame (DatetimeIndex), or None when
    the parquet is not mounted in the container.
    """
    return _load_signal_history()


def build_mcp_exog(
    series_index: pd.DatetimeIndex,
    future_signals: list[dict] | None = None,
) -> pd.DataFrame | None:
    """Build a complete MCP signal DataFrame aligned to series_index.

    Combines historical signals (parquet) with future signals stored in
    MongoDB after MCP fetch.  Returns None when the parquet is absent.
    """
    hist = _load_signal_history()
    if hist is None:
        return None

    base = hist.copy()

    if future_signals:
        rows: list[dict] = []
        for sig in future_signals:
            ym = sig.get("year_month")
            if not ym:
                continue
            if sig.get("available") is False or "error" in sig:
                continue
            try:
                ts = pd.Timestamp(ym + "-01")
            except Exception:
                continue
            row = {
                k: v
                for k, v in sig.items()
                if k not in ("year_month", "available") and v is not None
            }
            if row:
                rows.append({"__ts__": ts, **row})

        if rows:
            future_df = pd.DataFrame(rows).set_index("__ts__")
            future_df.index = pd.DatetimeIndex(future_df.index)
            future_df = future_df.reindex(columns=base.columns)
            base = pd.concat([base, future_df])
            base = base[~base.index.duplicated(keep="last")].sort_index()

    aligned = base.reindex(series_index).ffill().fillna(0.0)

    useful = [c for c in aligned.columns if float(aligned[c].std()) > 1e-8]
    if not useful:
        log.warning("mcp_exog_no_useful_columns")
        return None

    return aligned[useful]


def apply_ridge_correction(
    base_preds: np.ndarray,
    y_train: pd.Series,
    full_exog: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    alpha: float = 1.0,
    damping: float = 0.15,
) -> np.ndarray:
    """Additive Ridge correction for models without native exog support.

    Fits Ridge(historical_signals -> historical_target) on the training
    period.  Correction is:

        delta[i] = damping * (signal_future[i] - mean(signal_train[-12:]))

    where signal_* is the Ridge-implied inflation level from MCP features.
    The damping factor (0.15) limits overcorrection in out-of-distribution
    regimes.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return base_preds

    exog_train = full_exog.reindex(y_train.index).ffill().fillna(0.0)
    exog_future = full_exog.reindex(test_index).ffill().fillna(0.0)

    if exog_train.shape[0] < 12 or exog_future.shape[0] < 1:
        return base_preds

    X_train = exog_train.values.astype(np.float64)
    y_vals = y_train.values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y_vals)

    # Signal-implied inflation level: mean of last 12 training months
    signal_train_level = float(
        ridge.predict(scaler.transform(X_train[-12:])).mean()
    )

    X_future = scaler.transform(exog_future.values.astype(np.float64))
    signal_future = ridge.predict(X_future)  # shape (h,)

    correction = (signal_future - signal_train_level) * damping
    return (base_preds + correction).astype(np.float64)

"""Ridge regression adapter with ECB exogenous features.

Direct multi-step forecasting strategy:
  - For each horizon step d = 1..H, a separate Ridge_d model is trained.
  - Features: AR lags of the target [y[t-1], y[t-2], y[t-3], y[t-12]]
              + ECB rate features [dfr, mrr, dfr_diff, dfr_lag3, dfr_lag6, dfr_lag12]
              at the same time step t (no look-ahead — rates are published mid-month
              and are public knowledge before the IPC release).
  - All features are standardised with sklearn StandardScaler.
  - Falls back to AR-only mode when exogenous data is not available.

This matches the C1_inst experimental condition from the research:
  StandardScaler + Ridge(alpha=1.0) + ECB DFR/MRR features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app.forecasting.base import ForecastAdapter, ForecastInput, ForecastResult

AR_LAGS = [1, 2, 3, 12]
EXOG_COLS = ["dfr", "mrr", "dfr_diff", "dfr_lag3", "dfr_lag6", "dfr_lag12"]
_ALPHA = 1.0
_MIN_TRAIN = max(AR_LAGS) + 2   # at least 14 obs before the first valid target


class RidgeExogAdapter:
    slug = "ridge-exog"

    def run(self, inp: ForecastInput) -> ForecastResult:
        h = inp.horizon
        y = inp.series.sort_index()
        exog = inp.exog.reindex(y.index) if inp.exog is not None else None

        n = len(y)
        if n < _MIN_TRAIN + h:
            raise ValueError(
                f"ridge-exog requires at least {_MIN_TRAIN + h} observations, got {n}"
            )

        train_end = n - h
        y_train = y.iloc[:train_end]
        y_test = y.iloc[train_end:]

        # Columns actually present in exog (may be None or partial)
        live_exog_cols = (
            [c for c in EXOG_COLS if c in exog.columns]
            if exog is not None
            else []
        )

        def _row(i: int) -> list[float]:
            ar = [float(y.iloc[i - lag]) for lag in AR_LAGS]
            ex: list[float] = []
            if live_exog_cols and exog is not None:
                row = exog.iloc[i]
                ex = [0.0 if pd.isna(row[c]) else float(row[c]) for c in live_exog_cols]
            return ar + ex

        # Direct multi-step: train one Ridge per forecast horizon step
        models: list[Ridge] = []
        scalers: list[StandardScaler] = []
        max_lag = max(AR_LAGS)

        for d in range(1, h + 1):
            valid_indices = range(max_lag, train_end - d + 1)
            if len(valid_indices) < 2:
                raise ValueError(
                    f"Not enough training samples for direct step d={d} "
                    f"(need ≥2, got {len(valid_indices)})"
                )

            X = np.array([_row(i) for i in valid_indices], dtype=np.float64)
            y_target = np.array(
                [float(y.iloc[i + d]) for i in valid_indices], dtype=np.float64
            )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            ridge = Ridge(alpha=_ALPHA, fit_intercept=True)
            ridge.fit(X_scaled, y_target)

            models.append(ridge)
            scalers.append(scaler)

        # Forecast from origin = last training index
        origin = train_end - 1
        x_origin = np.array([_row(origin)], dtype=np.float64)

        preds = np.array(
            [float(m.predict(s.transform(x_origin))[0]) for m, s in zip(models, scalers)],
            dtype=np.float64,
        )

        return ForecastResult(
            predictions=preds,
            timestamps=list(y_test.index),
            train_actuals=y_train.values.astype(np.float64),
            test_actuals=y_test.values.astype(np.float64),
            model_slug=self.slug,
        )


_: ForecastAdapter = RidgeExogAdapter()  # type: ignore[assignment]

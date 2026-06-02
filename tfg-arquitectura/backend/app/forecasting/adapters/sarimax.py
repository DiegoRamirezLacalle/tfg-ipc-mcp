"""SARIMAX adapter with ECB exogenous features.

SARIMA(0,1,1)(0,1,1)12 base + ECB Deposit Facility Rate (dfr) and Main
Refinancing Rate (mrr) as exogenous regressors. If exog is absent or
missing these columns, falls back to pure SARIMA behaviour.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.forecasting.base import ForecastAdapter, ForecastInput, ForecastResult

ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)
EXOG_COLS = ["dfr", "mrr"]
_MIN_TRAIN = 24


class SARIMAXAdapter:
    slug = "sarimax"

    def run(self, inp: ForecastInput) -> ForecastResult:
        series = inp.series.sort_index()
        h = inp.horizon
        n = len(series)

        if n < _MIN_TRAIN + h:
            raise ValueError(f"Need at least {_MIN_TRAIN + h} observations, got {n}")

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        exog_train: np.ndarray | None = None
        exog_fc: np.ndarray | None = None

        if inp.exog is not None:
            exog = inp.exog.reindex(series.index)
            avail = [c for c in EXOG_COLS if c in exog.columns]
            if avail:
                exog_all = exog[avail].ffill().fillna(0.0)
                exog_train = exog_all.iloc[:-h].values
                exog_fc = exog_all.iloc[-h:].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train.values,
                exog=exog_train,
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)

        preds = result.forecast(steps=h, exog=exog_fc)

        return ForecastResult(
            predictions=np.asarray(preds),
            timestamps=list(y_test.index),
            train_actuals=y_train.values,
            test_actuals=y_test.values,
            model_slug=self.slug,
        )


_: ForecastAdapter = SARIMAXAdapter()  # type: ignore[assignment]

"""SARIMA(0,1,1)(0,1,1)12 adapter.

Orders from auto_arima on Spain IPC (02_sarima_seasonal.py / 04_backtesting_rolling.py).
Matches the research baseline exactly.
"""
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.forecasting.base import ForecastInput, ForecastResult

ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)
_MIN_TRAIN = 24  # two full cycles before fitting


class SARIMAAdapter:
    slug = "sarima"

    def run(self, inp: ForecastInput) -> ForecastResult:
        series = inp.series.sort_index()
        h = inp.horizon
        n = len(series)

        if n < _MIN_TRAIN + h:
            raise ValueError(f"Need at least {_MIN_TRAIN + h} observations, got {n}")

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        # Pass values as numpy to avoid pandas freq-string conflicts in statsmodels 0.14+
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train.values,
                order=ORDER,
                seasonal_order=SEASONAL_ORDER,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)

        preds = result.forecast(steps=h)

        return ForecastResult(
            predictions=np.asarray(preds),
            timestamps=list(y_test.index),
            train_actuals=y_train.values,
            test_actuals=y_test.values,
            model_slug=self.slug,
        )

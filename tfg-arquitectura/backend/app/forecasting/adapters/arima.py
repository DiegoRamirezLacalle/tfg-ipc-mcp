"""ARIMA(1,1,1) adapter.

Non-seasonal baseline matching the research rolling evaluation in
04_backtesting_rolling.py. Uses SARIMAX with no seasonal component -
numerically identical to statsmodels ARIMA.
"""
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.forecasting.base import ForecastInput, ForecastResult

ORDER = (1, 1, 1)
_MIN_TRAIN = 12


class ARIMAAdapter:
    slug = "arima"

    def run(self, inp: ForecastInput) -> ForecastResult:
        series = inp.series.sort_index()
        h = inp.horizon
        n = len(series)

        if n < _MIN_TRAIN + h:
            raise ValueError(f"Need at least {_MIN_TRAIN + h} observations, got {n}")

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train.values,
                order=ORDER,
                seasonal_order=(0, 0, 0, 0),
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

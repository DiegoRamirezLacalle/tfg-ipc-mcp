"""AutoARIMA adapter using pmdarima.

Automatically selects non-seasonal ARIMA orders via stepwise AIC search.
Falls back gracefully if pmdarima is not installed.
"""
import warnings

import numpy as np

from app.forecasting.base import ForecastInput, ForecastResult

_MIN_TRAIN = 24


class AutoARIMAAdapter:
    slug = "auto-arima"

    def run(self, inp: ForecastInput) -> ForecastResult:
        try:
            import pmdarima as pm
        except ImportError as exc:
            raise ImportError(
                "pmdarima is required for AutoARIMA — rebuild the backend image"
            ) from exc

        series = inp.series.sort_index()
        h = inp.horizon
        n = len(series)

        if n < _MIN_TRAIN + h:
            raise ValueError(f"Need at least {_MIN_TRAIN + h} observations, got {n}")

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                y_train.values,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3,
                max_q=3,
                max_d=2,
                information_criterion="aic",
            )

        preds = model.predict(n_periods=h)

        return ForecastResult(
            predictions=np.asarray(preds),
            timestamps=list(y_test.index),
            train_actuals=y_train.values,
            test_actuals=y_test.values,
            model_slug=self.slug,
        )

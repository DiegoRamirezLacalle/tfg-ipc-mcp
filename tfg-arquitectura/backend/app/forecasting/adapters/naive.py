"""Seasonal naïve baseline: y_hat[t+s] = y[t+s-12]."""
import numpy as np

from app.forecasting.base import ForecastInput, ForecastResult

_MIN_TRAIN = 13  # need at least one full seasonal cycle


class NaiveSeasonalAdapter:
    slug = "naive-seasonal"

    def run(self, inp: ForecastInput) -> ForecastResult:
        series = inp.series.sort_index()
        h = inp.horizon
        n = len(series)

        if n < _MIN_TRAIN + h:
            raise ValueError(f"Need at least {_MIN_TRAIN + h} observations, got {n}")

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        preds = np.array([
            float(y_train.iloc[-12 + ((s) % 12)])
            for s in range(h)
        ])

        return ForecastResult(
            predictions=preds,
            timestamps=list(y_test.index),
            train_actuals=y_train.values,
            test_actuals=y_test.values,
            model_slug=self.slug,
        )

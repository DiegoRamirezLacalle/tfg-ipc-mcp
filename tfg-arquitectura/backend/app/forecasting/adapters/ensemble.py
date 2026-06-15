"""Ensemble-stack adapter - inverse-MAE weighted combiner.

Creates a combined forecast from N completed runs by weighting each model's
predictions by the inverse of its MAE. A run with lower MAE gets more weight.

Usage: set experiment config to {"stack_run_ids": [1, 2, 3]}.
The backend's _execute_forecast loads those runs' predictions and MAEs,
builds stack_preds and stack_weights, then passes them here.
"""

from __future__ import annotations

import numpy as np

from app.forecasting.base import ForecastAdapter, ForecastInput, ForecastResult


class EnsembleStackAdapter:
    slug = "ensemble-stack"

    def run(self, inp: ForecastInput) -> ForecastResult:
        if inp.stack_preds is None or inp.stack_preds.empty:
            raise ValueError(
                "ensemble-stack requires stack_preds loaded by the run executor. "
                "Set config={'stack_run_ids': [run_id_1, run_id_2, ...]} on the experiment."
            )

        mat = inp.stack_preds.values.astype(np.float64)  # (horizon, n_models)
        h = inp.horizon

        if mat.shape[0] != h:
            raise ValueError(
                f"stack_preds has {mat.shape[0]} rows but experiment horizon is {h}"
            )

        if inp.stack_weights is not None:
            w = np.asarray(inp.stack_weights, dtype=np.float64)
            w = w / w.sum()
            preds = mat @ w
        else:
            preds = mat.mean(axis=1)

        series  = inp.series.sort_index()
        y_train = series.iloc[:-h]
        y_test  = series.iloc[-h:]

        return ForecastResult(
            predictions=preds,
            timestamps=list(inp.stack_preds.index),
            train_actuals=y_train.values.astype(np.float64),
            test_actuals=y_test.values.astype(np.float64),
            model_slug=self.slug,
        )


_: ForecastAdapter = EnsembleStackAdapter()  # type: ignore[assignment]

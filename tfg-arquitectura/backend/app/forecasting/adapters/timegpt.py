"""TimeGPT (Nixtla) adapter - API-based foundation model forecasting.

C0 (use_mcp=False): zero-shot forecast, no exogenous features.
C1_mcp (use_mcp=True): forecast with MCP signal history + future signals
  passed as X_df to NixtlaClient.  Nixtla handles the exog alignment.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.forecasting.base import ForecastAdapter, ForecastInput, ForecastResult

_SERIES_ID = "series"
_MIN_TRAIN = 12


class TimeGPTAdapter:
    slug = "timegpt"

    def run(self, inp: ForecastInput) -> ForecastResult:
        try:
            from nixtla import NixtlaClient  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "nixtla is not installed. Install with: pip install nixtla"
            ) from exc

        api_key = os.getenv("NIXTLA_API_KEY", "")
        if not api_key:
            raise ValueError(
                "NIXTLA_API_KEY environment variable is not set. "
                "Set it to your Nixtla API key to use TimeGPT."
            )

        h = inp.horizon
        series = inp.series.sort_index()
        if len(series) < _MIN_TRAIN + h:
            raise ValueError(
                f"TimeGPT requires at least {_MIN_TRAIN + h} observations, "
                f"got {len(series)}"
            )

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        df_input = pd.DataFrame({
            "unique_id": _SERIES_ID,
            "ds": y_train.index,
            "y": y_train.values,
        })

        # Build X_df covering train + forecast horizon when exog is available
        X_df: pd.DataFrame | None = None
        if inp.exog is not None:
            train_exog = inp.exog.reindex(y_train.index).ffill().fillna(0.0)
            future_exog = inp.exog.reindex(y_test.index).ffill().fillna(0.0)
            both = pd.concat([train_exog, future_exog])
            X_df = pd.DataFrame({"unique_id": _SERIES_ID, "ds": both.index})
            for col in both.columns:
                X_df[col] = both[col].values

        client = NixtlaClient(api_key=api_key)
        kwargs: dict = dict(
            df=df_input,
            h=h,
            freq="MS",
            time_col="ds",
            target_col="y",
            id_col="unique_id",
        )
        if X_df is not None:
            kwargs["X_df"] = X_df

        fc = client.forecast(**kwargs)
        fc = fc.sort_values("ds").reset_index(drop=True)
        preds = np.asarray(fc["TimeGPT"].values[:h], dtype=np.float64)

        return ForecastResult(
            predictions=preds,
            timestamps=list(y_test.index),
            train_actuals=y_train.values.astype(np.float64),
            test_actuals=y_test.values.astype(np.float64),
            model_slug=self.slug,
        )


_: ForecastAdapter = TimeGPTAdapter()  # type: ignore[assignment]

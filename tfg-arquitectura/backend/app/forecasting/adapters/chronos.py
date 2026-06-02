"""Chronos-2 (Amazon) adapter — zero-shot probabilistic forecasting (p50 point estimate).

C0 (exog=None): pure zero-shot.
C1_mcp (exog provided): zero-shot base + Ridge residual correction from MCP signals.
"""

from __future__ import annotations

import threading

import numpy as np

from app.forecasting.base import ForecastAdapter, ForecastInput, ForecastResult

_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_ID = "amazon/chronos-2"
_P50_IDX = 10  # median in the 21 quantiles output by Chronos-2
_MIN_TRAIN = 24


def _load_model(chronos_mod):
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = chronos_mod.Chronos2Pipeline.from_pretrained(
                    _MODEL_ID,
                    device_map="cpu",
                )
    return _MODEL


class Chronos2Adapter:
    slug = "chronos-2"

    def run(self, inp: ForecastInput) -> ForecastResult:
        try:
            import torch  # noqa: PLC0415
            from chronos import Chronos2Pipeline as _  # noqa: F401, PLC0415
            import chronos as chronos_mod  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting and torch are required. "
                "Install with: pip install chronos-forecasting torch"
            ) from exc

        h = inp.horizon
        series = inp.series.sort_index()
        if len(series) < _MIN_TRAIN + h:
            raise ValueError(
                f"Chronos-2 requires at least {_MIN_TRAIN + h} observations, "
                f"got {len(series)}"
            )

        y_train = series.iloc[:-h]
        y_test = series.iloc[-h:]

        model = _load_model(chronos_mod)
        context = torch.tensor(y_train.values, dtype=torch.float32)

        # preds shape: (n_variates=1, n_quantiles=21, prediction_length=h)
        preds_raw = model.predict([context], prediction_length=h)
        quantiles = preds_raw[0].numpy()  # (1, 21, h)
        p50 = quantiles[0, _P50_IDX, :]   # (h,) — median

        # Ridge correction from MCP signals (C1_mcp condition)
        if inp.exog is not None:
            from app.forecasting.mcp_exog import apply_ridge_correction  # noqa: PLC0415
            p50 = apply_ridge_correction(
                base_preds=np.asarray(p50, dtype=np.float64),
                y_train=y_train,
                full_exog=inp.exog,
                test_index=y_test.index,
            )

        return ForecastResult(
            predictions=np.asarray(p50, dtype=np.float64),
            timestamps=list(y_test.index),
            train_actuals=y_train.values.astype(np.float64),
            test_actuals=y_test.values.astype(np.float64),
            model_slug=self.slug,
        )


_: ForecastAdapter = Chronos2Adapter()  # type: ignore[assignment]

"""TimesFM (Google) adapter — zero-shot univariate forecasting.

Uses timesfm 1.x API:  TimesFm(hparams=..., checkpoint=...) → .forecast()

C0 (exog=None): pure zero-shot.
C1_mcp (exog provided): zero-shot base + Ridge residual correction from MCP signals.
"""

from __future__ import annotations

import threading

import numpy as np

from app.forecasting.base import ForecastAdapter, ForecastInput, ForecastResult

_MODEL_LOCK = threading.Lock()
_MODEL = None
_REPO_ID  = "google/timesfm-1.0-200m-pytorch"
_MIN_TRAIN = 24


def _load_model(timesfm_mod):
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = timesfm_mod.TimesFm(
                    hparams=timesfm_mod.TimesFmHparams(
                        backend="pytorch",
                        per_core_batch_size=1,
                        horizon_len=12,
                        context_len=512,
                    ),
                    checkpoint=timesfm_mod.TimesFmCheckpoint(
                        huggingface_repo_id=_REPO_ID,
                    ),
                )
    return _MODEL


class TimesFMAdapter:
    slug = "timesfm"

    def run(self, inp: ForecastInput) -> ForecastResult:
        try:
            import timesfm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "timesfm is not installed. Install with: pip install timesfm"
            ) from exc

        h = inp.horizon
        series = inp.series.sort_index()
        if len(series) < _MIN_TRAIN + h:
            raise ValueError(
                f"TimesFM requires at least {_MIN_TRAIN + h} observations, "
                f"got {len(series)}"
            )

        y_train = series.iloc[:-h]
        y_test  = series.iloc[-h:]

        model = _load_model(timesfm)
        context = y_train.values.astype(np.float32).tolist()

        # freq=0 → low-frequency (monthly / quarterly)
        point_out, _ = model.forecast(inputs=[context], freq=[0])
        preds = np.asarray(point_out[0][:h], dtype=np.float64)

        # Ridge correction from MCP signals (C1_mcp condition)
        if inp.exog is not None:
            from app.forecasting.mcp_exog import apply_ridge_correction  # noqa: PLC0415
            preds = apply_ridge_correction(
                base_preds=preds,
                y_train=y_train,
                full_exog=inp.exog,
                test_index=y_test.index,
            )

        return ForecastResult(
            predictions=preds,
            timestamps=list(y_test.index),
            train_actuals=y_train.values.astype(np.float64),
            test_actuals=y_test.values.astype(np.float64),
            model_slug=self.slug,
        )


_: ForecastAdapter = TimesFMAdapter()  # type: ignore[assignment]

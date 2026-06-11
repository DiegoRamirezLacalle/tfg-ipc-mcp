from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass
class ForecastInput:
    series: pd.Series                        # DatetimeIndex, monthly float values
    horizon: int                             # steps ahead to forecast
    config: dict[str, Any] = field(default_factory=dict)
    exog: pd.DataFrame | None = None         # optional exogenous features, same DatetimeIndex
    # for ensemble-stack: cols=run_ids, index=test timestamps
    stack_preds: pd.DataFrame | None = None
    stack_weights: np.ndarray | None = None  # per-model weights (inverse MAE); None = equal weights


@dataclass
class ForecastResult:
    predictions: np.ndarray    # shape (horizon,) — forecast values
    timestamps: list            # pd.Timestamp list matching predictions
    train_actuals: np.ndarray  # y_train used to fit
    test_actuals: np.ndarray   # held-out y_test (predictions evaluated against these)
    model_slug: str


class ForecastAdapter(Protocol):
    slug: str

    def run(self, inp: ForecastInput) -> ForecastResult: ...

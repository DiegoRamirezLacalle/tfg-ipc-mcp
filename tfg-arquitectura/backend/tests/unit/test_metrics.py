"""Unit tests for the shared evaluation metrics (shared/metrics.py).

Pure-function tests: no DB, no network, deterministic inputs.
"""

import numpy as np
import pytest

from shared.metrics import diebold_mariano, mae, mase, rmse


def test_mae_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0


def test_mae_and_rmse_known_values():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert rmse(y_true, y_pred) == pytest.approx(1.0)
    # RMSE penalises a single large error more than MAE does.
    y_spike = np.array([0.0, 0.0, 0.0, 4.0])
    assert rmse(y_true, y_spike) > mae(y_true, y_spike)


def test_mase_scales_by_seasonal_naive():
    # With m=1 the naive scale = mean|diff| of [0,1,2,3,4] = 1.0,
    # so MASE = MAE / 1.0. MAE here is |12 - 10| = 2.0.
    y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert mase(np.array([10.0]), np.array([12.0]), y_train, m=1) == pytest.approx(2.0)


def test_mase_raises_when_scale_is_zero():
    y_train = np.array([5.0, 5.0, 5.0, 5.0])  # constant -> naive scale 0
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), y_train, m=1)


def test_diebold_mariano_favours_lower_error_model():
    # model 1 errors are an order of magnitude smaller than model 2's.
    e1 = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 0.1, -0.05, 0.12])
    e2 = np.array([1.0, -1.2, 0.9, -1.1, 1.05, 0.95, -1.15, 1.1, -0.9, 1.0])
    res = diebold_mariano(e1, e2, h=1, power=2)
    assert set(res) == {"dm_stat", "p_value", "better"}
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["dm_stat"] < 0          # negative -> model 1 has lower loss
    assert res["better"] == "model1"

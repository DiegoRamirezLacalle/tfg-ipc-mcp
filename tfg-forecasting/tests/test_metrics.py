"""Unit tests for shared.metrics against hand-computed values.

These run without any data artifacts, so they are safe for CI; the
artifact-dependent integrity suite lives in check_artifacts_and_leakage.py.
"""

import numpy as np
import pytest

from shared.metrics import diebold_mariano, mae, mase, rmse, summary


Y_TRUE = np.array([1.0, 2.0, 3.0])
Y_PRED = np.array([2.0, 2.0, 5.0])  # abs errors: 1, 0, 2


def test_mae_known_value():
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(1.0)


def test_rmse_known_value():
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_mase_known_value():
    # y_train = 0..23 → seasonal naive (lag 12) errors are all exactly 12
    y_train = np.arange(24, dtype=float)
    assert mase(Y_TRUE, Y_PRED, y_train, m=12) == pytest.approx(1.0 / 12.0)


def test_mase_perfect_forecast_is_zero():
    y_train = np.arange(24, dtype=float)
    assert mase(Y_TRUE, Y_TRUE, y_train, m=12) == 0.0


def test_mase_constant_train_raises():
    # A constant series has zero naive scale; MASE must refuse to divide
    with pytest.raises(ValueError):
        mase(Y_TRUE, Y_PRED, np.ones(24), m=12)


def test_diebold_mariano_detects_better_model():
    rng = np.random.RandomState(42)
    e_small = rng.normal(0.0, 0.1, 100)
    e_large = rng.normal(0.0, 2.0, 100)

    res = diebold_mariano(e_small, e_large, h=1)
    assert res["better"] == "model1"
    assert res["p_value"] < 0.05

    # symmetric call must flag the other model
    res_sym = diebold_mariano(e_large, e_small, h=1)
    assert res_sym["better"] == "model2"
    assert res_sym["dm_stat"] == pytest.approx(-res["dm_stat"], abs=1e-9)


def test_diebold_mariano_tie_on_same_distribution():
    rng = np.random.RandomState(0)
    e1 = rng.normal(0.0, 1.0, 200)
    e2 = rng.normal(0.0, 1.0, 200)
    assert diebold_mariano(e1, e2, h=1)["better"] == "tie"


def test_summary_matches_individual_metrics():
    y_train = np.arange(24, dtype=float)
    s = summary(Y_TRUE, Y_PRED, y_train, m=12)
    assert s["MAE"] == pytest.approx(mae(Y_TRUE, Y_PRED))
    assert s["RMSE"] == pytest.approx(rmse(Y_TRUE, Y_PRED))
    assert s["MASE"] == pytest.approx(mase(Y_TRUE, Y_PRED, y_train, m=12))

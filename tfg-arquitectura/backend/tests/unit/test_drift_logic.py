"""Unit tests for the KS drift computation (app/api/v1/drift.compute_drift_stats).

Tests the pure residual logic directly, with no DB or HTTP layer.
"""

from app.api.v1.drift import compute_drift_stats


def test_insufficient_residuals_returns_neutral_result():
    res = compute_drift_stats([0.1, 0.2, 0.1])  # n = 3 < 4
    assert res["drifted"] is False
    assert res["p_value"] is None
    assert res["ks_statistic"] is None
    assert res["n_early"] == 3
    assert res["n_recent"] == 0


def test_drift_detected_on_shifted_distribution():
    # Early window sits tight around 0; recent window is shifted far away,
    # so the two samples never overlap -> KS statistic 1.0, tiny p-value.
    residuals = [0.0, 0.01, -0.01, 0.0, 0.02, -0.02, 0.0,   # early (7)
                 5.0, 5.1, 4.9, 5.0, 5.2]                    # recent (5)
    res = compute_drift_stats(residuals)
    assert res["drifted"] is True
    assert res["ks_statistic"] == 1.0
    assert res["p_value"] < 0.05
    assert res["n_early"] == 7
    assert res["n_recent"] == 5


def test_no_drift_on_homogeneous_residuals():
    residuals = [0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1]
    res = compute_drift_stats(residuals)
    assert res["drifted"] is False
    assert res["p_value"] is not None
    assert res["n_early"] + res["n_recent"] == len(residuals)

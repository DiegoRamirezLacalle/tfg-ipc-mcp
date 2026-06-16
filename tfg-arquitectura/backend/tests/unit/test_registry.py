"""Unit tests for the forecasting adapter registry.

Verifies the registry is internally consistent and exposes the expected
models, without running any forecast.
"""

import pytest

from app.forecasting.registry import get_adapter, supported_slugs


def test_supported_slugs_consistent_with_adapters():
    slugs = supported_slugs()
    assert len(slugs) >= 8
    # Every registered slug must resolve to an adapter that reports that slug.
    for slug in slugs:
        assert get_adapter(slug).slug == slug


def test_known_models_are_registered():
    slugs = set(supported_slugs())
    expected = {
        "naive-seasonal", "sarima", "sarimax", "ridge-exog",
        "timesfm", "chronos-2", "timegpt", "ensemble-stack",
    }
    assert expected <= slugs


def test_get_adapter_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_adapter("not-a-real-model")

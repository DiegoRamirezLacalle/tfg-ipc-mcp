from app.forecasting.adapters.arima import ARIMAAdapter
from app.forecasting.adapters.auto_arima import AutoARIMAAdapter
from app.forecasting.adapters.chronos import Chronos2Adapter
from app.forecasting.adapters.ensemble import EnsembleStackAdapter
from app.forecasting.adapters.naive import NaiveSeasonalAdapter
from app.forecasting.adapters.ridge import RidgeExogAdapter
from app.forecasting.adapters.sarima import SARIMAAdapter
from app.forecasting.adapters.sarimax import SARIMAXAdapter
from app.forecasting.adapters.timegpt import TimeGPTAdapter
from app.forecasting.adapters.timesfm import TimesFMAdapter
from app.forecasting.base import ForecastAdapter

_REGISTRY: dict[str, ForecastAdapter] = {
    NaiveSeasonalAdapter.slug: NaiveSeasonalAdapter(),
    ARIMAAdapter.slug: ARIMAAdapter(),
    AutoARIMAAdapter.slug: AutoARIMAAdapter(),
    SARIMAAdapter.slug: SARIMAAdapter(),
    SARIMAXAdapter.slug: SARIMAXAdapter(),
    RidgeExogAdapter.slug: RidgeExogAdapter(),
    TimesFMAdapter.slug: TimesFMAdapter(),
    Chronos2Adapter.slug: Chronos2Adapter(),
    TimeGPTAdapter.slug: TimeGPTAdapter(),
    EnsembleStackAdapter.slug: EnsembleStackAdapter(),
}


def get_adapter(slug: str) -> ForecastAdapter:
    adapter = _REGISTRY.get(slug)
    if adapter is None:
        raise KeyError(slug)
    return adapter


def supported_slugs() -> list[str]:
    return list(_REGISTRY.keys())

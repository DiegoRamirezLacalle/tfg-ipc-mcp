"""What-if counterfactual simulator.

GET /api/v1/whatif/setup?series_id=<id>&horizon=<h>

Returns everything the frontend needs for an instant, client-side
signal-sensitivity simulation:

  - history  : last 36 observed points (chart context)
  - baseline : pure time-series base forecast (ARIMA) for `horizon` steps
  - signals  : macro / monetary-policy levers, each with a per-horizon
               marginal effect (change in the target per +1 unit move in
               the signal at the forecast origin)

Each effect comes from a direct multi-step Ridge fit on the h-step change
y[t+d] - y[t], not the raw level: fitting on changes isolates the signal's
contribution to future movements instead of recovering shared level trends.
Signals are standardised before the fit, matching the C1 conditions.

Ridge is linear, so the frontend reconstructs any counterfactual exactly
(no server round-trip per slider move):

    counterfactual[d] = baseline[d] + sum_i (slider_i - baseline_i) * effect_i[d]

Lever sources: ECB deposit rate (dfr) from the features-exog dataset;
FOMC / US CPI signals from mcp_signals_global.parquet (varying columns only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.permissions import get_current_user
from app.forecasting.mcp_exog import load_signal_history
from app.models.dataset import Dataset, Observation, Series
from app.models.user import User

router = APIRouter(tags=["whatif"])
log = structlog.get_logger()

_HISTORY_POINTS = 36
_RIDGE_ALPHA = 1.0
_MIN_TRAIN = 24

# Curated levers. `source` selects where the column is loaded from:
#   "ecb" -> Postgres features-exog dataset | "mcp" -> mcp_signals_global.parquet
# Only columns that actually vary in the data survive into the response.
_LEVERS: list[dict] = [
    {
        "key": "dfr",
        "source": "ecb",
        "label": "ECB Deposit Rate",
        "hint": "ECB policy rate (%) - the main euro-area lever",
    },
    {
        "key": "fomc_hawkish_score",
        "source": "mcp",
        "label": "Fed (FOMC) Stance",
        "hint": "0 = dovish | 1 = hawkish",
    },
    {
        "key": "fomc_forward_guidance_num",
        "source": "mcp",
        "label": "Fed Forward Guidance",
        "hint": "-1 = easing path | +1 = tightening path",
    },
    {
        "key": "us_cpi_direction_num",
        "source": "mcp",
        "label": "US CPI Direction",
        "hint": "-1 = decelerating | +1 = accelerating",
    },
]

_ECB_KEYS = [lv["key"] for lv in _LEVERS if lv["source"] == "ecb"]
_MCP_KEYS = [lv["key"] for lv in _LEVERS if lv["source"] == "mcp"]


def _base_forecast(y: pd.Series, horizon: int) -> np.ndarray:
    """Pure time-series base forecast. ARIMA with graceful fallbacks."""
    values = y.values.astype(np.float64)
    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415

        for order in [(2, 1, 1), (1, 1, 1), (1, 1, 0)]:
            try:
                fitted = ARIMA(values, order=order).fit()
                fc = np.asarray(fitted.forecast(steps=horizon), dtype=np.float64)
                if np.all(np.isfinite(fc)):
                    return fc
            except Exception:  # noqa: BLE001
                continue
    except ImportError:
        pass

    drift = float(np.mean(np.diff(values[-13:]))) if len(values) > 13 else 0.0
    last = float(values[-1])
    return np.array([last + drift * (d + 1) for d in range(horizon)], dtype=np.float64)


def _per_step_change_effects(
    signals: pd.DataFrame, y: pd.Series, horizon: int
) -> dict[str, list[float]]:
    """One Ridge per horizon step d: signals[t] -> (y[t+d] - y[t]).

    Fitting on the h-step *change* (not the level) isolates each signal's genuine
    contribution to future movements and avoids spurious level correlation.

    Returns {signal_key: [effect_d1, ..., effect_dh]} where each effect is the
    change in y (original units) per +1 unit change in that signal.
    """
    from sklearn.linear_model import Ridge  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    cols = list(signals.columns)
    effects: dict[str, list[float]] = {c: [] for c in cols}

    X_all = signals.values.astype(np.float64)
    y_all = y.values.astype(np.float64)
    n = len(y_all)

    for d in range(1, horizon + 1):
        if n - d < _MIN_TRAIN:
            for c in cols:
                effects[c].append(0.0)
            continue

        X = X_all[: n - d]
        target = y_all[d:] - y_all[: n - d]  # h-step change

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = Ridge(alpha=_RIDGE_ALPHA)
        ridge.fit(X_scaled, target)

        scale = scaler.scale_.copy()
        scale[scale == 0.0] = 1.0
        per_unit = ridge.coef_ / scale

        for i, c in enumerate(cols):
            effects[c].append(round(float(per_unit[i]), 6))

    return effects


async def _load_ecb_levers(db: AsyncSession, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load ECB lever series (e.g. dfr) from the features-exog dataset."""
    out = pd.DataFrame(index=index)
    ds = await db.scalar(select(Dataset).where(Dataset.slug == "features-exog"))
    if not ds:
        return out
    for key in _ECB_KEYS:
        s = await db.scalar(
            select(Series).where(Series.dataset_id == ds.id, Series.slug == key)
        )
        if not s:
            continue
        rows = await db.execute(
            select(Observation)
            .where(Observation.series_id == s.id)
            .order_by(Observation.timestamp)
        )
        obs = rows.scalars().all()
        if not obs:
            continue
        sidx = pd.DatetimeIndex([o.timestamp for o in obs])
        if sidx.tz is not None:
            sidx = sidx.tz_localize(None)
        ser = pd.Series([float(o.value) for o in obs], index=sidx)
        out[key] = ser.reindex(index).ffill().bfill()
    return out


@router.get("/whatif/setup")
async def whatif_setup(
    series_id: int = Query(..., description="Target series to simulate"),
    horizon: int = Query(6, ge=1, le=24, description="Forecast horizon (months)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    series = await db.get(Series, series_id)
    if not series:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Series not found")

    obs_rows = await db.execute(
        select(Observation)
        .where(Observation.series_id == series_id)
        .order_by(Observation.timestamp)
    )
    obs = obs_rows.scalars().all()
    if len(obs) < _MIN_TRAIN + horizon:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Series needs at least {_MIN_TRAIN + horizon} observations, has {len(obs)}",
        )

    idx = pd.DatetimeIndex([o.timestamp for o in obs])
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    y = pd.Series([float(o.value) for o in obs], index=idx, name="value").sort_index()

    # Base time-series forecast (signal-independent)
    base = _base_forecast(y, horizon)
    last_ts = y.index[-1]
    future_idx = pd.date_range(
        start=last_ts + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    baseline = [
        {"timestamp": ts.isoformat(), "value": round(float(v), 4)}
        for ts, v in zip(future_idx, base)
    ]
    history = [
        {"timestamp": ts.isoformat(), "value": round(float(v), 4)}
        for ts, v in zip(y.index[-_HISTORY_POINTS:], y.values[-_HISTORY_POINTS:])
    ]

    # -- Assemble lever matrix from both sources, aligned to the target index --
    combo = pd.DataFrame(index=y.index)
    combo = combo.join(await _load_ecb_levers(db, y.index))

    hist = load_signal_history()
    if hist is not None:
        for key in _MCP_KEYS:
            if key in hist.columns:
                combo[key] = (
                    hist[key].reindex(y.index).ffill().bfill().fillna(0.0)
                )

    signals_out: list[dict] = []
    varying = [c for c in combo.columns if float(combo[c].std()) > 1e-8]
    if varying:
        aligned = combo[varying].ffill().bfill().fillna(0.0)
        effects = _per_step_change_effects(aligned, y, horizon)
        lever_by_key = {lv["key"]: lv for lv in _LEVERS}
        for key in [lv["key"] for lv in _LEVERS if lv["key"] in varying]:
            lv = lever_by_key[key]
            col = aligned[key]
            lo, hi = float(col.min()), float(col.max())
            span = hi - lo
            pad = span * 0.1 if span > 0 else 0.5
            signals_out.append(
                {
                    "key": key,
                    "label": lv["label"],
                    "hint": lv["hint"],
                    "min": round(lo - pad, 4),
                    "max": round(hi + pad, 4),
                    "step": round(max(span, 0.5) / 40.0, 4),
                    "baseline_value": round(float(col.iloc[-1]), 4),
                    "effect_per_step": effects[key],
                }
            )

    return {
        "series_id": series_id,
        "series_name": series.name,
        "unit": series.unit,
        "horizon": horizon,
        "history": history,
        "baseline": baseline,
        "signals": signals_out,
        "signals_available": bool(signals_out),
    }

"""Explicit policies for supplying exogenous / contextual variables to a
forecast made at a given origin, *without* consuming information published
after that origin.

Motivation
----------
In rolling-origin backtesting the model standing at `origin = t` may only use
data with timestamp <= t. Two subtle leakage bugs have appeared in this project:

  * passing the *realised* future path of a "known" covariate (e.g. the ECB
    rate at t+1..t+h) instead of what was actually known at t;
  * computing an exogenous correction from covariates without making the
    "as-of-origin" cut explicit.

This module makes the intent explicit. Pick a policy per use-case:

  * ``CARRY_FORWARD``  - repeat the last value observed at/<= origin across the
    whole horizon. Use for "known-future" covariates whose future path is
    genuinely unknown at origin (ECB/Fed rates, etc.).
  * ``NEUTRAL``        - use a neutral constant (historical mean over the
    window <= origin by default). Use when you want the covariate to contribute
    nothing relative to "average conditions".
  * ``KNOWN_AT_ORIGIN``- the single vector observed exactly at origin. Use for a
    point-in-time correction (e.g. a Ridge nowcast of the next change).
  * ``FORWARD_PATH``   - a genuine *forecast* of the covariate's future path
    (damped random-walk-with-drift) instead of a flat line. The drift is the
    mean monthly change over the trailing ``drift_window`` months and fades by
    ``phi`` per step, so recent momentum is propagated without the runaway
    extrapolation of an undamped RW-drift. Uses only data <= origin.

All helpers slice with ``df.loc[:origin]`` and therefore never read a row dated
after ``origin``. Call :func:`assert_no_future` in tests to enforce this.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class ExogPolicy(str, Enum):
    """How to materialise covariate values for the forecast horizon."""

    CARRY_FORWARD = "carry_forward"
    NEUTRAL = "neutral"
    KNOWN_AT_ORIGIN = "known_at_origin"
    FORWARD_PATH = "forward_path"


# Defaults for the FORWARD_PATH policy (fixed a priori; not tuned on test data).
DEFAULT_DRIFT_WINDOW = 12
DEFAULT_PHI = 0.85


def damped_rw_drift_path(
    values: np.ndarray,
    horizon: int,
    drift_window: int = DEFAULT_DRIFT_WINDOW,
    phi: float = DEFAULT_PHI,
) -> np.ndarray:
    """Damped random-walk-with-drift forecast for a single covariate series.

    ``x_{T+k} = x_T + drift * sum_{j=0..k-1} phi^j`` where ``drift`` is the mean
    monthly change over the trailing ``drift_window`` months. With ``phi < 1``
    the per-step increment fades, preventing runaway extrapolation while still
    propagating recent momentum (vs a flat carry-forward).

    ``values`` must already be the as-of-origin series (index <= origin); this
    function never looks past the array it is given.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    values = np.asarray(values, dtype=np.float64)
    last = float(values[-1])
    tail = values[-(drift_window + 1):]
    diffs = np.diff(tail)
    drift = float(np.nanmean(diffs)) if diffs.size else 0.0
    out = np.empty(horizon, dtype=np.float64)
    cum = 0.0
    weight = 1.0
    for k in range(horizon):
        cum += weight * drift
        out[k] = last + cum
        weight *= phi
    return out


def _window(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Rows with index <= origin (the only data legitimately known at origin)."""
    window = df.loc[:origin]
    if window.empty:
        raise ValueError(f"No data at or before origin {origin!r}.")
    return window


def value_at_origin(df: pd.DataFrame, cols: list[str], origin: pd.Timestamp) -> np.ndarray:
    """Return the covariate vector (len = len(cols)) observed at/<= ``origin``.

    Uses the last available row up to ``origin`` (forward-fills implicitly by
    taking ``.iloc[-1]`` of the as-of window), so a missing value exactly at
    ``origin`` falls back to the most recent known value — never the future.
    """
    window = _window(df, origin)
    return window[cols].ffill().iloc[-1].to_numpy(dtype=np.float64)


def build_future_covariates(
    df: pd.DataFrame,
    cols: list[str],
    origin: pd.Timestamp,
    horizon: int,
    policy: ExogPolicy = ExogPolicy.CARRY_FORWARD,
    *,
    fillna: float | None = None,
    drift_window: int = DEFAULT_DRIFT_WINDOW,
    phi: float = DEFAULT_PHI,
) -> dict[str, np.ndarray]:
    """Future covariate arrays (each of length ``horizon``) under ``policy``.

    Returned as ``{col: np.ndarray(horizon)}`` — the shape expected by
    Chronos-2's ``future_covariates``. Only data with index <= ``origin`` is
    read, so there is no look-ahead regardless of the chosen policy.

    Parameters
    ----------
    df       : feature frame indexed by a monotonically increasing DatetimeIndex.
    cols     : covariate column names.
    origin   : forecast origin; the horizon is ``origin+1 .. origin+horizon``.
    horizon  : number of future steps.
    policy   : CARRY_FORWARD (last known value), NEUTRAL (window mean),
               KNOWN_AT_ORIGIN (value at origin, repeated), or FORWARD_PATH
               (damped RW-with-drift forecast of the path).
    fillna   : if given, fill residual NaNs in the as-of window with this value
               before computing (matches the flat-hold scripts that ``fillna(0)``).
    drift_window, phi : FORWARD_PATH hyperparameters (see ``damped_rw_drift_path``).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    window = _window(df, origin)

    out: dict[str, np.ndarray] = {}
    for col in cols:
        if col not in window.columns:
            raise KeyError(f"Column {col!r} not in frame.")
        series = window[col]
        series = series.fillna(fillna) if fillna is not None else series.ffill()
        if policy == ExogPolicy.FORWARD_PATH:
            out[col] = damped_rw_drift_path(
                series.to_numpy(dtype=np.float64), horizon, drift_window, phi
            )
            continue
        if policy in (ExogPolicy.CARRY_FORWARD, ExogPolicy.KNOWN_AT_ORIGIN):
            val = float(series.iloc[-1])
        elif policy == ExogPolicy.NEUTRAL:
            val = float(series.mean())
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"Unknown policy {policy!r}.")
        out[col] = np.full(horizon, val, dtype=np.float64)
    return out


def assert_no_future(df: pd.DataFrame, origin: pd.Timestamp) -> None:
    """Raise ``AssertionError`` if ``df`` contains any row dated after ``origin``.

    Use in tests/checks to assert that the slice handed to a model was cut
    as-of the origin (i.e. ``df = full.loc[:origin]``).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("assert_no_future expects a DatetimeIndex.")
    latest = df.index.max()
    assert latest <= pd.Timestamp(origin), (
        f"Future-data leakage: frame reaches {latest.date()} "
        f"but the forecast origin is {pd.Timestamp(origin).date()}."
    )


if __name__ == "__main__":  # lightweight self-test
    idx = pd.date_range("2002-01-01", "2024-12-01", freq="MS")
    demo = pd.DataFrame({"dfr": np.linspace(-0.5, 4.0, len(idx))}, index=idx)
    org = pd.Timestamp("2022-06-01")

    cf = build_future_covariates(demo, ["dfr"], org, 12, ExogPolicy.CARRY_FORWARD)
    assert cf["dfr"].shape == (12,)
    assert np.allclose(cf["dfr"], demo.loc[org, "dfr"])  # flat = last known
    neu = build_future_covariates(demo, ["dfr"], org, 12, ExogPolicy.NEUTRAL)
    assert np.allclose(neu["dfr"], demo.loc[:org, "dfr"].mean())
    assert_no_future(demo.loc[:org], org)
    try:
        assert_no_future(demo, org)  # full frame reaches 2024 -> must fail
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise SystemExit("assert_no_future failed to catch leakage")
    print("exog_policies self-test OK")

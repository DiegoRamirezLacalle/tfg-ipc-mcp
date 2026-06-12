"""Shared constants for tfg-forecasting and tfg-arquitectura."""

# Experiment date windows — frozen; changing them invalidates every backtest
DATE_START     = "2002-01-01"   # first available historical date
DATE_TRAIN_END = "2020-12-01"   # end of training set
DATE_VAL_END   = "2022-06-01"   # end of validation set
DATE_TEST_END  = "2024-12-01"   # end of test set

# Forecast horizon (months)
FORECAST_HORIZON = 12

# Frequency
FREQ = "MS"   # Month Start (pandas)

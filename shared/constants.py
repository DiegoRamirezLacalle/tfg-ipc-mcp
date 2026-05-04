"""Shared constants for tfg-forecasting and tfg-arquitectura."""

# Series identifiers
SERIES_IPC_SPAIN = "IPC_ESP"
SERIES_HICP_EA   = "HICP_EA"
SERIES_ECB_RATE  = "ECB_RATE"

ALL_SERIES = [SERIES_IPC_SPAIN, SERIES_HICP_EA, SERIES_ECB_RATE]

# Experiment date windows
DATE_START     = "2002-01-01"   # first available historical date
DATE_TRAIN_END = "2020-12-01"   # end of training set
DATE_VAL_END   = "2022-06-01"   # end of validation set
DATE_TEST_END  = "2024-12-01"   # end of test set

# Forecast horizon (months)
FORECAST_HORIZON = 12

# Experimental conditions
CONDITION_C0 = "C0"   # numeric history only
CONDITION_C1 = "C1"   # history + MCP signals

# Frequency
FREQ = "MS"   # Month Start (pandas)

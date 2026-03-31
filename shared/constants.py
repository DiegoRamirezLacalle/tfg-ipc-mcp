"""
Constantes compartidas entre tfg-forecasting y tfg-arquitectura.
Editar aquí afecta a ambos TFGs de forma automática.
"""

# ── Identificadores de series ────────────────────────────────
SERIES_IPC_SPAIN = "IPC_ESP"
SERIES_HICP_EA   = "HICP_EA"
SERIES_ECB_RATE  = "ECB_RATE"

ALL_SERIES = [SERIES_IPC_SPAIN, SERIES_HICP_EA, SERIES_ECB_RATE]

# ── Ventanas temporales del experimento ──────────────────────
DATE_START        = "2002-01-01"   # inicio histórico disponible
DATE_TRAIN_END    = "2020-12-01"   # fin del set de entrenamiento
DATE_VAL_END      = "2022-06-01"   # fin de validación
DATE_TEST_END     = "2024-12-01"   # fin del set de test

# Horizonte de predicción (meses)
FORECAST_HORIZON  = 12

# ── Condiciones experimentales ───────────────────────────────
CONDITION_C0 = "C0"   # solo histórico numérico
CONDITION_C1 = "C1"   # histórico + señales MCP

# ── Frecuencia ───────────────────────────────────────────────
FREQ = "MS"   # Month Start (pandas)

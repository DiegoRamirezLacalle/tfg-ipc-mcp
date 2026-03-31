# TFG IPC-MCP — Monorepo

Doble TFG de ingeniería: predicción de series temporales económicas con contexto semántico via MCP.

## Estructura

```
tfg-ipc-mcp/
├── tfg-forecasting/      # TFG 1 — Ciencia de datos
├── tfg-arquitectura/     # TFG 2 — Plataforma web
├── shared/               # Código común (métricas, utilidades, constantes)
├── pyproject.toml        # Dependencias Python compartidas
├── docker-compose.yml    # Orquestación completa
└── .env.example          # Variables de entorno (copiar a .env)
```

## Inicio rápido

```bash
cp .env.example .env
# Editar .env con las credenciales reales

# Levantar infraestructura
docker compose up -d postgres mongo

# Instalar dependencias Python (entorno virtual recomendado)
pip install -e ".[dev]"

# Ejecutar ETL inicial
python tfg-forecasting/01_etl/01_ingest_ipc_spain.py
```

## Condiciones experimentales

| Condición | Descripción |
|-----------|-------------|
| C0        | Modelo entrenado solo con histórico numérico |
| C1        | Modelo con señales exógenas extraídas de noticias via MCP |

## Modelos evaluados

ARIMA → SARIMA → SARIMAX → LSTM → TimeGPT → TimesFM

Evaluación: backtesting rolling-origin con MAE / RMSE / MASE + test Diebold-Mariano.

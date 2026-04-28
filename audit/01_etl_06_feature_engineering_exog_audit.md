# Audit — 01_etl/06_feature_engineering_exog.py

## Propósito
Une `ipc_spain_index.parquet` (solo `indice_general`) con `ecb_rates_monthly.parquet`, genera `dfr_diff`, `dfr_lag3`, `dfr_lag6`, `dfr_lag12`, e imprime correlaciones con el target.

## Inputs / Outputs
- **Reads**: `data/processed/ipc_spain_index.parquet`, `data/processed/ecb_rates_monthly.parquet`.
- **Writes**: `data/processed/features_exog.parquet`.

## Dependencias internas
- Ninguna.

## Métricas
- LOC totales: 63
- Funcionales: ~40
- Comentarios/docstrings/blank: ~23
- Sin debug prints excesivos (solo info).

## Code smells
- Docstrings y comentarios en español.
- Output `features_exog.parquet` no aparece referenciado en PROJECT_CONTEXT (puede haber sido reemplazado por `features_c1.parquet`). Verificar que aún lo use alguien antes de tocar la lógica.
- `df.index.freq = "MS"` directo: sirve, pero asume continuidad — `pd.infer_freq` o `pd.date_range` reindex sería más robusto. No tocar (paridad).

## Riesgo de refactor
**BAJO-MEDIO**. Si nadie consume `features_exog.parquet` puede llegar a ser código muerto. Si lo consume algún SARIMAX legacy, hay que preservar nombres y orden de columnas.

## Acciones propuestas (FASE 3)
1. Verificar consumidores de `features_exog.parquet` con grep.
2. Si hay consumidores, traducir docstrings + logger; mantener orden y nombres de columnas.
3. Si NO hay consumidores, candidato a eliminación (proponer al usuario antes).

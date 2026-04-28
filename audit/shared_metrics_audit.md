# Audit — shared/metrics.py

## Propósito
Funciones de evaluación: `mae`, `rmse`, `mase`, `diebold_mariano` (DM-HLN simplificado) y un `summary` que devuelve dict con MAE/RMSE/MASE.

## Inputs / Outputs
- **Reads**: nada (funciones puras sobre `np.ndarray`).
- **Writes**: nada.

## Dependencias internas
- Ninguna; solo `numpy` y `scipy.stats`.

## Métricas
- LOC totales: 75
- Funcionales: ~40
- Comentarios/docstrings/blank: ~35
- Sin debug.

## Code smells
- Docstrings y docstring de módulo en español.
- `mase` recalcula la escala naive estacional cada llamada con el `y_train` pasado. La convención del TFG fija la escala MASE sobre 2002-2020 (Spain: 1.4051; Global: 1.1720; Europe: 1.4558) — se usan factores precomputados, así que es probable que esta función `mase` no se use o se llame solo en algún sitio aislado.
- `diebold_mariano` aquí es una versión propia (sin corrección HLN explícita para muestras pequeñas). Los scripts oficiales `07_evaluation/01_diebold_mariano_tests.py` y `05_diebold_mariano_europe.py` parecen reimplementar el test → revisar si son redundantes o intencionalmente diferentes.
- En `diebold_mariano`, la decisión `better` está hardcoded a 5% — opaco y poco testeable; aceptable pero merece comentario.
- `np.roll` para autocovarianza: válido si los lags se usan correctamente, pero la fórmula no documenta el factor `(n - k)/n` típico de HLN.
- `summary` con kwarg `m=12` está bien.

## Riesgo de refactor
**MEDIO**. Si `mae`, `rmse` o `mase` se usan para calcular números que ya están publicados en `08_results/*.json`, cualquier cambio en redondeo, en la fórmula naive o en signos rompe la comparación. Hay que verificar TODOS los call sites antes de tocarlas.

## Acciones propuestas (FASE 3)
1. Traducir docstrings y comentarios a inglés.
2. Verificar quién llama a `mase` y `diebold_mariano` — confirmar si los scripts de DM las usan o si reimplementan in-place.
3. Si la versión oficial de DM en producción es la de `07_evaluation/`, marcar `diebold_mariano` aquí como deprecate o eliminar tras confirmación.
4. Conservar nombres y firmas: `mae`, `rmse`, `mase`, `diebold_mariano`, `summary`.
5. No tocar las fórmulas (paridad numérica).

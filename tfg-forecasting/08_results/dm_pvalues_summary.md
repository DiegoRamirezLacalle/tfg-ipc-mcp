# Diebold-Mariano p-values - main C0 vs C1 comparisons

> DM (Harvey-Leybourne-Newbold corrected), MAE-based (power=1). model1 = C0, model2 = C1. `better` = C1 if the C1 (with-signals) model has lower error, C0 otherwise. Significance: ** p<0.05, * p<0.10, ns = not significant. `[VERIFY]` = forecast not available (cannot be tested).

| Series | Track | Family | C1 signal | h=1 | h=3 | h=6 | h=12 |
|--------|-------|--------|-----------|-----|-----|-----|------|
| Spain | Foundation | timesfm | mcp | 0.746 (C0) | 0.286 (C0) | 0.015** (C0) | 0.000** (C0) |
| Spain | Foundation | chronos2 | mcp | 0.593 (C0) | 0.663 (C0) | 0.121 (C0) | 0.001** (C0) |
| Spain | Foundation | timegpt | mcp | 0.006** (C0) | 0.019** (C0) | 0.007** (C0) | 0.005** (C0) |
| Spain | Foundation | chronos2 | energy | 0.025** (C0) | 0.089* (C0) | 0.447 (C0) | 0.783 (C0) |
| Spain | Foundation | timegpt | energy | 0.016** (C0) | 0.028** (C0) | 0.007** (C0) | 0.000** (C0) |
| Spain | Foundation | chronos2 | energy_only | 0.275 (C0) | 0.575 (C0) | 0.936 (C0) | 0.684 (C1) |
| Spain | Foundation | timegpt | energy_only | 0.000** (C0) | 0.002** (C0) | 0.003** (C0) | 0.001** (C0) |
| Spain | Foundation | chronos2 | inst | 0.522 (C0) | 0.952 (C0) | 0.639 (C0) | 0.477 (C0) |
| Spain | Foundation | timesfm | inst | 0.268 (C0) | 0.005** (C0) | 0.015** (C0) | 0.028** (C0) |
| Spain | Foundation | timegpt | inst | 0.002** (C0) | 0.016** (C0) | 0.015** (C0) | 0.017** (C0) |
| Spain | Foundation | chronos2 | macro | 0.134 (C0) | 0.224 (C0) | 0.439 (C0) | 0.768 (C0) |
| Spain | Foundation | timesfm | macro | 0.092* (C0) | 0.397 (C0) | 0.575 (C0) | 0.360 (C0) |
| Spain | Foundation | timegpt | macro | 0.006** (C0) | 0.013** (C0) | 0.001** (C0) | 0.000** (C0) |
| Europe | Foundation | chronos2 | inst | 0.607 (C0) | 0.762 (C0) | 0.588 (C0) | 0.637 (C0) |
| Europe | Foundation | chronos2 | mcp | 0.004** (C0) | 0.028** (C0) | 0.082* (C0) | 0.304 (C0) |
| Europe | Foundation | chronos2 | full | 0.341 (C0) | 0.378 (C0) | 0.456 (C0) | 0.917 (C0) |
| Europe | Foundation | timesfm | inst | 0.124 (C0) | 0.647 (C1) | 0.385 (C1) | 0.947 (C1) |
| Europe | Foundation | timesfm | mcp | 0.028** (C0) | 0.774 (C0) | 0.631 (C1) | 0.794 (C0) |
| Europe | Foundation | timesfm | full | 0.090* (C0) | 0.889 (C1) | 0.445 (C1) | 0.755 (C1) |
| Europe | Foundation | timegpt | inst | 0.001** (C0) | 0.001** (C0) | 0.000** (C0) | 0.000** (C0) |
| Europe | Foundation | timegpt | mcp | 0.005** (C0) | 0.004** (C0) | 0.009** (C0) | 0.065* (C0) |
| Europe | Foundation | timegpt | full | 0.001** (C0) | 0.001** (C0) | 0.000** (C0) | 0.000** (C0) |
| Global | Foundation | timesfm | inst | 0.867 (C0) | 0.281 (C1) | 0.171 (C1) | 0.267 (C1) |
| Global | Foundation | chronos2 | inst | 0.117 (C1) | 0.740 (C1) | 0.653 (C1) | 0.463 (C1) |
| Global | Foundation | timegpt | inst | [VERIFY] | [VERIFY] | [VERIFY] | [VERIFY] |
| Europe | Classical (ARIMAX) | sarimax | inst | 0.551 (C0) | 0.361 (C1) | 0.031** (C1) | 0.061* (C1) |
| Europe | Classical (ARIMAX) | sarimax | full | 0.820 (C1) | 0.567 (C1) | 0.345 (C1) | 0.266 (C1) |

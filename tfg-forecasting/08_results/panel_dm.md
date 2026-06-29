# Panel Diebold–Mariano — pooled across Spain + Global + Europe (Chronos-2)

> Phase A. Pools the three CPI series into one test to gain power. Loss differential `d = |e_A| - |e_B|` (>0 => B better) is scaled per series by its MASE denominator so the units are commensurable, then tested with a **cluster-robust mean test clustered by origin** (panel-DM analog; df = clusters-1). `dbar` = pooled scaled mean (in MASE units). ** p<0.05, * p<0.10. Per-series columns are raw dMAE% (B vs A; negative = B better).

| Contrast | h | n | clusters | pooled dbar | dir | t | p | sig | Spain | Global | Europe |
|---|--:|--:|--:|--:|---|--:|--:|---|--:|--:|--:|
| C1 context vs C0 | 1 | 141 | 47 | -0.0011 | A>B | -0.050 | 0.9606 | ns | 5.5 | -20.4 | 7.6 |
| C1 context vs C0 | 3 | 405 | 45 | -0.0029 | A>B | -0.093 | 0.9261 | ns | 0.4 | -4.5 | 3.6 |
| C1 context vs C0 | 6 | 756 | 42 | -0.0087 | A>B | -0.185 | 0.8540 | ns | 2.6 | -7.9 | 5.5 |
| C1 context vs C0 | 12 | 1296 | 36 | 0.0131 | B>A | 0.157 | 0.8762 | ns | 4.0 | -14.5 | 4.4 |
| C1 context vs C0 | 1+3+6+12 | 2598 | 47 | 0.0035 | B>A | 0.059 | 0.9535 | ns | 3.4 | -12.7 | 4.6 |
| C1 forward-path vs C0 | 1 | 141 | 47 | -0.0000 | A>B | -0.001 | 0.9990 | ns | 4.2 | -23.6 | 10.0 |
| C1 forward-path vs C0 | 3 | 405 | 45 | 0.0169 | B>A | 0.657 | 0.5148 | ns | -1.1 | -14.1 | -0.2 |
| C1 forward-path vs C0 | 6 | 756 | 42 | 0.0301 | B>A | 0.789 | 0.4345 | ns | 1.0 | -17.5 | -0.3 |
| C1 forward-path vs C0 | 12 | 1296 | 36 | 0.0621 | B>A | 0.967 | 0.3399 | ns | 5.0 | -20.4 | -1.5 |
| C1 forward-path vs C0 | 1+3+6+12 | 2598 | 47 | 0.0424 | B>A | 0.913 | 0.3662 | ns | 3.5 | -19.5 | -1.0 |
| C1 forward vs flat-hold | 1 | 141 | 47 | 0.0010 | B>A | 0.124 | 0.9015 | ns | -1.3 | -4.0 | 2.3 |
| C1 forward vs flat-hold | 3 | 405 | 45 | 0.0198 | B>A | 1.587 | 0.1197 | ns | -1.5 | -10.1 | -3.7 |
| C1 forward vs flat-hold | 6 | 756 | 42 | 0.0388 | B>A | 1.838 | 0.0734 | * | -1.5 | -10.4 | -5.6 |
| C1 forward vs flat-hold | 12 | 1296 | 36 | 0.0490 | B>A | 1.346 | 0.1869 | ns | 1.0 | -6.9 | -5.7 |
| C1 forward vs flat-hold | 1+3+6+12 | 2598 | 47 | 0.0389 | B>A | 1.503 | 0.1396 | ns | 0.1 | -7.8 | -5.4 |

## Caveat (read before citing)

- This pools only **three** series. Clustering by origin gives ~36–47 *time* clusters (adequate for the t-reference), but the **cross-sectional** dimension is still 3 — a genuinely robust panel needs ~20–40 countries (Phase B). Treat a clean pooled point estimate here as the green light for Phase B, not as definitive inference.
- B = the with-context / forward-path model, A = the reference (C0 or flat-hold). `dir = B>A` means context/forward-path has the lower error.
- No retraining: pooled from stored Chronos-2 forecasts only.

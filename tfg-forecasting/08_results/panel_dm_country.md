# Phase B - euro-area country panel DM (19 countries, Chronos-2)

> Powered version of Phase A: pools ~19 euro-area HICP countries (shared euro covariates) instead of 3 series. Loss differential `d=|e_A|-|e_B|` (>0 => B better) scaled per country by its MASE denominator, then a cluster-robust mean test. `p_honest` = the more conservative of origin-clustered (time, ~47 clusters x19 countries) and country-clustered (19 clusters). `B>A` means context/forward-path has lower error. ** p<0.05, * p<0.10.

| Contrast | h | n | pooled dbar | dir | p (time) | p (country) | p honest | sig | countries B better |
|---|--:|--:|--:|---|--:|--:|--:|---|--:|
| C1 context vs C0 | 1 | 893 | -0.0078 | A>B | 0.7466 | 0.7189 | 0.7466 | ns | 8/19 |
| C1 context vs C0 | 3 | 2565 | -0.0078 | A>B | 0.7962 | 0.7715 | 0.7962 | ns | 8/19 |
| C1 context vs C0 | 6 | 4788 | -0.0060 | A>B | 0.8776 | 0.8845 | 0.8845 | ns | 7/19 |
| C1 context vs C0 | 12 | 8208 | 0.0626 | B>A | 0.2773 | 0.4028 | 0.4028 | ns | 9/19 |
| C1 context vs C0 | 1+3+6+12 | 16454 | 0.0278 | B>A | 0.5363 | 0.6029 | 0.6029 | ns | 10/19 |
| C1 forward-path vs C0 | 1 | 893 | 0.0042 | B>A | 0.8292 | 0.8132 | 0.8292 | ns | 8/19 |
| C1 forward-path vs C0 | 3 | 2565 | 0.0231 | B>A | 0.3258 | 0.2715 | 0.3258 | ns | 10/19 |
| C1 forward-path vs C0 | 6 | 4788 | 0.0500 | B>A | 0.0963 | 0.1134 | 0.1134 | ns | 10/19 |
| C1 forward-path vs C0 | 12 | 8208 | 0.1515 | B>A | 0.0008 | 0.0080 | 0.0080 | ** | 15/19 |
| C1 forward-path vs C0 | 1+3+6+12 | 16454 | 0.0940 | B>A | 0.0067 | 0.0209 | 0.0209 | ** | 13/19 |
| C1 forward vs flat-hold | 1 | 893 | 0.0120 | B>A | 0.0898 | 0.0241 | 0.0898 | * | 13/19 |
| C1 forward vs flat-hold | 3 | 2565 | 0.0309 | B>A | 0.0134 | 0.0009 | 0.0134 | ** | 16/19 |
| C1 forward vs flat-hold | 6 | 4788 | 0.0560 | B>A | 0.0055 | 0.0007 | 0.0055 | ** | 16/19 |
| C1 forward vs flat-hold | 12 | 8208 | 0.0889 | B>A | 0.0092 | 0.0076 | 0.0092 | ** | 14/19 |
| C1 forward vs flat-hold | 1+3+6+12 | 16454 | 0.0661 | B>A | 0.0072 | 0.0038 | 0.0072 | ** | 16/19 |

## Notes

- B = with-context / forward-path; A = reference (C0 or flat-hold).
- `countries B better` is a scale-free sign count (how many of the N countries have lower total |error| under B) - robust to the MASE scaling.
- Covariates are the shared euro-area set (EPU Europe, Brent, ECB rate, ESI, EUR/USD), common to all members; per-country covariates would be a further extension.
- No retraining: pooled from stored Chronos-2 panel forecasts only.

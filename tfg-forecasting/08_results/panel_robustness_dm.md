# Phase B robustness + placebo - euro-area panel (19 countries, Chronos-2)

> All-horizon pooled, cluster-robust (honest p = max of time- and country-clustering). `B>A` => the variant has lower error. Informed forward-path variants (canonical phi=0.85/w=12, undamped phi=1.0, w=24) should beat flat-hold; the random-sign PLACEBO (same magnitude, random direction) should NOT. ** p<0.05, * p<0.10.

| Contrast | n | pooled dbar | dir | p (time) | p (country) | p honest | sig | countries B better |
|---|--:|--:|---|--:|--:|--:|---|--:|
| fwd canonical vs flat-hold | 16454 | 0.0661 | B>A | 0.0072 | 0.0038 | 0.0072 | ** | 16/19 |
| fwd canonical vs C0 | 16454 | 0.0940 | B>A | 0.0067 | 0.0209 | 0.0209 | ** | 13/19 |
| fwd phi=1.0 (undamped) vs flat | 16454 | 0.0632 | B>A | 0.0807 | 0.0493 | 0.0807 | * | 14/19 |
| fwd window=24 vs flat | 16454 | 0.0407 | B>A | 0.0001 | 0.0027 | 0.0027 | ** | 14/19 |
| fwd phi=1.0 vs C0 | 16454 | 0.0911 | B>A | 0.0009 | 0.0092 | 0.0092 | ** | 15/19 |
| fwd window=24 vs C0 | 16454 | 0.0685 | B>A | 0.1310 | 0.1482 | 0.1482 | ns | 10/19 |
| PLACEBO rand-sign vs flat | 16454 | -0.0181 | A>B | 0.0804 | 0.0648 | 0.0804 | * | 6/19 |
| PLACEBO rand-sign vs C0 | 16454 | 0.0098 | B>A | 0.8123 | 0.8613 | 0.8613 | ns | 8/19 |
| fwd canonical vs PLACEBO | 16454 | 0.0842 | B>A | 0.0009 | 0.0033 | 0.0033 | ** | 16/19 |

## Reading

- **Not tuned:** if every *informed* forward-path setting beats flat-hold (B>A, significant), the result is not an artifact of phi=0.85/window=12.
- **Informed, not just non-flat:** if PLACEBO vs flat-hold is ns (or A>B) while *fwd canonical vs PLACEBO* is B>A, the benefit is the informed direction of recent momentum, not merely adding a non-flat path.
- Placebo is one fixed-seed random-sign realization (seed 20260629).

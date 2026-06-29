# Thesis-critical DM recomputation

Independent recomputation from prediction Parquets only. Existing DM JSON/CSV/MD summaries are not read.

DM method: abs-error and squared-error loss, HAC lags h-1, Harvey-Leybourne-Newbold adjustment, two-sided Student-t p-value with df=n-1.
Delta is model B versus model A, so negative means model B has lower MAE.

## File Checks

| key | file | expected model | labels | rows | max error diff | max abs diff | status |
|---|---|---|---|---:|---:|---:|---|
| spain_timesfm_c0 | timesfm_C0_predictions.parquet | timesfm_C0 | timesfm_C0 | 866 | 0.000 | 0.000 | ok |
| spain_timesfm_c1_inst | timesfm_C1_inst_predictions.parquet | timesfm_C1_inst | timesfm_C1_inst | 866 | 0.000 | 0.000 | ok |
| spain_timesfm_c1_validated | timesfm_C1_validated_predictions.parquet | timesfm_C1_validated | timesfm_C1_validated | 866 | 0.000 | 0.000 | ok |
| spain_chronos_c0 | chronos2_C0_predictions.parquet | chronos2_C0 | chronos2_C0 | 866 | 0.000 | 0.000 | ok |
| spain_chronos_c1_inst | chronos2_C1_inst_predictions.parquet | chronos2_C1_inst | chronos2_C1_inst | 866 | 0.000 | 0.000 | ok |
| spain_chronos_c1_fwd | chronos2_C1_fwd_spain_predictions.parquet | chronos2_C1_fwd_spain | chronos2_C1_fwd_spain | 866 | 0.000 | 0.000 | ok |
| spain_chronos_c1_validated | chronos2_C1_validated_predictions.parquet | chronos2_C1_validated | chronos2_C1_validated | 866 | 0.000 | 0.000 | ok |
| spain_timesfm_c1_regime | timesfm_C1_regime_predictions.parquet | timesfm_C1_regime | timesfm_C1_regime | 866 | 0.000 | 0.000 | ok |
| spain_chronos_c1_regime | chronos2_C1_regime_predictions.parquet | chronos2_C1_regime | chronos2_C1_regime | 866 | 0.000 | 0.000 | ok |
| global_chronos_c0 | chronos2_C0_global_predictions.parquet | chronos2_C0_global | chronos2_C0_global | 866 | 0.000 | 0.000 | ok |
| global_chronos_c1_inst | chronos2_C1_inst_global_predictions.parquet | chronos2_C1_inst_global | chronos2_C1_inst_global | 866 | 0.000 | 0.000 | ok |
| global_chronos_c1_validated | chronos2_C1_validated_global_predictions.parquet | chronos2_C1_validated_global | chronos2_C1_validated_global | 866 | 0.000 | 0.000 | ok |
| global_timesfm_c0 | timesfm_C0_global_predictions.parquet | timesfm_C0_global | timesfm_C0_global | 866 | 0.000 | 0.000 | ok |
| global_timesfm_c1_inst | timesfm_C1_inst_global_predictions.parquet | timesfm_C1_inst_global | timesfm_C1_inst_global | 866 | 0.000 | 0.000 | ok |
| global_timesfm_c1_validated | timesfm_C1_validated_global_predictions.parquet | timesfm_C1_validated_global | timesfm_C1_validated_global | 866 | 0.000 | 0.000 | ok |
| global_chronos_c1_fwd | chronos2_C1_fwd_global_predictions.parquet | chronos2_C1_fwd_global | chronos2_C1_fwd_global | 866 | 0.000 | 0.000 | ok |
| global_chronos_c1_regime | chronos2_C1_regime_global_predictions.parquet | chronos2_C1_regime_global | chronos2_C1_regime_global | 866 | 0.000 | 0.000 | ok |
| global_timesfm_c1_regime | timesfm_C1_regime_global_predictions.parquet | timesfm_C1_regime_global | timesfm_C1_regime_global | 866 | 0.000 | 0.000 | ok |
| global_arima | rolling_predictions_global.parquet | arima | arima, arima111, arimax, naive | 866 | 0.000 | 0.000 | ok |
| global_auto_arima | autoarima_global_predictions.parquet | auto_arima | auto_arima, naive | 866 | 0.000 | 0.000 | ok |
| europe_timesfm_c0 | timesfm_C0_europe_predictions.parquet | timesfm_C0_europe | timesfm_C0_europe | 866 | 0.000 | 0.000 | ok |
| europe_timesfm_c1_full | timesfm_C1_full_europe_predictions.parquet | timesfm_C1_full_europe | timesfm_C1_full_europe | 866 | 0.000 | 0.000 | ok |
| europe_timesfm_c1_validated | timesfm_C1_validated_europe_predictions.parquet | timesfm_C1_validated_europe | timesfm_C1_validated_europe | 866 | 0.000 | 0.000 | ok |
| europe_chronos_c0 | chronos2_C0_europe_predictions.parquet | chronos2_C0_europe | chronos2_C0_europe | 866 | 0.000 | 0.000 | ok |
| europe_chronos_c1_inst | chronos2_C1_inst_europe_predictions.parquet | chronos2_C1_inst_europe | chronos2_C1_inst_europe | 866 | 0.000 | 0.000 | ok |
| europe_chronos_c1_fwd | chronos2_C1_fwd_europe_predictions.parquet | chronos2_C1_fwd_europe | chronos2_C1_fwd_europe | 866 | 0.000 | 0.000 | ok |
| europe_timesfm_c1_regime | timesfm_C1_regime_europe_predictions.parquet | timesfm_C1_regime_europe | timesfm_C1_regime_europe | 866 | 0.000 | 0.000 | ok |
| europe_chronos_c1_regime | chronos2_C1_regime_europe_predictions.parquet | chronos2_C1_regime_europe | chronos2_C1_regime_europe | 866 | 0.000 | 0.000 | ok |
| europe_sarima | rolling_predictions_europe.parquet | sarima | naive, sarima, sarimax | 866 | 0.000 | 0.000 | ok |
| europe_auto_arima | autoarima_europe_predictions.parquet | auto_arima | auto_arima, naive | 866 | 0.000 | 0.000 | ok |

## Path Level

| series | comparison | h | n | origins | MAE A | MAE B | delta % [95% boot CI] | better | p abs | sig | p sq | sig sq |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|
| Spain | TimesFM context | 1 | 47 | 47 | 0.4364 | 0.4454 | 2.0% [-1.6, 5.7] | timesfm_C0 | 0.2784 | ns | 0.1917 | ns |
| Spain | TimesFM context | 3 | 135 | 45 | 0.7320 | 0.7460 | 1.9% [0.3, 3.7] | timesfm_C0 | 0.0095 | ** | 0.0522 | * |
| Spain | TimesFM context | 6 | 252 | 42 | 1.0866 | 1.1000 | 1.2% [0.1, 2.4] | timesfm_C0 | 0.0180 | ** | 0.0265 | ** |
| Spain | TimesFM context | 12 | 432 | 36 | 1.8635 | 1.8781 | 0.8% [0.0, 1.6] | timesfm_C0 | 0.0364 | ** | 0.0388 | ** |
| Spain | TimesFM validated context | 1 | 47 | 47 | 0.4364 | 0.4364 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM validated context | 3 | 135 | 45 | 0.7320 | 0.7320 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM validated context | 6 | 252 | 42 | 1.0866 | 1.0866 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM validated context | 12 | 432 | 36 | 1.8635 | 1.8635 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 context | 1 | 47 | 47 | 0.5202 | 0.5488 | 5.5% [-11.0, 25.3] | chronos2_C0 | 0.5300 | ns | 0.7810 | ns |
| Spain | Chronos-2 context | 3 | 135 | 45 | 0.8179 | 0.8212 | 0.4% [-10.5, 11.6] | chronos2_C0 | 0.9532 | ns | 0.7676 | ns |
| Spain | Chronos-2 context | 6 | 252 | 42 | 1.2060 | 1.2370 | 2.6% [-5.6, 10.1] | chronos2_C0 | 0.6465 | ns | 0.2792 | ns |
| Spain | Chronos-2 context | 12 | 432 | 36 | 1.9898 | 2.0694 | 4.0% [-4.3, 12.2] | chronos2_C0 | 0.4876 | ns | 0.1873 | ns |
| Spain | Chronos-2 validated context | 1 | 47 | 47 | 0.5202 | 0.5202 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 validated context | 3 | 135 | 45 | 0.8179 | 0.8179 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 validated context | 6 | 252 | 42 | 1.2060 | 1.2060 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 validated context | 12 | 432 | 36 | 1.9898 | 1.9898 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM regime context | 1 | 47 | 47 | 0.4364 | 0.4364 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | TimesFM regime context | 3 | 135 | 45 | 0.7320 | 0.7320 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | TimesFM regime context | 6 | 252 | 42 | 1.0866 | 1.0866 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | TimesFM regime context | 12 | 432 | 36 | 1.8635 | 1.8635 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 1 | 47 | 47 | 0.5202 | 0.5202 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 3 | 135 | 45 | 0.8179 | 0.8179 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 6 | 252 | 42 | 1.2060 | 1.2060 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 12 | 432 | 36 | 1.9898 | 1.9898 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 forward-path context | 1 | 47 | 47 | 0.5202 | 0.5418 | 4.2% [-11.4, 22.4] | chronos2_C0 | 0.6125 | ns | 0.7296 | ns |
| Spain | Chronos-2 forward-path context | 3 | 135 | 45 | 0.8179 | 0.8090 | -1.1% [-11.8, 9.3] | chronos2_C1_fwd_spain | 0.8740 | ns | 0.6852 | ns |
| Spain | Chronos-2 forward-path context | 6 | 252 | 42 | 1.2060 | 1.2181 | 1.0% [-9.2, 10.1] | chronos2_C0 | 0.8772 | ns | 0.3204 | ns |
| Spain | Chronos-2 forward-path context | 12 | 432 | 36 | 1.9898 | 2.0892 | 5.0% [-4.9, 14.6] | chronos2_C0 | 0.4769 | ns | 0.1853 | ns |
| Spain | Chronos-2 forward vs flat-hold | 1 | 47 | 47 | 0.5488 | 0.5418 | -1.3% [-7.3, 4.4] | chronos2_C1_fwd_spain | 0.6587 | ns | 0.9812 | ns |
| Spain | Chronos-2 forward vs flat-hold | 3 | 135 | 45 | 0.8212 | 0.8090 | -1.5% [-5.8, 1.8] | chronos2_C1_fwd_spain | 0.4686 | ns | 0.7614 | ns |
| Spain | Chronos-2 forward vs flat-hold | 6 | 252 | 42 | 1.2370 | 1.2181 | -1.5% [-6.1, 2.0] | chronos2_C1_fwd_spain | 0.4943 | ns | 0.8563 | ns |
| Spain | Chronos-2 forward vs flat-hold | 12 | 432 | 36 | 2.0694 | 2.0892 | 1.0% [-2.6, 4.1] | chronos2_C1_inst | 0.6720 | ns | 0.3786 | ns |
| Global | Chronos-2 context | 1 | 47 | 47 | 0.2519 | 0.2004 | -20.4% [-40.6, 7.0] | chronos2_C1_inst_global | 0.1283 | ns | 0.1316 | ns |
| Global | Chronos-2 context | 3 | 135 | 45 | 0.3580 | 0.3417 | -4.5% [-23.5, 21.0] | chronos2_C1_inst_global | 0.7448 | ns | 0.8156 | ns |
| Global | Chronos-2 context | 6 | 252 | 42 | 0.6423 | 0.5914 | -7.9% [-30.2, 22.0] | chronos2_C1_inst_global | 0.6608 | ns | 0.5835 | ns |
| Global | Chronos-2 context | 12 | 432 | 36 | 1.3375 | 1.1433 | -14.5% [-36.9, 15.9] | chronos2_C1_inst_global | 0.4756 | ns | 0.5088 | ns |
| Global | Chronos-2 validated context | 1 | 47 | 47 | 0.2519 | 0.2723 | 8.1% [1.4, 15.9] | chronos2_C0_global | 0.0301 | ** | 0.0793 | * |
| Global | Chronos-2 validated context | 3 | 135 | 45 | 0.3580 | 0.3558 | -0.6% [-3.5, 2.9] | chronos2_C1_validated_global | 0.7475 | ns | 0.3417 | ns |
| Global | Chronos-2 validated context | 6 | 252 | 42 | 0.6423 | 0.6228 | -3.0% [-4.2, -1.5] | chronos2_C1_validated_global | 0.0058 | ** | 0.0093 | ** |
| Global | Chronos-2 validated context | 12 | 432 | 36 | 1.3375 | 1.3011 | -2.7% [-3.5, -1.9] | chronos2_C1_validated_global | 0.0003 | ** | 0.0016 | ** |
| Global | TimesFM context | 1 | 47 | 47 | 0.2103 | 0.2137 | 1.6% [-16.3, 24.1] | timesfm_C0_global | 0.8690 | ns | 0.8668 | ns |
| Global | TimesFM context | 3 | 135 | 45 | 0.3760 | 0.3487 | -7.3% [-16.8, 4.4] | timesfm_C1_inst_global | 0.2918 | ns | 0.1043 | ns |
| Global | TimesFM context | 6 | 252 | 42 | 0.6475 | 0.6072 | -6.2% [-12.1, 0.6] | timesfm_C1_inst_global | 0.1820 | ns | 0.0344 | ** |
| Global | TimesFM context | 12 | 432 | 36 | 1.2333 | 1.1913 | -3.4% [-7.0, 1.1] | timesfm_C1_inst_global | 0.2801 | ns | 0.0187 | ** |
| Global | TimesFM validated context | 1 | 47 | 47 | 0.2103 | 0.1952 | -7.2% [-15.0, 0.6] | timesfm_C1_validated_global | 0.0860 | * | 0.2331 | ns |
| Global | TimesFM validated context | 3 | 135 | 45 | 0.3760 | 0.3605 | -4.1% [-8.8, -0.1] | timesfm_C1_validated_global | 0.0985 | * | 0.1014 | ns |
| Global | TimesFM validated context | 6 | 252 | 42 | 0.6475 | 0.6303 | -2.7% [-5.2, -0.1] | timesfm_C1_validated_global | 0.0726 | * | 0.0947 | * |
| Global | TimesFM validated context | 12 | 432 | 36 | 1.2333 | 1.2225 | -0.9% [-2.3, 0.5] | timesfm_C1_validated_global | 0.3538 | ns | 0.1469 | ns |
| Global | Chronos-2 forward-path context | 1 | 47 | 47 | 0.2519 | 0.1925 | -23.6% [-43.3, 2.0] | chronos2_C1_fwd_global | 0.0750 | * | 0.0382 | ** |
| Global | Chronos-2 forward-path context | 3 | 135 | 45 | 0.3580 | 0.3074 | -14.1% [-31.5, 8.7] | chronos2_C1_fwd_global | 0.2838 | ns | 0.2145 | ns |
| Global | Chronos-2 forward-path context | 6 | 252 | 42 | 0.6423 | 0.5299 | -17.5% [-34.7, 4.2] | chronos2_C1_fwd_global | 0.2161 | ns | 0.0842 | * |
| Global | Chronos-2 forward-path context | 12 | 432 | 36 | 1.3375 | 1.0648 | -20.4% [-35.8, 1.1] | chronos2_C1_fwd_global | 0.1658 | ns | 0.0858 | * |
| Global | Chronos-2 forward vs flat-hold | 1 | 47 | 47 | 0.2004 | 0.1925 | -4.0% [-13.9, 7.9] | chronos2_C1_fwd_global | 0.4825 | ns | 0.2069 | ns |
| Global | Chronos-2 forward vs flat-hold | 3 | 135 | 45 | 0.3417 | 0.3074 | -10.1% [-20.7, 3.2] | chronos2_C1_fwd_global | 0.2173 | ns | 0.0691 | * |
| Global | Chronos-2 forward vs flat-hold | 6 | 252 | 42 | 0.5914 | 0.5299 | -10.4% [-21.0, 4.8] | chronos2_C1_fwd_global | 0.2842 | ns | 0.0487 | ** |
| Global | Chronos-2 forward vs flat-hold | 12 | 432 | 36 | 1.1433 | 1.0648 | -6.9% [-19.1, 9.2] | chronos2_C1_fwd_global | 0.5156 | ns | 0.0797 | * |
| Global | Chronos-2 regime context | 1 | 47 | 47 | 0.2519 | 0.2487 | -1.3% [-3.1, 0.7] | chronos2_C1_regime_global | 0.2007 | ns | 0.0502 | * |
| Global | Chronos-2 regime context | 3 | 135 | 45 | 0.3580 | 0.3524 | -1.6% [-2.7, -0.6] | chronos2_C1_regime_global | 0.0191 | ** | 0.0381 | ** |
| Global | Chronos-2 regime context | 6 | 252 | 42 | 0.6423 | 0.6361 | -1.0% [-1.8, -0.3] | chronos2_C1_regime_global | 0.0390 | ** | 0.0391 | ** |
| Global | Chronos-2 regime context | 12 | 432 | 36 | 1.3375 | 1.3301 | -0.5% [-1.1, -0.2] | chronos2_C1_regime_global | 0.0544 | * | 0.0540 | * |
| Global | TimesFM regime context | 1 | 47 | 47 | 0.2103 | 0.2137 | 1.6% [-0.5, 4.3] | timesfm_C0_global | 0.1761 | ns | 0.4644 | ns |
| Global | TimesFM regime context | 3 | 135 | 45 | 0.3760 | 0.3792 | 0.9% [-0.2, 2.1] | timesfm_C0_global | 0.2692 | ns | 0.3764 | ns |
| Global | TimesFM regime context | 6 | 252 | 42 | 0.6475 | 0.6506 | 0.5% [-0.2, 1.3] | timesfm_C0_global | 0.3509 | ns | 0.3475 | ns |
| Global | TimesFM regime context | 12 | 432 | 36 | 1.2333 | 1.2358 | 0.2% [-0.2, 0.7] | timesfm_C0_global | 0.5060 | ns | 0.3838 | ns |
| Global | Chronos-2 vs ARIMA | 1 | 47 | 47 | 0.2004 | 0.1907 | -4.9% [-21.6, 18.9] | arima | 0.6461 | ns | 0.4348 | ns |
| Global | Chronos-2 vs ARIMA | 3 | 135 | 45 | 0.3417 | 0.3565 | 4.3% [-12.4, 25.0] | chronos2_C1_inst_global | 0.6858 | ns | 0.6700 | ns |
| Global | Chronos-2 vs ARIMA | 6 | 252 | 42 | 0.5914 | 0.6821 | 15.3% [-3.8, 39.2] | chronos2_C1_inst_global | 0.2470 | ns | 0.2902 | ns |
| Global | Chronos-2 vs ARIMA | 12 | 432 | 36 | 1.1433 | 1.5444 | 35.1% [8.7, 74.5] | chronos2_C1_inst_global | 0.0606 | * | 0.0961 | * |
| Global | Chronos-2 vs AutoARIMA | 1 | 47 | 47 | 0.2004 | 0.1787 | -10.9% [-26.2, 7.0] | auto_arima | 0.2202 | ns | 0.2527 | ns |
| Global | Chronos-2 vs AutoARIMA | 3 | 135 | 45 | 0.3417 | 0.3305 | -3.3% [-11.5, 6.1] | auto_arima | 0.5305 | ns | 0.9130 | ns |
| Global | Chronos-2 vs AutoARIMA | 6 | 252 | 42 | 0.5914 | 0.6059 | 2.5% [-6.7, 12.1] | chronos2_C1_inst_global | 0.6583 | ns | 0.2197 | ns |
| Global | Chronos-2 vs AutoARIMA | 12 | 432 | 36 | 1.1433 | 1.3294 | 16.3% [4.4, 29.4] | chronos2_C1_inst_global | 0.0416 | ** | 0.0140 | ** |
| Global | Chronos-2 validated vs ARIMA | 1 | 47 | 47 | 0.2723 | 0.1907 | -30.0% [-48.6, -2.5] | arima | 0.0407 | ** | 0.0247 | ** |
| Global | Chronos-2 validated vs ARIMA | 3 | 135 | 45 | 0.3558 | 0.3565 | 0.2% [-20.9, 30.0] | chronos2_C1_validated_global | 0.9883 | ns | 0.8245 | ns |
| Global | Chronos-2 validated vs ARIMA | 6 | 252 | 42 | 0.6228 | 0.6821 | 9.5% [-10.8, 35.1] | chronos2_C1_validated_global | 0.4801 | ns | 0.5945 | ns |
| Global | Chronos-2 validated vs ARIMA | 12 | 432 | 36 | 1.3011 | 1.5444 | 18.7% [-2.2, 46.6] | chronos2_C1_validated_global | 0.2006 | ns | 0.2809 | ns |
| Global | Chronos-2 validated vs AutoARIMA | 1 | 47 | 47 | 0.2723 | 0.1787 | -34.4% [-53.1, -6.7] | auto_arima | 0.0205 | ** | 0.0284 | ** |
| Global | Chronos-2 validated vs AutoARIMA | 3 | 135 | 45 | 0.3558 | 0.3305 | -7.1% [-28.1, 23.1] | auto_arima | 0.6436 | ns | 0.9293 | ns |
| Global | Chronos-2 validated vs AutoARIMA | 6 | 252 | 42 | 0.6228 | 0.6059 | -2.7% [-27.1, 29.2] | auto_arima | 0.8848 | ns | 0.9164 | ns |
| Global | Chronos-2 validated vs AutoARIMA | 12 | 432 | 36 | 1.3011 | 1.3294 | 2.2% [-22.5, 34.1] | chronos2_C1_validated_global | 0.9128 | ns | 0.6389 | ns |
| Global | TimesFM vs ARIMA | 1 | 47 | 47 | 0.2137 | 0.1907 | -10.8% [-26.7, 9.4] | arima | 0.2846 | ns | 0.3381 | ns |
| Global | TimesFM vs ARIMA | 3 | 135 | 45 | 0.3487 | 0.3565 | 2.2% [-12.8, 18.7] | timesfm_C1_inst_global | 0.8208 | ns | 0.4512 | ns |
| Global | TimesFM vs ARIMA | 6 | 252 | 42 | 0.6072 | 0.6821 | 12.3% [-4.6, 31.5] | timesfm_C1_inst_global | 0.2780 | ns | 0.2297 | ns |
| Global | TimesFM vs ARIMA | 12 | 432 | 36 | 1.1913 | 1.5444 | 29.6% [5.4, 63.0] | timesfm_C1_inst_global | 0.0780 | * | 0.1538 | ns |
| Global | TimesFM vs AutoARIMA | 1 | 47 | 47 | 0.2137 | 0.1787 | -16.4% [-33.6, 6.4] | auto_arima | 0.1374 | ns | 0.3728 | ns |
| Global | TimesFM vs AutoARIMA | 3 | 135 | 45 | 0.3487 | 0.3305 | -5.2% [-20.3, 12.0] | auto_arima | 0.6051 | ns | 0.8831 | ns |
| Global | TimesFM vs AutoARIMA | 6 | 252 | 42 | 0.6072 | 0.6059 | -0.2% [-13.9, 14.4] | auto_arima | 0.9812 | ns | 0.5040 | ns |
| Global | TimesFM vs AutoARIMA | 12 | 432 | 36 | 1.1913 | 1.3294 | 11.6% [-1.6, 26.5] | timesfm_C1_inst_global | 0.1892 | ns | 0.0953 | * |
| Global | TimesFM validated vs ARIMA | 1 | 47 | 47 | 0.1952 | 0.1907 | -2.3% [-17.0, 16.9] | arima | 0.7943 | ns | 0.2561 | ns |
| Global | TimesFM validated vs ARIMA | 3 | 135 | 45 | 0.3605 | 0.3565 | -1.1% [-16.0, 16.2] | arima | 0.9144 | ns | 0.9197 | ns |
| Global | TimesFM validated vs ARIMA | 6 | 252 | 42 | 0.6303 | 0.6821 | 8.2% [-9.9, 30.0] | timesfm_C1_validated_global | 0.5126 | ns | 0.6193 | ns |
| Global | TimesFM validated vs ARIMA | 12 | 432 | 36 | 1.2225 | 1.5444 | 26.3% [1.3, 63.2] | timesfm_C1_validated_global | 0.1432 | ns | 0.2936 | ns |
| Global | TimesFM validated vs AutoARIMA | 1 | 47 | 47 | 0.1952 | 0.1787 | -8.5% [-21.7, 7.6] | auto_arima | 0.2912 | ns | 0.2866 | ns |
| Global | TimesFM validated vs AutoARIMA | 3 | 135 | 45 | 0.3605 | 0.3305 | -8.3% [-19.9, 4.3] | auto_arima | 0.2818 | ns | 0.5085 | ns |
| Global | TimesFM validated vs AutoARIMA | 6 | 252 | 42 | 0.6303 | 0.6059 | -3.9% [-15.0, 8.3] | auto_arima | 0.6066 | ns | 0.9961 | ns |
| Global | TimesFM validated vs AutoARIMA | 12 | 432 | 36 | 1.2225 | 1.3294 | 8.7% [-3.6, 23.6] | timesfm_C1_validated_global | 0.2862 | ns | 0.1995 | ns |
| Europe | TimesFM context | 1 | 47 | 47 | 0.3526 | 0.4358 | 23.6% [-3.8, 56.3] | timesfm_C0_europe | 0.1005 | ns | 0.1142 | ns |
| Europe | TimesFM context | 3 | 135 | 45 | 0.6970 | 0.6913 | -0.8% [-9.9, 9.5] | timesfm_C1_full_europe | 0.8912 | ns | 0.8485 | ns |
| Europe | TimesFM context | 6 | 252 | 42 | 1.0350 | 0.9950 | -3.9% [-11.0, 4.3] | timesfm_C1_full_europe | 0.4535 | ns | 0.3455 | ns |
| Europe | TimesFM context | 12 | 432 | 36 | 2.0144 | 1.9946 | -1.0% [-5.5, 4.2] | timesfm_C1_full_europe | 0.7610 | ns | 0.5249 | ns |
| Europe | TimesFM validated context | 1 | 47 | 47 | 0.3526 | 0.3516 | -0.3% [-1.6, 1.0] | timesfm_C1_validated_europe | 0.6711 | ns | 0.6733 | ns |
| Europe | TimesFM validated context | 3 | 135 | 45 | 0.6970 | 0.7012 | 0.6% [0.1, 1.1] | timesfm_C0_europe | 0.0607 | * | 0.0038 | ** |
| Europe | TimesFM validated context | 6 | 252 | 42 | 1.0350 | 1.0399 | 0.5% [0.1, 0.8] | timesfm_C0_europe | 0.0416 | ** | 0.0072 | ** |
| Europe | TimesFM validated context | 12 | 432 | 36 | 2.0144 | 2.0161 | 0.1% [-0.1, 0.3] | timesfm_C0_europe | 0.4940 | ns | 0.2254 | ns |
| Europe | TimesFM regime context | 1 | 47 | 47 | 0.3526 | 0.3517 | -0.3% [-1.2, 0.7] | timesfm_C1_regime_europe | 0.5922 | ns | 0.4951 | ns |
| Europe | TimesFM regime context | 3 | 135 | 45 | 0.6970 | 0.6997 | 0.4% [0.1, 0.7] | timesfm_C0_europe | 0.0660 | * | 0.0353 | ** |
| Europe | TimesFM regime context | 6 | 252 | 42 | 1.0350 | 1.0384 | 0.3% [0.1, 0.6] | timesfm_C0_europe | 0.0301 | ** | 0.0161 | ** |
| Europe | TimesFM regime context | 12 | 432 | 36 | 2.0144 | 2.0153 | 0.0% [-0.1, 0.2] | timesfm_C0_europe | 0.4817 | ns | 0.5168 | ns |
| Europe | Chronos-2 regime context | 1 | 47 | 47 | 0.5124 | 0.5113 | -0.2% [-0.8, 0.5] | chronos2_C1_regime_europe | 0.4896 | ns | 0.0871 | * |
| Europe | Chronos-2 regime context | 3 | 135 | 45 | 0.8200 | 0.8193 | -0.1% [-0.5, 0.3] | chronos2_C1_regime_europe | 0.7688 | ns | 0.6421 | ns |
| Europe | Chronos-2 regime context | 6 | 252 | 42 | 1.2521 | 1.2520 | -0.0% [-0.3, 0.3] | chronos2_C1_regime_europe | 0.9383 | ns | 0.8508 | ns |
| Europe | Chronos-2 regime context | 12 | 432 | 36 | 2.3003 | 2.3006 | 0.0% [-0.2, 0.2] | chronos2_C0_europe | 0.9149 | ns | 0.6966 | ns |
| Europe | Chronos-2 forward-path context | 1 | 47 | 47 | 0.5124 | 0.5639 | 10.0% [-10.9, 41.4] | chronos2_C0_europe | 0.4229 | ns | 0.7715 | ns |
| Europe | Chronos-2 forward-path context | 3 | 135 | 45 | 0.8200 | 0.8180 | -0.2% [-15.5, 16.9] | chronos2_C1_fwd_europe | 0.9813 | ns | 0.8409 | ns |
| Europe | Chronos-2 forward-path context | 6 | 252 | 42 | 1.2521 | 1.2478 | -0.3% [-13.8, 13.8] | chronos2_C1_fwd_europe | 0.9701 | ns | 0.8666 | ns |
| Europe | Chronos-2 forward-path context | 12 | 432 | 36 | 2.3003 | 2.2648 | -1.5% [-12.8, 11.7] | chronos2_C1_fwd_europe | 0.8551 | ns | 0.7334 | ns |
| Europe | Chronos-2 forward vs flat-hold | 1 | 47 | 47 | 0.5513 | 0.5639 | 2.3% [-5.4, 10.1] | chronos2_C1_inst_europe | 0.5527 | ns | 0.6959 | ns |
| Europe | Chronos-2 forward vs flat-hold | 3 | 135 | 45 | 0.8491 | 0.8180 | -3.7% [-10.1, 3.3] | chronos2_C1_fwd_europe | 0.3979 | ns | 0.2413 | ns |
| Europe | Chronos-2 forward vs flat-hold | 6 | 252 | 42 | 1.3213 | 1.2478 | -5.6% [-11.8, 1.7] | chronos2_C1_fwd_europe | 0.2410 | ns | 0.0542 | * |
| Europe | Chronos-2 forward vs flat-hold | 12 | 432 | 36 | 2.4020 | 2.2648 | -5.7% [-12.4, 2.4] | chronos2_C1_fwd_europe | 0.2647 | ns | 0.0563 | * |
| Europe | TimesFM vs SARIMA | 1 | 47 | 47 | 0.4358 | 0.4126 | -5.3% [-28.4, 29.2] | sarima | 0.7139 | ns | 0.3713 | ns |
| Europe | TimesFM vs SARIMA | 3 | 135 | 45 | 0.6913 | 0.7171 | 3.7% [-14.5, 24.7] | timesfm_C1_full_europe | 0.7457 | ns | 0.5934 | ns |
| Europe | TimesFM vs SARIMA | 6 | 252 | 42 | 0.9950 | 1.2259 | 23.2% [5.5, 45.3] | timesfm_C1_full_europe | 0.0435 | ** | 0.0166 | ** |
| Europe | TimesFM vs SARIMA | 12 | 432 | 36 | 1.9946 | 2.4109 | 20.9% [8.7, 37.6] | timesfm_C1_full_europe | 0.0125 | ** | 0.0272 | ** |
| Europe | TimesFM vs AutoARIMA | 1 | 47 | 47 | 0.4358 | 0.3758 | -13.7% [-32.7, 11.6] | auto_arima | 0.2600 | ns | 0.1234 | ns |
| Europe | TimesFM vs AutoARIMA | 3 | 135 | 45 | 0.6913 | 0.6575 | -4.9% [-18.8, 9.7] | auto_arima | 0.5656 | ns | 0.9725 | ns |
| Europe | TimesFM vs AutoARIMA | 6 | 252 | 42 | 0.9950 | 1.1473 | 15.3% [1.4, 31.1] | timesfm_C1_full_europe | 0.0824 | * | 0.0273 | ** |
| Europe | TimesFM vs AutoARIMA | 12 | 432 | 36 | 1.9946 | 2.5100 | 25.8% [13.5, 41.9] | timesfm_C1_full_europe | 0.0007 | ** | 0.0054 | ** |
| Europe | TimesFM validated vs SARIMA | 1 | 47 | 47 | 0.3516 | 0.4126 | 17.3% [-4.4, 45.4] | timesfm_C1_validated_europe | 0.1227 | ns | 0.2353 | ns |
| Europe | TimesFM validated vs SARIMA | 3 | 135 | 45 | 0.7012 | 0.7171 | 2.3% [-11.9, 18.4] | timesfm_C1_validated_europe | 0.7998 | ns | 0.6558 | ns |
| Europe | TimesFM validated vs SARIMA | 6 | 252 | 42 | 1.0399 | 1.2259 | 17.9% [5.1, 33.2] | timesfm_C1_validated_europe | 0.0270 | ** | 0.0082 | ** |
| Europe | TimesFM validated vs SARIMA | 12 | 432 | 36 | 2.0161 | 2.4109 | 19.6% [9.7, 33.4] | timesfm_C1_validated_europe | 0.0022 | ** | 0.0082 | ** |
| Europe | TimesFM validated vs AutoARIMA | 1 | 47 | 47 | 0.3516 | 0.3758 | 6.9% [-10.7, 28.3] | timesfm_C1_validated_europe | 0.4729 | ns | 0.6459 | ns |
| Europe | TimesFM validated vs AutoARIMA | 3 | 135 | 45 | 0.7012 | 0.6575 | -6.2% [-21.8, 10.0] | auto_arima | 0.5278 | ns | 0.8256 | ns |
| Europe | TimesFM validated vs AutoARIMA | 6 | 252 | 42 | 1.0399 | 1.1473 | 10.3% [-5.8, 29.4] | timesfm_C1_validated_europe | 0.3423 | ns | 0.2543 | ns |
| Europe | TimesFM validated vs AutoARIMA | 12 | 432 | 36 | 2.0161 | 2.5100 | 24.5% [9.6, 44.7] | timesfm_C1_validated_europe | 0.0126 | ** | 0.0338 | ** |

## Endpoint Level

| series | comparison | h | n | origins | MAE A | MAE B | delta % [95% boot CI] | better | p abs | sig | p sq | sig sq |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|
| Spain | TimesFM context | 1 | 47 | 47 | 0.4364 | 0.4454 | 2.0% [-1.6, 5.7] | timesfm_C0 | 0.2784 | ns | 0.1917 | ns |
| Spain | TimesFM context | 3 | 45 | 45 | 1.0179 | 1.0376 | 1.9% [0.5, 3.6] | timesfm_C0 | 0.0056 | ** | 0.0046 | ** |
| Spain | TimesFM context | 6 | 42 | 42 | 1.5794 | 1.5909 | 0.7% [-0.3, 1.9] | timesfm_C0 | 0.1936 | ns | 0.0758 | * |
| Spain | TimesFM context | 12 | 36 | 36 | 3.0884 | 3.1024 | 0.5% [-0.1, 1.1] | timesfm_C0 | 0.1043 | ns | 0.1090 | ns |
| Spain | TimesFM validated context | 1 | 47 | 47 | 0.4364 | 0.4364 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM validated context | 3 | 45 | 45 | 1.0179 | 1.0179 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM validated context | 6 | 42 | 42 | 1.5794 | 1.5794 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM validated context | 12 | 36 | 36 | 3.0884 | 3.0884 | 0.0% [0.0, 0.0] | timesfm_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 context | 1 | 47 | 47 | 0.5202 | 0.5488 | 5.5% [-11.0, 25.3] | chronos2_C0 | 0.5300 | ns | 0.7810 | ns |
| Spain | Chronos-2 context | 3 | 45 | 45 | 1.0943 | 1.0696 | -2.3% [-13.4, 7.7] | chronos2_C1_inst | 0.7684 | ns | 0.6254 | ns |
| Spain | Chronos-2 context | 6 | 42 | 42 | 1.7756 | 1.8643 | 5.0% [-4.1, 14.2] | chronos2_C0 | 0.4091 | ns | 0.2322 | ns |
| Spain | Chronos-2 context | 12 | 36 | 36 | 3.0825 | 3.2432 | 5.2% [-3.4, 15.0] | chronos2_C0 | 0.4080 | ns | 0.3268 | ns |
| Spain | Chronos-2 validated context | 1 | 47 | 47 | 0.5202 | 0.5202 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 validated context | 3 | 45 | 45 | 1.0943 | 1.0943 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 validated context | 6 | 42 | 42 | 1.7756 | 1.7756 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | Chronos-2 validated context | 12 | 36 | 36 | 3.0825 | 3.0825 | 0.0% [0.0, 0.0] | chronos2_C1_validated | NA | NA | NA | NA |
| Spain | TimesFM regime context | 1 | 47 | 47 | 0.4364 | 0.4364 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | TimesFM regime context | 3 | 45 | 45 | 1.0179 | 1.0179 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | TimesFM regime context | 6 | 42 | 42 | 1.5794 | 1.5794 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | TimesFM regime context | 12 | 36 | 36 | 3.0884 | 3.0884 | 0.0% [0.0, 0.0] | timesfm_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 1 | 47 | 47 | 0.5202 | 0.5202 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 3 | 45 | 45 | 1.0943 | 1.0943 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 6 | 42 | 42 | 1.7756 | 1.7756 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 regime context | 12 | 36 | 36 | 3.0825 | 3.0825 | 0.0% [0.0, 0.0] | chronos2_C1_regime | NA | NA | NA | NA |
| Spain | Chronos-2 forward-path context | 1 | 47 | 47 | 0.5202 | 0.5418 | 4.2% [-11.4, 22.4] | chronos2_C0 | 0.6125 | ns | 0.7296 | ns |
| Spain | Chronos-2 forward-path context | 3 | 45 | 45 | 1.0943 | 1.0544 | -3.7% [-15.8, 7.1] | chronos2_C1_fwd_spain | 0.6603 | ns | 0.6183 | ns |
| Spain | Chronos-2 forward-path context | 6 | 42 | 42 | 1.7756 | 1.8293 | 3.0% [-8.5, 14.7] | chronos2_C0 | 0.5873 | ns | 0.2900 | ns |
| Spain | Chronos-2 forward-path context | 12 | 36 | 36 | 3.0825 | 3.3002 | 7.1% [-3.6, 19.2] | chronos2_C0 | 0.3736 | ns | 0.3077 | ns |
| Spain | Chronos-2 forward vs flat-hold | 1 | 47 | 47 | 0.5488 | 0.5418 | -1.3% [-7.3, 4.4] | chronos2_C1_fwd_spain | 0.6587 | ns | 0.9812 | ns |
| Spain | Chronos-2 forward vs flat-hold | 3 | 45 | 45 | 1.0696 | 1.0544 | -1.4% [-5.6, 1.9] | chronos2_C1_fwd_spain | 0.6068 | ns | 0.8514 | ns |
| Spain | Chronos-2 forward vs flat-hold | 6 | 42 | 42 | 1.8643 | 1.8293 | -1.9% [-7.1, 2.0] | chronos2_C1_fwd_spain | 0.5154 | ns | 0.9424 | ns |
| Spain | Chronos-2 forward vs flat-hold | 12 | 36 | 36 | 3.2432 | 3.3002 | 1.8% [-2.4, 6.3] | chronos2_C1_inst | 0.0113 | ** | 0.4061 | ns |
| Global | Chronos-2 context | 1 | 47 | 47 | 0.2519 | 0.2004 | -20.4% [-40.6, 7.0] | chronos2_C1_inst_global | 0.1283 | ns | 0.1316 | ns |
| Global | Chronos-2 context | 3 | 45 | 45 | 0.4898 | 0.4963 | 1.3% [-22.2, 35.4] | chronos2_C0_global | 0.9595 | ns | 0.9669 | ns |
| Global | Chronos-2 context | 6 | 42 | 42 | 1.1443 | 0.9898 | -13.5% [-38.9, 22.2] | chronos2_C1_inst_global | 0.7711 | ns | 0.7425 | ns |
| Global | Chronos-2 context | 12 | 36 | 36 | 2.3131 | 1.9698 | -14.8% [-38.3, 15.7] | chronos2_C1_inst_global | 0.6762 | ns | 0.7553 | ns |
| Global | Chronos-2 validated context | 1 | 47 | 47 | 0.2519 | 0.2723 | 8.1% [1.4, 15.9] | chronos2_C0_global | 0.0301 | ** | 0.0793 | * |
| Global | Chronos-2 validated context | 3 | 45 | 45 | 0.4898 | 0.4610 | -5.9% [-9.3, -2.7] | chronos2_C1_validated_global | 0.0729 | * | 0.1667 | ns |
| Global | Chronos-2 validated context | 6 | 42 | 42 | 1.1443 | 1.1066 | -3.3% [-4.3, -2.1] | chronos2_C1_validated_global | 0.1960 | ns | 0.2014 | ns |
| Global | Chronos-2 validated context | 12 | 36 | 36 | 2.3131 | 2.2620 | -2.2% [-2.9, -1.5] | chronos2_C1_validated_global | 0.1189 | ns | 0.1514 | ns |
| Global | TimesFM context | 1 | 47 | 47 | 0.2103 | 0.2137 | 1.6% [-16.3, 24.1] | timesfm_C0_global | 0.8690 | ns | 0.8668 | ns |
| Global | TimesFM context | 3 | 45 | 45 | 0.5448 | 0.5051 | -7.3% [-15.0, 2.1] | timesfm_C1_inst_global | 0.4450 | ns | 0.2357 | ns |
| Global | TimesFM context | 6 | 42 | 42 | 1.0760 | 1.0178 | -5.4% [-9.5, -0.9] | timesfm_C1_inst_global | 0.4615 | ns | 0.3322 | ns |
| Global | TimesFM context | 12 | 36 | 36 | 2.0113 | 1.9708 | -2.0% [-4.9, 1.4] | timesfm_C1_inst_global | 0.6481 | ns | 0.4208 | ns |
| Global | TimesFM validated context | 1 | 47 | 47 | 0.2103 | 0.1952 | -7.2% [-15.0, 0.6] | timesfm_C1_validated_global | 0.0860 | * | 0.2331 | ns |
| Global | TimesFM validated context | 3 | 45 | 45 | 0.5448 | 0.5303 | -2.7% [-6.2, 0.7] | timesfm_C1_validated_global | 0.3156 | ns | 0.2772 | ns |
| Global | TimesFM validated context | 6 | 42 | 42 | 1.0760 | 1.0547 | -2.0% [-3.9, -0.3] | timesfm_C1_validated_global | 0.3023 | ns | 0.4552 | ns |
| Global | TimesFM validated context | 12 | 36 | 36 | 2.0113 | 2.0188 | 0.4% [-0.6, 1.8] | timesfm_C0_global | 0.5596 | ns | 0.5827 | ns |
| Global | Chronos-2 forward-path context | 1 | 47 | 47 | 0.2519 | 0.1925 | -23.6% [-43.3, 2.0] | chronos2_C1_fwd_global | 0.0750 | * | 0.0382 | ** |
| Global | Chronos-2 forward-path context | 3 | 45 | 45 | 0.4898 | 0.4295 | -12.3% [-33.1, 16.0] | chronos2_C1_fwd_global | 0.5536 | ns | 0.4742 | ns |
| Global | Chronos-2 forward-path context | 6 | 42 | 42 | 1.1443 | 0.8924 | -22.0% [-40.0, 3.3] | chronos2_C1_fwd_global | 0.5217 | ns | 0.3667 | ns |
| Global | Chronos-2 forward-path context | 12 | 36 | 36 | 2.3131 | 1.8460 | -20.2% [-38.6, 2.0] | chronos2_C1_fwd_global | 0.4076 | ns | 0.3273 | ns |
| Global | Chronos-2 forward vs flat-hold | 1 | 47 | 47 | 0.2004 | 0.1925 | -4.0% [-13.9, 7.9] | chronos2_C1_fwd_global | 0.4825 | ns | 0.2069 | ns |
| Global | Chronos-2 forward vs flat-hold | 3 | 45 | 45 | 0.4963 | 0.4295 | -13.5% [-25.4, 1.3] | chronos2_C1_fwd_global | 0.3457 | ns | 0.2687 | ns |
| Global | Chronos-2 forward vs flat-hold | 6 | 42 | 42 | 0.9898 | 0.8924 | -9.8% [-21.1, 5.4] | chronos2_C1_fwd_global | 0.5990 | ns | 0.3342 | ns |
| Global | Chronos-2 forward vs flat-hold | 12 | 36 | 36 | 1.9698 | 1.8460 | -6.3% [-19.0, 9.4] | chronos2_C1_fwd_global | 0.6667 | ns | 0.4619 | ns |
| Global | Chronos-2 regime context | 1 | 47 | 47 | 0.2519 | 0.2487 | -1.3% [-3.1, 0.7] | chronos2_C1_regime_global | 0.2007 | ns | 0.0502 | * |
| Global | Chronos-2 regime context | 3 | 45 | 45 | 0.4898 | 0.4834 | -1.3% [-2.4, -0.4] | chronos2_C1_regime_global | 0.2005 | ns | 0.1877 | ns |
| Global | Chronos-2 regime context | 6 | 42 | 42 | 1.1443 | 1.1380 | -0.6% [-1.1, -0.1] | chronos2_C1_regime_global | 0.3346 | ns | 0.2994 | ns |
| Global | Chronos-2 regime context | 12 | 36 | 36 | 2.3131 | 2.3057 | -0.3% [-0.7, -0.1] | chronos2_C1_regime_global | 0.2400 | ns | 0.2309 | ns |
| Global | TimesFM regime context | 1 | 47 | 47 | 0.2103 | 0.2137 | 1.6% [-0.5, 4.3] | timesfm_C0_global | 0.1761 | ns | 0.4644 | ns |
| Global | TimesFM regime context | 3 | 45 | 45 | 0.5448 | 0.5486 | 0.7% [-0.1, 1.7] | timesfm_C0_global | 0.4457 | ns | 0.4441 | ns |
| Global | TimesFM regime context | 6 | 42 | 42 | 1.0760 | 1.0786 | 0.2% [-0.2, 0.8] | timesfm_C0_global | 0.5131 | ns | 0.3509 | ns |
| Global | TimesFM regime context | 12 | 36 | 36 | 2.0113 | 2.0112 | -0.0% [-0.3, 0.4] | timesfm_C1_regime_global | 0.6460 | ns | 0.1571 | ns |
| Global | Chronos-2 vs ARIMA | 1 | 47 | 47 | 0.2004 | 0.1907 | -4.9% [-21.6, 18.9] | arima | 0.6461 | ns | 0.4348 | ns |
| Global | Chronos-2 vs ARIMA | 3 | 45 | 45 | 0.4963 | 0.5393 | 8.7% [-10.0, 33.2] | chronos2_C1_inst_global | 0.6304 | ns | 0.6753 | ns |
| Global | Chronos-2 vs ARIMA | 6 | 42 | 42 | 0.9898 | 1.2439 | 25.7% [2.9, 57.4] | chronos2_C1_inst_global | 0.4939 | ns | 0.5615 | ns |
| Global | Chronos-2 vs ARIMA | 12 | 36 | 36 | 1.9698 | 2.8944 | 46.9% [14.2, 100.1] | chronos2_C1_inst_global | 0.3989 | ns | 0.4613 | ns |
| Global | Chronos-2 vs AutoARIMA | 1 | 47 | 47 | 0.2004 | 0.1787 | -10.9% [-26.2, 7.0] | auto_arima | 0.2202 | ns | 0.2527 | ns |
| Global | Chronos-2 vs AutoARIMA | 3 | 45 | 45 | 0.4963 | 0.4925 | -0.8% [-9.7, 8.5] | auto_arima | 0.9089 | ns | 0.7974 | ns |
| Global | Chronos-2 vs AutoARIMA | 6 | 42 | 42 | 0.9898 | 1.0714 | 8.2% [-2.8, 20.1] | chronos2_C1_inst_global | 0.4764 | ns | 0.3891 | ns |
| Global | Chronos-2 vs AutoARIMA | 12 | 36 | 36 | 1.9698 | 2.4821 | 26.0% [11.5, 43.9] | chronos2_C1_inst_global | 0.2764 | ns | 0.2491 | ns |
| Global | Chronos-2 validated vs ARIMA | 1 | 47 | 47 | 0.2723 | 0.1907 | -30.0% [-48.6, -2.5] | arima | 0.0407 | ** | 0.0247 | ** |
| Global | Chronos-2 validated vs ARIMA | 3 | 45 | 45 | 0.4610 | 0.5393 | 17.0% [-10.4, 55.5] | chronos2_C1_validated_global | 0.3824 | ns | 0.5519 | ns |
| Global | Chronos-2 validated vs ARIMA | 6 | 42 | 42 | 1.1066 | 1.2439 | 12.4% [-9.2, 43.1] | chronos2_C1_validated_global | 0.5661 | ns | 0.6684 | ns |
| Global | Chronos-2 validated vs ARIMA | 12 | 36 | 36 | 2.2620 | 2.8944 | 28.0% [0.4, 63.7] | chronos2_C1_validated_global | NA | NA | NA | NA |
| Global | Chronos-2 validated vs AutoARIMA | 1 | 47 | 47 | 0.2723 | 0.1787 | -34.4% [-53.1, -6.7] | auto_arima | 0.0205 | ** | 0.0284 | ** |
| Global | Chronos-2 validated vs AutoARIMA | 3 | 45 | 45 | 0.4610 | 0.4925 | 6.8% [-20.5, 45.8] | chronos2_C1_validated_global | 0.8143 | ns | 0.8302 | ns |
| Global | Chronos-2 validated vs AutoARIMA | 6 | 42 | 42 | 1.1066 | 1.0714 | -3.2% [-30.6, 35.0] | auto_arima | 0.9442 | ns | 0.9605 | ns |
| Global | Chronos-2 validated vs AutoARIMA | 12 | 36 | 36 | 2.2620 | 2.4821 | 9.7% [-16.1, 42.0] | chronos2_C1_validated_global | 0.7732 | ns | 0.5530 | ns |
| Global | TimesFM vs ARIMA | 1 | 47 | 47 | 0.2137 | 0.1907 | -10.8% [-26.7, 9.4] | arima | 0.2846 | ns | 0.3381 | ns |
| Global | TimesFM vs ARIMA | 3 | 45 | 45 | 0.5051 | 0.5393 | 6.8% [-10.1, 25.1] | timesfm_C1_inst_global | 0.6354 | ns | 0.4663 | ns |
| Global | TimesFM vs ARIMA | 6 | 42 | 42 | 1.0178 | 1.2439 | 22.2% [1.1, 49.0] | timesfm_C1_inst_global | 0.4711 | ns | 0.5520 | ns |
| Global | TimesFM vs ARIMA | 12 | 36 | 36 | 1.9708 | 2.8944 | 46.9% [13.8, 100.4] | timesfm_C1_inst_global | 0.4233 | ns | 0.5455 | ns |
| Global | TimesFM vs AutoARIMA | 1 | 47 | 47 | 0.2137 | 0.1787 | -16.4% [-33.6, 6.4] | auto_arima | 0.1374 | ns | 0.3728 | ns |
| Global | TimesFM vs AutoARIMA | 3 | 45 | 45 | 0.5051 | 0.4925 | -2.5% [-17.3, 14.3] | auto_arima | 0.8712 | ns | 0.7993 | ns |
| Global | TimesFM vs AutoARIMA | 6 | 42 | 42 | 1.0178 | 1.0714 | 5.3% [-7.7, 19.6] | timesfm_C1_inst_global | 0.7303 | ns | 0.6079 | ns |
| Global | TimesFM vs AutoARIMA | 12 | 36 | 36 | 1.9708 | 2.4821 | 25.9% [11.7, 45.1] | timesfm_C1_inst_global | 0.0664 | * | 0.2253 | ns |
| Global | TimesFM validated vs ARIMA | 1 | 47 | 47 | 0.1952 | 0.1907 | -2.3% [-17.0, 16.9] | arima | 0.7943 | ns | 0.2561 | ns |
| Global | TimesFM validated vs ARIMA | 3 | 45 | 45 | 0.5303 | 0.5393 | 1.7% [-15.3, 22.2] | timesfm_C1_validated_global | 0.9164 | ns | 0.9396 | ns |
| Global | TimesFM validated vs ARIMA | 6 | 42 | 42 | 1.0547 | 1.2439 | 17.9% [-3.5, 45.8] | timesfm_C1_validated_global | 0.5909 | ns | 0.7479 | ns |
| Global | TimesFM validated vs ARIMA | 12 | 36 | 36 | 2.0188 | 2.8944 | 43.4% [10.3, 97.5] | timesfm_C1_validated_global | 0.4736 | ns | 0.6178 | ns |
| Global | TimesFM validated vs AutoARIMA | 1 | 47 | 47 | 0.1952 | 0.1787 | -8.5% [-21.7, 7.6] | auto_arima | 0.2912 | ns | 0.2866 | ns |
| Global | TimesFM validated vs AutoARIMA | 3 | 45 | 45 | 0.5303 | 0.4925 | -7.1% [-19.4, 6.3] | auto_arima | 0.5168 | ns | 0.7382 | ns |
| Global | TimesFM validated vs AutoARIMA | 6 | 42 | 42 | 1.0547 | 1.0714 | 1.6% [-10.0, 14.7] | timesfm_C1_validated_global | 0.8935 | ns | 0.8153 | ns |
| Global | TimesFM validated vs AutoARIMA | 12 | 36 | 36 | 2.0188 | 2.4821 | 22.9% [9.0, 42.7] | timesfm_C1_validated_global | 0.0978 | * | 0.2562 | ns |
| Europe | TimesFM context | 1 | 47 | 47 | 0.3526 | 0.4358 | 23.6% [-3.8, 56.3] | timesfm_C0_europe | 0.1005 | ns | 0.1142 | ns |
| Europe | TimesFM context | 3 | 45 | 45 | 0.9799 | 0.9184 | -6.3% [-15.9, 4.1] | timesfm_C1_full_europe | 0.4853 | ns | 0.4955 | ns |
| Europe | TimesFM context | 6 | 42 | 42 | 1.5317 | 1.4582 | -4.8% [-11.7, 2.4] | timesfm_C1_full_europe | 0.4898 | ns | 0.4500 | ns |
| Europe | TimesFM context | 12 | 36 | 36 | 3.6189 | 3.6711 | 1.4% [-2.3, 5.9] | timesfm_C0_europe | 0.6715 | ns | 0.8947 | ns |
| Europe | TimesFM validated context | 1 | 47 | 47 | 0.3526 | 0.3516 | -0.3% [-1.6, 1.0] | timesfm_C1_validated_europe | 0.6711 | ns | 0.6733 | ns |
| Europe | TimesFM validated context | 3 | 45 | 45 | 0.9799 | 0.9855 | 0.6% [0.1, 1.0] | timesfm_C0_europe | 0.1179 | ns | 0.0702 | * |
| Europe | TimesFM validated context | 6 | 42 | 42 | 1.5317 | 1.5359 | 0.3% [-0.0, 0.6] | timesfm_C0_europe | 0.3536 | ns | 0.2959 | ns |
| Europe | TimesFM validated context | 12 | 36 | 36 | 3.6189 | 3.6156 | -0.1% [-0.3, 0.1] | timesfm_C1_validated_europe | 0.4801 | ns | 0.7168 | ns |
| Europe | TimesFM regime context | 1 | 47 | 47 | 0.3526 | 0.3517 | -0.3% [-1.2, 0.7] | timesfm_C1_regime_europe | 0.5922 | ns | 0.4951 | ns |
| Europe | TimesFM regime context | 3 | 45 | 45 | 0.9799 | 0.9841 | 0.4% [0.1, 0.8] | timesfm_C0_europe | 0.0577 | * | 0.0969 | * |
| Europe | TimesFM regime context | 6 | 42 | 42 | 1.5317 | 1.5351 | 0.2% [-0.0, 0.5] | timesfm_C0_europe | 0.2511 | ns | 0.2536 | ns |
| Europe | TimesFM regime context | 12 | 36 | 36 | 3.6189 | 3.6147 | -0.1% [-0.3, -0.0] | timesfm_C1_regime_europe | 0.2250 | ns | 0.3876 | ns |
| Europe | Chronos-2 regime context | 1 | 47 | 47 | 0.5124 | 0.5113 | -0.2% [-0.8, 0.5] | chronos2_C1_regime_europe | 0.4896 | ns | 0.0871 | * |
| Europe | Chronos-2 regime context | 3 | 45 | 45 | 1.1131 | 1.1128 | -0.0% [-0.4, 0.3] | chronos2_C1_regime_europe | 0.9193 | ns | 0.9088 | ns |
| Europe | Chronos-2 regime context | 6 | 42 | 42 | 1.9216 | 1.9228 | 0.1% [-0.2, 0.2] | chronos2_C0_europe | 0.6967 | ns | 0.4928 | ns |
| Europe | Chronos-2 regime context | 12 | 36 | 36 | 4.1149 | 4.1171 | 0.1% [-0.1, 0.2] | chronos2_C0_europe | NA | NA | NA | NA |
| Europe | Chronos-2 forward-path context | 1 | 47 | 47 | 0.5124 | 0.5639 | 10.0% [-10.9, 41.4] | chronos2_C0_europe | 0.4229 | ns | 0.7715 | ns |
| Europe | Chronos-2 forward-path context | 3 | 45 | 45 | 1.1131 | 1.0880 | -2.3% [-15.6, 12.7] | chronos2_C1_fwd_europe | 0.8607 | ns | 0.9801 | ns |
| Europe | Chronos-2 forward-path context | 6 | 42 | 42 | 1.9216 | 1.8942 | -1.4% [-15.5, 13.9] | chronos2_C1_fwd_europe | 0.9386 | ns | 0.8242 | ns |
| Europe | Chronos-2 forward-path context | 12 | 36 | 36 | 4.1149 | 3.8590 | -6.2% [-16.1, 4.0] | chronos2_C1_fwd_europe | 0.1575 | ns | 0.0208 | ** |
| Europe | Chronos-2 forward vs flat-hold | 1 | 47 | 47 | 0.5513 | 0.5639 | 2.3% [-5.4, 10.1] | chronos2_C1_inst_europe | 0.5527 | ns | 0.6959 | ns |
| Europe | Chronos-2 forward vs flat-hold | 3 | 45 | 45 | 1.1514 | 1.0880 | -5.5% [-12.2, 1.8] | chronos2_C1_fwd_europe | 0.3842 | ns | 0.3237 | ns |
| Europe | Chronos-2 forward vs flat-hold | 6 | 42 | 42 | 2.0333 | 1.8942 | -6.8% [-13.4, 0.8] | chronos2_C1_fwd_europe | 0.4492 | ns | 0.3528 | ns |
| Europe | Chronos-2 forward vs flat-hold | 12 | 36 | 36 | 4.0739 | 3.8590 | -5.3% [-11.5, 2.0] | chronos2_C1_fwd_europe | 0.6019 | ns | 0.4566 | ns |
| Europe | TimesFM vs SARIMA | 1 | 47 | 47 | 0.4358 | 0.4126 | -5.3% [-28.4, 29.2] | sarima | 0.7139 | ns | 0.3713 | ns |
| Europe | TimesFM vs SARIMA | 3 | 45 | 45 | 0.9184 | 1.0195 | 11.0% [-10.0, 37.7] | timesfm_C1_full_europe | 0.5439 | ns | 0.4011 | ns |
| Europe | TimesFM vs SARIMA | 6 | 42 | 42 | 1.4582 | 2.0831 | 42.9% [23.4, 70.2] | timesfm_C1_full_europe | 0.0038 | ** | 0.0502 | * |
| Europe | TimesFM vs SARIMA | 12 | 36 | 36 | 3.6711 | 4.2537 | 15.9% [5.3, 31.4] | timesfm_C1_full_europe | 0.1213 | ns | 0.0041 | ** |
| Europe | TimesFM vs AutoARIMA | 1 | 47 | 47 | 0.4358 | 0.3758 | -13.7% [-32.7, 11.6] | auto_arima | 0.2600 | ns | 0.1234 | ns |
| Europe | TimesFM vs AutoARIMA | 3 | 45 | 45 | 0.9184 | 0.9579 | 4.3% [-11.4, 21.8] | timesfm_C1_full_europe | 0.6681 | ns | 0.4535 | ns |
| Europe | TimesFM vs AutoARIMA | 6 | 42 | 42 | 1.4582 | 1.9586 | 34.3% [15.2, 60.1] | timesfm_C1_full_europe | 0.1283 | ns | 0.0481 | ** |
| Europe | TimesFM vs AutoARIMA | 12 | 36 | 36 | 3.6711 | 4.7976 | 30.7% [14.8, 52.2] | timesfm_C1_full_europe | 0.1791 | ns | 0.1287 | ns |
| Europe | TimesFM validated vs SARIMA | 1 | 47 | 47 | 0.3516 | 0.4126 | 17.3% [-4.4, 45.4] | timesfm_C1_validated_europe | 0.1227 | ns | 0.2353 | ns |
| Europe | TimesFM validated vs SARIMA | 3 | 45 | 45 | 0.9855 | 1.0195 | 3.5% [-12.0, 20.9] | timesfm_C1_validated_europe | 0.7568 | ns | 0.6366 | ns |
| Europe | TimesFM validated vs SARIMA | 6 | 42 | 42 | 1.5359 | 2.0831 | 35.6% [20.8, 58.2] | timesfm_C1_validated_europe | 0.0013 | ** | 0.0110 | ** |
| Europe | TimesFM validated vs SARIMA | 12 | 36 | 36 | 3.6156 | 4.2537 | 17.6% [7.7, 33.0] | timesfm_C1_validated_europe | 0.2585 | ns | 0.4447 | ns |
| Europe | TimesFM validated vs AutoARIMA | 1 | 47 | 47 | 0.3516 | 0.3758 | 6.9% [-10.7, 28.3] | timesfm_C1_validated_europe | 0.4729 | ns | 0.6459 | ns |
| Europe | TimesFM validated vs AutoARIMA | 3 | 45 | 45 | 0.9855 | 0.9579 | -2.8% [-19.6, 16.0] | auto_arima | 0.8410 | ns | 0.9577 | ns |
| Europe | TimesFM validated vs AutoARIMA | 6 | 42 | 42 | 1.5359 | 1.9586 | 27.5% [6.4, 57.3] | timesfm_C1_validated_europe | 0.2585 | ns | 0.2032 | ns |
| Europe | TimesFM validated vs AutoARIMA | 12 | 36 | 36 | 3.6156 | 4.7976 | 32.7% [14.3, 58.0] | timesfm_C1_validated_europe | 0.2289 | ns | 0.2073 | ns |

## Notes

- Path-level rows pool every step in each forecast path for the requested horizon.
- Endpoint-only rows use only step == horizon, so h12 has fewer paired observations.
- Bootstrap intervals are paired by forecast origin to preserve within-origin dependence.

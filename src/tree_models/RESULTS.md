# Tree-model comparison results

Completed nested LOLO-CV comparison. RMSE and MAE are kg/ha; R² is pooled over OOF predictions. These are descriptive performance estimates; uncertainty intervals and paired exploratory analyses remain outstanding.

## main_min5: primary 0–20 cm soil

| Model | Tier | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| CatBoost | 1 | 2138.3 | 1251.1 | -0.103 |
| CatBoost | 2 | 2143.6 | 1257.9 | -0.109 |
| CatBoost | 3 | 2096.7 | 1232.7 | -0.061 |
| MLR | 1 | 1989.5 | 1250.1 | 0.045 |
| MLR | 2 | 2068.7 | 1333.4 | -0.033 |
| MLR | 3 | 2157.5 | 1452.2 | -0.123 |
| RF | 1 | 2066.5 | 1208.4 | -0.031 |
| RF | 2 | 2063.2 | 1246.3 | -0.027 |
| RF | 3 | 2049.1 | 1248.2 | -0.013 |
| Ridge | 3 | 2064.4 | 1241.6 | -0.028 |
| XGBoost | 1 | 2108.4 | 1220.6 | -0.073 |
| XGBoost | 2 | 2072.8 | 1187.9 | -0.037 |
| XGBoost | 3 | 2077.0 | 1215.0 | -0.041 |
| mean | 0 | 2073.2 | 1212.6 | -0.037 |

## Soil depth sensitivity: main_min5

| Model | Full-tier RMSE 0–20 cm | Full-tier RMSE 20–50 cm |
|---|---:|---:|
| MLR | 2157.5 | 2239.2 |
| Ridge | 2064.4 | 2065.0 |
| RF | 2049.1 | 2037.1 |
| XGBoost | 2077.0 | 2072.4 |
| CatBoost | 2096.7 | 2084.9 |

## all_cells: primary 0–20 cm soil

| Model | Tier | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| CatBoost | 1 | 2110.1 | 1256.3 | -0.055 |
| CatBoost | 2 | 2080.4 | 1224.2 | -0.025 |
| CatBoost | 3 | 2060.6 | 1224.8 | -0.006 |
| MLR | 1 | 1979.0 | 1251.0 | 0.072 |
| MLR | 2 | 2057.6 | 1346.6 | -0.003 |
| MLR | 3 | 2140.7 | 1407.5 | -0.086 |
| RF | 1 | 2071.8 | 1263.0 | -0.017 |
| RF | 2 | 2048.4 | 1260.9 | 0.006 |
| RF | 3 | 2037.4 | 1253.4 | 0.017 |
| Ridge | 3 | 2049.1 | 1265.0 | 0.005 |
| XGBoost | 1 | 2136.7 | 1267.9 | -0.082 |
| XGBoost | 2 | 2192.4 | 1308.9 | -0.139 |
| XGBoost | 3 | 2105.6 | 1265.2 | -0.050 |
| mean | 0 | 2089.9 | 1251.3 | -0.035 |

## Soil depth sensitivity: all_cells

| Model | Full-tier RMSE 0–20 cm | Full-tier RMSE 20–50 cm |
|---|---:|---:|
| MLR | 2140.7 | 2237.2 |
| Ridge | 2049.1 | 2046.7 |
| RF | 2037.4 | 2036.6 |
| XGBoost | 2105.6 | 2172.6 |
| CatBoost | 2060.6 | 2064.0 |

## Checks and interpretation

All 420 tuning tasks completed. 7074 completed trials and 1326 pruned trials, from 20 attempted trials per task. Median pruning reduces unnecessary fits; a pruned trial is not a fully evaluated parameter candidate.

The accepted primary soil depth and both reference yield panels were retained. This comparison must not be used to choose a yield-cleaning policy. Added sources can alter performance in either direction in this small panel; these results do not demonstrate causal predictor effects. No tree-based feature pruning, held-out-LGA tuning, prediction clipping or target transformations were applied.

Next step: fully nested stacking, which needs separate tuning inside each outer-training OOF generation split. Then complete LOYO sensitivity, uncertainty characterisation, paired exploratory comparisons and final-model SHAP.

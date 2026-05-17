# Task 2: Forecast-only RMSE Back-calculation

Source: latest inference run `checkpoints_20260516_101105`

## Formula

```
skill = 1 - increment_rmse / forecast_only_rmse
forecast_only_rmse = increment_rmse / (1 - skill)
```

## Results

| Split       | Variable  | Skill       | Inc RMSE    | Forecast-Only RMSE | Inc/Fcst Ratio |
|:------------|:----------|------------:|------------:|-------------------:|---------------:|
| source_val  | surface   | +0.4369 | 0.003001 | 0.00532912 | 0.56x |
| source_val  | rootzone  | -1.6062 | 0.000642 | 0.00024632 | 2.61x |
| target_query | surface   | -1.9504 | 0.004401 | 0.00149176 | 2.95x |
| target_query | rootzone  | -12.2087 | 0.000579 | 0.00004386 | 13.21x |

## Key Findings

- **Surface**: forecast-only RMSE is 0.005329 (source) / 0.001492 (target)
  - Source-only inc RMSE is 0.6x (source) / 3.0x (target) forecast-only
- **Rootzone**: forecast-only RMSE is 0.000246 (source) / 0.00004386 (target)
  - Source-only inc RMSE is 2.6x (source) / 13.2x (target) forecast-only
  - Target_query rootzone forecast-only RMSE is **extremely small** (4.39e-05)
- **Rootzone skill < 0 on source_val** (-1.6) confirms model is worse than forecast-only even in-domain
- **Rootzone target_query skill = -12.2** is dominated by tiny denominator

> **Conclusion**: The large negative rootzone skill is primarily driven by an extremely small forecast-only RMSE denominator, not by catastrophically large prediction errors. The increment RMSE itself is small (~5-6e-4), but the forecast is already so good that any nonzero correction degrades performance.
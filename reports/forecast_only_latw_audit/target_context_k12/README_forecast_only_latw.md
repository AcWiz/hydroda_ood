# P0 Forecast-only Latitude-weighted RMSE Audit

## Protocol

- **Protocol freeze ID**: `hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12`
- **K**: 12, **seed**: 0
- **Generated**: 2026-05-19T00:50:14.246747

## Splits evaluated

| Split | Role |
|-------|------|
| target_context | target_support |

## Sanity check results

- **forecast_only_skill_is_zero**: `skip` — Column analysis_skill_vs_forecast not in rows
- **analysis_mse_latw_equals_increment_mse_latw**: `unknown` — 

## Output files

- `metrics_long_forecast_only_latw.csv` — per-sample, per-variable, per-metric long-format rows
- `summary_forecast_only_latw_by_split_variable.csv` — aggregated by split + variable
- `summary_forecast_only_latw_by_region.csv` — aggregated by split + region + variable
- `forecast_only_latw_diagnostics.json` — sanity checks + coverage stats

## Aggregation protocol

Primary metric: `analysis_rmse_latw = sqrt(mean(analysis_mse_latw over samples))`
即 per-sample weighted MSE → 时间平均 → sqrt

Diagnostic: `analysis_rmse_sqrt_before_time_avg_mean_latw = mean(sqrt(per_sample_mse_latw))`

## No-leakage declaration

This audit evaluates **forecast-only** (pred_increment=0) and therefore does not
train any model. target_query labels are used ONLY as evaluation labels post-prediction.
No normalization, early stopping, model selection, or threshold calibration uses
target_query labels.
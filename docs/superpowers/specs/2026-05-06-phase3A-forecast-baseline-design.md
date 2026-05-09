# Phase 3A — Forecast-Only Baseline and Metrics Harness

**Date:** 2026-05-06
**Author:** Claude
**Status:** Approved

## Overview

Phase 3A implements the forecast-only baseline (`pred_increment = 0`) and a metrics/evaluation harness that will be reused by all Phase 3–5 models. This is a pure sanity baseline: the model predicts zero increment, so `pred_analysis = forecast`.

## Components

### `hydroda/baselines/forecast.py`
`ForecastBaseline` class with `predict(sample) → dict` returning `pred_increment_surface/rootzone` as zero arrays and `pred_analysis_surface/rootzone` as copies of the forecast fields. `method_id = "forecast_only"`.

### `hydroda/metrics/skill.py`
Pure numpy metric functions (no torch, no model) that operate on 2D arrays with a scalar `metric_mask > 0.5` filter:

| Function | Returns |
|---|---|
| `analysis_rmse(pred, true, mask)` | RMSE of analysis |
| `analysis_mae(pred, true, mask)` | MAE of analysis |
| `analysis_skill_vs_forecast(pred, true, forecast, mask)` | `1 - rmse(p,a)/rmse(f,a)`, range ≈ 0 for forecast-only |
| `increment_rmse(pred_inc, true_inc, mask)` | RMSE of increment |
| `increment_mae(pred_inc, true_inc, mask)` | MAE of increment |
| `increment_bias(pred_inc, true_inc, mask)` | Mean signed error |
| `increment_corr(pred_inc, true_inc, mask)` | Pearson correlation |
| `sign_accuracy_deadzone(pred_inc, true_inc, mask, epsilon=0.005)` | Fraction where sign matches in deadzone |
| `valid_pixel_count(mask)` | Count of `mask > 0.5` pixels |
| `effective_mask_fraction(mask)` | Fraction of total pixels that are valid |

All functions return `np.nan` if no valid pixels after masking.

### `hydroda/evaluation/harness.py`
`evaluate_split(dataset, predictor, split_role, experiment_id, protocol_freeze_id) → List[Dict]`

Iterates all samples in `dataset`, calls `predictor.predict(sample)`, computes per-sample per-variable metrics, returns a list of result dicts (long format, one row per `{metric, variable}` combination). Uses `sample["metric_mask"]` for masking.

### `scripts/run_phase3A.py`
Entry point script that:
1. Builds 240 `HydroDADataset` instances across all 6 regions × 4 K values × 10 seeds, for all 3 split roles
2. Evaluates each with `ForecastBaseline`
3. Aggregates results into `artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv`

### `tests/test_forecast_baseline.py`
7 sanity tests:
1. `test_forecast_baseline_pred_increment_zero` — all zeros
2. `test_pred_analysis_equals_forecast_for_forecast_baseline` — exact match
3. `test_forecast_skill_is_zero` — `analysis_skill ≈ 0` (rtol=0.1, atol=0.05)
4. `test_metrics_respect_metric_mask` — values zero outside mask
5. `test_metrics_long_schema` — CSV columns match spec exactly
6. `test_no_target_query_training_in_baseline` — no fitting, no target_query labels
7. `test_region_balanced_aggregation` — per-region mean weighted equally

## Output

**CSV:** `artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv`
Columns: `experiment_id,method,country_id,target_region_id,active_region_ids,split_role,K,seed,variable,metric,value,n_valid_pixels,n_time_steps,protocol_freeze_id`

**Report:** `reports/experiments/phase3A_forecast_only_US.md`
Per-region RMSE tables for surface/rootzone, valid coverage %, forecast-only sanity table, warnings for US-R3/R4/R6, no-leakage statement.

## No-Leakage Compliance

- Forecast baseline uses NO training data at all
- `metric_mask` derived from `base_valid_mask` + `active_region_mask` + finite check (no target_query labels)
- No fitting step; no use of analysis increments for normalization or selection

## Verification

```bash
python scripts/run_phase3A.py
python -m pytest tests/test_forecast_baseline.py -v
python -c "import pandas as pd; df = pd.read_csv('artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv'); print(df.columns.tolist(), df.shape)"
```

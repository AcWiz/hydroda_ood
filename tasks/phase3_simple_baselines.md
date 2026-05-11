# Phase 3 — Forecast-only 与 Metrics/Evaluation Sanity

## 目标

V4 不再把 source mean、target mean、ridge 等 heuristic baselines 放入论文主表。Phase 3 的目标是建立可信 evaluation 闭环：forecast-only、metric routing、mask、analysis reconstruction、region-balanced aggregation 全部正确。

## 必读

```text
CLAUDE.md
context/01_RESEARCH_CONTRACT.md
specs/protocol_v4.yaml
specs/metrics.yaml
checklists/no_leakage_checklist.md
```

## 需要实现或修复

```text
hydroda/baselines/forecast.py
hydroda/metrics/skill.py
hydroda/evaluation/harness.py
hydroda/data/protocol.py
hydroda/data/leakage_guard.py
tests/test_metrics.py
tests/test_evaluation_harness_metric_routing.py
scripts/run_forecast_only.py
```

## 必须修复的问题

```text
1. evaluation harness 不能把 pred_analysis 当成 pred_increment 计算 increment metrics。
2. evaluation harness 不能引用未定义的 SPLITS_JSON。
3. method 字段必须由 caller 显式传入，不能硬编码 forecast。
4. Dataset sample 必须包含 month、season，便于 prompt 和 stratified evaluation。
5. forecast-only 的 surface/rootzone Skill 应精确接近 0。
6. perfect increment prediction 的 Skill 应接近 1。
```

## Baseline 策略

论文主表 Phase 3 只输出：

```text
Forecast-only
```

以下旧 baseline 如保留代码，只能标记为 internal_sanity，不得进入 paper main table：

```text
source_mean_increment
target_support_mean_increment
monthly_mean_increment
ridge_calibration
```

## 输出

```text
artifacts/results/forecast_only/{split_id}/metrics_long.csv
artifacts/results/forecast_only/{split_id}/summary.json
reports/experiments/forecast_only_us_dev.md
```

## 验收标准

```text
1. forecast-only pred_increment 全零。
2. forecast-only pred_analysis == forecast。
3. forecast-only skill == 0，允许浮点误差。
4. increment_rmse 使用 pred_increment 和 true_increment，不使用 pred_analysis。
5. no-leakage checklist 全部通过。
```

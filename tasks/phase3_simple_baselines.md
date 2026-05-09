# Phase 3 — Simple Baselines

## 目标

先实现审稿人最关心的强简单基线，证明任务不是普通均值校正即可解决。

## 必读

```text
context/05_METRICS_AND_REPORTING.md
context/06_BASELINES_AND_MODEL_ROADMAP.md
specs/baselines.yaml
specs/metrics.yaml
```

## 需要实现

```text
hydroda/baselines/forecast.py
hydroda/baselines/mean_increment.py
hydroda/baselines/monthly_mean.py
hydroda/baselines/ridge.py
hydroda/metrics/skill.py
hydroda/metrics/increment.py
hydroda/metrics/aggregation.py
scripts/run_baseline.py
scripts/evaluate.py
tests/test_metrics.py
tests/test_simple_baselines.py
```

## Baselines

```text
forecast
source_mean_increment
target_support_mean_increment
target_monthly_support_increment
ridge
```

## 输出

```text
artifacts/metrics/{experiment_id}/metrics_long.csv
reports/experiments/{experiment_id}.md
reports/tables/simple_baselines_US.csv
```

## 验收标准

```text
1. forecast baseline 的 pred_increment 全零。
2. perfect prediction sanity skill = 1。
3. forecast baseline sanity skill = 0。
4. metrics 分 surface/rootzone 报告。
5. 主表使用 region-balanced aggregation。
6. K=0 不运行 target support label baselines。
```

# Phase 6 — Reporting and Paper Artifacts

## 目标

从 metrics/artifacts 生成论文级表格、曲线、事件图和实验报告。

## 需要实现

```text
hydroda/reports/tables.py
hydroda/reports/figures.py
scripts/make_tables.py
scripts/make_figures.py
scripts/make_experiment_report.py
```

## 输出

```text
reports/tables/table_simple_baselines.csv
reports/tables/table_k_curves.csv
reports/tables/table_adaptation_ablation.csv
reports/figures/k_curve_*.png
reports/figures/event_case_*.png
reports/experiments/summary.md
```

## 验收标准

```text
1. 所有表格只从 metrics_long.csv 生成。
2. 每个数字可追溯到 experiment_id。
3. region-balanced 和 pixel-weighted 分开。
4. event cases 不用于模型选择。
5. 报告明确列出 failed / missing runs。
```

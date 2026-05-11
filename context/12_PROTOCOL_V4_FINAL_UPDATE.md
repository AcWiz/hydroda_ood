# 12_PROTOCOL_V4_FINAL_UPDATE.md — Protocol V4-final 冻结说明

本文件记录 2026-05-11 冻结的新时间协议。若旧文档、旧任务、旧 artifacts 中仍出现旧版时间协议引用，以本文件、`CLAUDE.md`、`完整研究计划方案.md`、`specs/protocol_v4.yaml` 为准。

## 冻结协议

```text
source_fit:     2015-01-01 to 2020-12-31
source_val:     2021-01-01 to 2021-12-31  source continents only
target_context: 2022-01-01 to 2022-12-31  held-out target continent; K-cycle calibration only
target_query:   2023-01-01 to 2025-12-31  final offline evaluation only
```

## 关键纪律

`source_val=2021` 只能来自 source continents，用于 checkpoint selection、early stopping、hyperparameter selection 和 architecture selection。`target_context=2022` 只能来自 held-out target continent，用于 K-cycle calibration、adapter/LoRA/HyperDA-Calib/HyperDA-Refine 的 target adaptation 或 calibration prompt summary。`target_context` 不能用于 checkpoint 或超参数选择。`target_query=2023-2025` 只能用于最终评估，不能用于 prompt construction、normalization、support-date selection、training、early stopping、model selection、threshold calibration 或 region definition。

## K 定义

K 是 target context year 2022 中的 labeled DA analysis cycles 数量，不是 patch 数、pixel 数或 mini-batch 数。主实验只使用 `K ∈ {0, 4, 12}`；K=24 若保留，只能作为 optional internal ablation，不能进入主表。

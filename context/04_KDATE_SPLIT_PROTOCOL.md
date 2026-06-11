# 04 Zero/Few-Shot Split Protocol

## 1. 时间协议

V4.4 主协议：

```text
source_fit:     2015-01-01 到 2021-12-31
source_val:     2022-01-01 到 2022-12-31
target_context: 2015-01-01 到 2021-12-31 input-side only
target_support: K labeled cycles from 2015-2021, K in {0,4,12}
target_val:     unused in main protocol
target_eval:    2023-01-01 到 2025-12-31
```

如果实际 DA.nc 不覆盖这些年份，split builder 不能静默替换时间协议；必须报告冲突并进入 degraded protocol。

---

## 2. 主协议：zero/few-shot generalization

目标域适配/泛化使用：

```text
adaptation_setting ∈ {zero_shot_context, few_shot_k4, few_shot_k12}
target_context_dates = all available 2015-2021 target input-side cycles
target_support_dates = K labeled target cycles selected by calendar + availability/base_valid coverage
target_eval_dates = all frozen held-out target_eval cycles in 2023-2025
```

K=0 不使用 target labels；K=4/12 只允许轻量 target-specific 变量在 K 个
labeled support cycles 上更新。`target_val=2022` 不用于主协议 checkpoint
selection、early stopping 或 residual-gain calibration。2023-2025 target_eval
labels 只能在预测后用于最终 metric computation。

`target_full_train` 只作为 legacy/internal reproduction 路径，必须显式 opt-in。

---

## 3. Source 覆盖范围

主协议保持：

```text
source_fit:   2015-2021
source_val:   2022
```

理由：2022 是 source-domain checkpoint / early stopping / hyperparameter /
architecture selection gate。把 source 2022 混入主 source_fit 会移除 clean
selection gate。

允许但必须标注为 secondary ablation：

```text
expanded_source_fit_2015_2022_extra_data_ablation
```

---

## 4. Legacy K-date ablation

K-shot support selection 是主协议的一部分：

```text
K=0:  不使用 target analysis labels；使用 target_context input-only stream
K=4:  每个季节选 1 个 valid target_support date
K=12: 每个月选 1 个 valid target_support date
K=24: optional internal ablation only
```

K 表示 target_support 2015-2021 中 labeled DA analysis dates/cycles 数量，不是
patch count、pixel count、mini-batch count 或 sample count after augmentation。

Valid support date 只能根据：

```text
calendar constraints
input availability
mask / finite input ratio
region coverage
```

禁止根据：

```text
analysis increment magnitude
model error
target_eval/query performance
```

---

## 5. Split manifest

每次生成 splits 必须输出机器可读 manifest。主协议内容至少包括：

```json
{
  "benchmark_id": "hydroda_ood_us_v1",
  "protocol_freeze_id": "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025",
  "country_id": "US",
  "target_region_id": "US-R1",
  "source_region_ids": ["US-R2", "US-R3", "US-R4", "US-R5", "US-R6"],
  "adaptation_setting": "few_shot_k4",
  "K": 4,
  "source_train_dates": [],
  "source_val_dates": [],
  "target_context_dates": [],
  "target_support_dates": [],
  "target_eval_dates": [],
  "target_context_dates_hash": "",
  "target_support_dates_hash": "",
  "target_eval_dates_hash": "",
  "selection_uses_analysis": false,
  "selection_uses_query_labels": false,
  "target_eval_labels_used_for_training": false,
  "target_eval_labels_used_for_prompt": false,
  "target_eval_labels_used_for_normalization": false,
  "target_eval_labels_used_for_model_selection": false
}
```

旧字段 `target_train_dates` 和 `target_query_dates` 可以作为兼容 alias 保留，
但新代码和表格应优先使用 `target_context_dates`、`target_support_dates` 与
`target_eval_dates`。

---

## 6. Leakage tests

必须覆盖：

```text
test_target_context_dates_in_2015_2021_only
test_target_support_dates_in_2015_2021_only
test_source_val_dates_in_2022_only
test_target_eval_dates_in_2023_2025_only
test_no_overlap_between_target_support_and_eval
test_manifest_flags_no_eval_label_usage
test_normalization_excludes_target_support_and_target_eval
test_model_selection_excludes_target_eval
test_target_val_unused_in_main_protocol
```

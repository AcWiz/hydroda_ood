# 04 Target-Train Split Protocol and Legacy K-date Ablation

## 1. 时间协议

V4.3 主协议：

```text
source_fit:   2015-01-01 到 2021-12-31
source_val:   2022-01-01 到 2022-12-31
target_train: 2015-01-01 到 2021-12-31
target_val:   2022-01-01 到 2022-12-31
target_eval:  2023-01-01 到 2025-12-31
```

如果实际 DA.nc 不覆盖这些年份，split builder 不能静默替换时间协议；必须报告冲突并进入 degraded protocol。

---

## 2. 主协议：target_full_train

主协议不再从目标域采样 K 个 support dates。目标域适配/泛化使用：

```text
adaptation_setting = target_full_train
target_adaptation_dates = all available labeled target_train cycles in 2015-2021
target_val_dates = target 2022 only for preregistered adaptation validation
target_eval_dates = all frozen held-out target_eval cycles in 2023-2025
```

这些 2015-2021 target_train labels 可用于构造 target-specific operator / prompt /
adapter / generated parameters。2023-2025 target_eval labels 只能在预测后用于
最终 metric computation。

主协议中 `target_support` 只作为旧代码别名，不应作为论文主表概念。

---

## 3. Source 覆盖范围

主协议保持：

```text
source_fit:   2015-2021
source_val:   2022
```

理由：2022 是 source-domain checkpoint / early stopping / hyperparameter /
architecture selection gate。source_fit 扩展到 2021，以匹配 target historical
adaptation window；把 source 2022 混入主 source_fit 会移除 clean selection gate。

允许但必须标注为 secondary ablation：

```text
expanded_source_fit_2015_2022_extra_data_ablation
```

---

## 4. Legacy K-date ablation

旧 K-date 协议仅保留为 secondary few-shot ablation：

```text
K=0:  不使用 target_train analysis labels；可使用 target_train input-only stream
K=4:  每个季节/水文季度选 1 个 valid target_train date
K=12: 每个月选 1 个 valid target_train date
K=24: optional internal ablation only
```

K 表示 target_train 2015-2021 中 labeled DA analysis dates/cycles 数量，不是
patch count、pixel count、mini-batch count 或 sample count after augmentation。

Legacy valid support date 只能根据：

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
  "protocol_freeze_id": "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025",
  "country_id": "US",
  "target_region_id": "US-R1",
  "source_region_ids": ["US-R2", "US-R3", "US-R4", "US-R5", "US-R6"],
  "adaptation_setting": "target_full_train",
  "K": null,
  "K_legacy": null,
  "source_train_dates": [],
  "source_val_dates": [],
  "target_train_dates": [],
  "target_adaptation_dates": [],
  "target_eval_dates": [],
  "target_train_dates_hash": "",
  "target_eval_dates_hash": "",
  "selection_uses_analysis": false,
  "selection_uses_query_labels": false,
  "target_eval_labels_used_for_training": false,
  "target_eval_labels_used_for_prompt": false,
  "target_eval_labels_used_for_normalization": false,
  "target_eval_labels_used_for_model_selection": false
}
```

旧字段 `target_support_dates` 和 `target_query_dates` 可以作为兼容 alias 保留，
但新代码和表格应优先使用 `target_train_dates` 与 `target_eval_dates`。

---

## 6. Leakage tests

必须覆盖：

```text
test_target_train_dates_in_2015_2021_only
test_source_val_dates_in_2022_only
test_target_eval_dates_in_2023_2025_only
test_no_overlap_between_target_train_and_eval
test_manifest_flags_no_eval_label_usage
test_normalization_excludes_target_train_and_target_eval
test_model_selection_excludes_target_eval
test_legacy_kshot_marked_secondary
```

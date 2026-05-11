# 04 K-date Split Protocol

## 1. 时间协议

V4-final 冻结版：

```text
source_fit:     2015-01-01 到 2020-12-31
source_val:     2021-01-01 到 2021-12-31
target_support: 2022-01-01 到 2022-12-31
target_query:   2023-01-01 到 2025-12-31
```

如果实际 DA.nc 不覆盖这些年份，split builder 不能静默替换时间协议；必须报告冲突并进入 degraded protocol。

---

## 2. K 的含义

```text
K = number of target DA analysis dates/cycles
```

不是：

```text
patch count
pixel count
mini-batch count
sample count after augmentation
```

---

## 3. K-date selection rules

```text
K=0:
  不使用 target support analysis labels。
  可使用 target 2022 input-only stream 构建 region descriptor。

注：K=0 时 target_context year (2022) 只作为 input-only，无 labels。

K=4:
  每个季节/水文季度选 1 个 valid support date。

K=12:
  每个月选 1 个 valid support date。

K=24:
  每个月选 2 个 support dates：上半月 1 个，下半月 1 个。
```

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
query performance
```

---

## 4. Seeds

默认：

```text
seed = 0..4 for development
seed = 0..9 for final paper
```

同一个 `country, region, K, seed` 下所有方法必须使用完全相同的 support dates。

---

## 5. Split manifest

每次生成 splits 必须输出：

```text
artifacts/splits/{benchmark_id}/{country}/{region}/K{K}_seed{seed}.json
```

内容至少包括：

```json
{
  "benchmark_id": "hydroda_ood_us_v1",
  "country_id": "US",
  "target_region_id": "US-R1",
  "source_region_ids": ["US-R2", "US-R3", "US-R4", "US-R5", "US-R6"],
  "K": 4,
  "seed": 0,
  "source_train_dates": [],
  "target_support_dates": [],
  "target_query_dates": [],
  "selection_uses_analysis": false,
  "selection_uses_query_labels": false,
  "created_by": "scripts/build_splits.py"
}
```

---

## 6. Leakage tests

Phase 2 必须实现：

```text
test_support_dates_in_2022_only
test_query_dates_after_2023_only
test_no_overlap_between_support_and_query
test_same_support_dates_across_methods
test_k0_has_no_support_labels
test_manifest_flags_no_query_label_usage
```

V4-final 关键约束：
- target 2022 只能用于 K-shot calibration (target_support)
- target 2023–2025 只能用于最终 evaluation (target_query)
- target 2022 不能用于 early stopping / hyperparameter selection

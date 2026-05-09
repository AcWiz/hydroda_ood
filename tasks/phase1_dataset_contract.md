# Phase 1 — Dataset Contract

## 目标

实现 `HydroDADataset`，显式返回 forecast、analysis、increment、mask、metadata。

## 必读

```text
context/02_DATA_AND_LEAKAGE_CONTRACT.md
specs/hydroda_dataset_contract.yaml
```

## 需要实现

```text
hydroda/data/dataset.py
hydroda/data/normalization.py
hydroda/data/patch_sampler.py
tests/test_dataset_contract.py
```

## Dataset invariant

```text
increment = analysis - forecast
pred_analysis = forecast + pred_increment
```

测试必须覆盖：

```text
sample keys
shape consistency
increment reconstruction
loss_mask finite behavior
region_mask and obs_mask not conflated
normalization provenance
```

## 禁止

```text
禁止用 target query years 计算 normalization
禁止把 obs_mask 当 loss_mask
禁止在 dataset 里硬编码 US-R1 等区域逻辑
```

## 验收标准

```text
1. 可以读取至少一个 date/region/patch sample。
2. sample 满足 specs/hydroda_dataset_contract.yaml。
3. test_increment_reconstruction 通过。
4. Dataset 不依赖具体 target region 的 hard-coded 分支。
```

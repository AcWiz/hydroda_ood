# Phase 2 — Region Masks and K-date Splits

## 目标

构建 region masks，生成 leave-one-region-out K-date split manifests。

Phase 2 分成两个可分离部分：

```text
Phase 2A: temporal K-date split manifests
Phase 2B: spatial region masks
```

其中 Phase 2B 的正式 US-R1..US-R6 scientific region masks 依赖 geolocation gate。

---

## 必读

```text
context/03_REGION_PROTOCOL.md
context/04_KDATE_SPLIT_PROTOCOL.md
context/11_GEOLOCATION_RECOVERY_PROTOCOL.md
specs/regions_v2.yaml
specs/kdate_protocol.yaml
specs/geolocation_sources.yaml
checklists/no_leakage_checklist.md
```

---

## Gate rules

### Temporal K-date splits

可以在 Phase 1 dataset contract 完成后实现：

```text
source_val: 2021
target_context: 2022
target_query: 2023–2025
K: 0, 4, 12
```

但 split manifest 中的 region_id 可以先处于 pending 状态，或使用 development-only grid region id。

### Scientific US-R1..US-R6 masks

必须满足：

```text
artifacts/geolocation/US_latlon.nc exists
lat/lon shape compatible with [256, 640]
lat/lon sanity checks pass
```

否则禁止输出正式：

```text
US-R1 Southwest Desert / Four Corners
US-R2 Southern Great Plains
US-R3 California Central Valley
US-R4 Corn Belt
US-R5 Southeast US
US-R6 Central Rockies
```

### Development-only fallback

如果 geolocation 仍未恢复，可以构建：

```text
US-GRID-R1..US-GRID-R6
```

但所有文件和报告必须标注：

```text
development-only grid OOD split; not final hydroclimatic region split
```

这些 fallback 只能用于 pipeline/debugging，不能用于论文主 claim。

---

## 需要实现

```text
hydroda/regions/masks.py
hydroda/regions/quality_report.py
hydroda/splits/kdate.py
hydroda/splits/manifest.py
tests/test_region_masks.py
tests/test_split_no_leakage.py
scripts/build_region_masks.py
scripts/build_splits.py
```

---

## 输出

Scientific regions 成功时：

```text
artifacts/regions/US/region_masks.nc
artifacts/regions/US/region_quality.csv
reports/regions/region_quality_US.md
artifacts/splits/hydroda_ood_us_v1/US/{region}/K{K}_seed{seed}.json
```

Development-only fallback 时：

```text
artifacts/regions/US_grid_dev/region_masks.nc
reports/regions/region_quality_US_grid_dev.md
artifacts/splits/hydroda_ood_us_grid_dev/US/{region}/K{K}_seed{seed}.json
```

---

## Split matrix

```text
target_region: US-R1..US-R6 if scientific geolocation is available
fallback_target_region: US-GRID-R1..US-GRID-R6 if only grid split is available
K: 0, 4, 12
seed: 0..4 initially
```

---

## 验收标准

```text
1. 每个 region 有 mask 和 quality stats。
2. 每个 target_region/K/seed 有 split manifest。
3. K=0 没有 support labels。
4. source_val dates 全部在 2021。
5. target_context dates 全部在 2022。
6. target_query dates 全部在 2023–2025。
6. support 和 query 无重叠。
7. split selection flags 显示没有使用 query labels。
8. scientific region masks 只有在 geolocation gate 通过后才生成。
9. development-only grid regions 不得被报告为 hydroclimatic regimes。
```

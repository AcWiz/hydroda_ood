---
description: Build fixed HydroDA-OOD region masks from pre-registered region specs
argument-hint: [country=US]
---

# Claude Code Command: build-hydroda-regions

请执行 Phase 2A: Fixed Region Masks。

## 必读

```text
CLAUDE.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/03_REGION_PROTOCOL.md
tasks/phase2A_fixed_region_masks.md
specs/regions_v2.yaml
checklists/no_leakage_checklist.md
```

## 当前任务

使用固定区域范围生成 US scientific region masks：

```text
input geolocation: artifacts/geolocation/US_latlon.nc
region spec: specs/regions_v2.yaml
```

## 输出

```text
artifacts/regions/US/US_region_masks.nc
artifacts/regions/US/US_region_stats.json
reports/regions/US_region_mask_summary.md
figures/regions/US_region_masks_preview.png
```

## 强约束

- 不要使用 analysis increments。
- 不要使用 model errors。
- 不要根据 base_valid_mask coverage 移动区域边界。
- 不要把 base_valid_mask 当 region_mask。
- CN/AU 只做 spec validation，等待 geolocation artifact 后再生成正式 masks。

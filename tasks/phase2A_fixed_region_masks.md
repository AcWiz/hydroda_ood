# Phase 2A — Fixed Three-country Region Masks

## 目标

根据 `specs/regions_v2.yaml` 的 fixed_bbox_v1 定义生成 region masks。

当前可立即执行：

```text
US-R1..US-R6 using artifacts/geolocation/US_latlon.nc
```

CN/AU 区域已经预注册，但在对应 geolocation artifact 到位前只做 spec validation，不生成正式 masks。

---

## 必读

```text
CLAUDE.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/03_REGION_PROTOCOL.md
context/11_GEOLOCATION_RECOVERY_PROTOCOL.md
specs/regions_v2.yaml
specs/hydroda_dataset_contract.yaml
checklists/no_leakage_checklist.md
```

---

## 必须实现

```text
hydroda/regions/fixed_regions.py
hydroda/regions/masks.py
hydroda/regions/quality_report.py
scripts/build_region_masks.py
tests/test_fixed_region_specs.py
tests/test_us_region_masks.py
```

---

## US 输出

```text
artifacts/regions/US/US_region_masks.nc
artifacts/regions/US/US_region_stats.json
reports/regions/US_region_mask_summary.md
figures/regions/US_region_masks_preview.png
```

NetCDF 至少包含：

```text
lat[H,W]
lon[H,W]
region_mask_integer[H,W]
region_mask_onehot[region,H,W]
region_id[region]
region_name[region]
regime[region]
```

---

## 测试要求

```text
test_region_spec_has_three_countries
test_each_country_has_six_regions
test_region_ids_unique
test_bboxes_have_valid_latlon_ranges
test_no_bbox_overlap_within_country_under_half_open_rule
test_us_bboxes_within_recovered_latlon_domain
test_us_region_masks_shape_256_640
test_us_six_regions_nonempty
test_us_region_masks_no_overlap
test_region_construction_does_not_read_analysis_or_target
```

---

## 禁止事项

```text
不要读取 analysis increments 来定义区域
不要读取 model errors 来定义区域
不要根据 mask coverage 动态移动 bbox
不要把 base_valid_mask 当 region_mask
不要生成 CN/AU 正式 masks，除非对应 geolocation artifact 存在
```

---

## 完成汇报

用中文报告：

```text
完成内容：
- 新增文件：
- 修改文件：
- 运行命令：
- 生成 artifacts：
- 每个 US region pixel count：
- overlap 检查：
- CN/AU spec validation：
- 泄漏风险：
- 下一步：
```

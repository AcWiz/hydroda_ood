# Phase 0.5 — Geolocation and Dataset-Contract Resolution

## 目标

Phase 0 已完成 NetCDF audit。当前需要把实际数据结构、mask 语义和 geolocation gate 固化成可执行系统。

已知：

```text
DA.nc uses input[T, 12, H, W] and target[T, 4, H, W]
height = 256, width = 640
input channel 11 is dynamic binary mask
DA.nc itself has no lat/lon/projection metadata
user confirms /fastersharefiles2/fenglonghan/dataset/SMAP contains coordinate-related information
```

本阶段目标：

```text
1. 固化 multi-channel dataset contract；
2. 将 input channel 11 暂定为 base_valid_mask；
3. 扫描整个 SMAP 目录恢复 geolocation；
4. 给出 Phase 1 / Phase 2 gate 判定。
```

---

## 必读

```text
CLAUDE.md
context/00_EXECUTABLE_CONTEXT_MAP.md
context/02_DATA_AND_LEAKAGE_CONTRACT.md
context/11_GEOLOCATION_RECOVERY_PROTOCOL.md
specs/hydroda_dataset_contract.yaml
specs/geolocation_sources.yaml
checklists/no_leakage_checklist.md
```

---

## 必须完成

### 1. Dataset contract finalization

更新或确认：

```text
specs/hydroda_dataset_contract.yaml
```

必须反映实际结构：

```text
input[T, 12, H, W]
target[T, 4, H, W]
channel mapping by coordinate labels or audit mapping
```

保留 semantic aliases：

```text
sm_surface_forecast
sm_rootzone_forecast
sm_surface_analysis
sm_rootzone_analysis
base_valid_mask
```

不要假设所有语义变量都是 NetCDF 中独立 data variables。

---

### 2. Directory-level geolocation lookup

新增或更新：

```text
scripts/scan_smap_geolocation_sources.py
scripts/geolocation_recovery.py
hydroda/data/geolocation.py
tests/test_geolocation_candidate_scan.py
```

递归扫描：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP
```

寻找 lat/lon、projection、grid mapping、coordinate lookup、domain mask、原始处理脚本或配置。

轻量读取，不要一次性加载超大数组。

---

### 3. Candidate validation

对候选文件进行：

```text
shape compatibility check
lat/lon range sanity check
projection/transform metadata check
source trace recording
confidence scoring
```

若成功恢复坐标，输出：

```text
artifacts/geolocation/US_latlon.nc
```

若只恢复 projection/transform，输出：

```text
artifacts/geolocation/US_grid_mapping.json
```

---

## 必须输出

```text
artifacts/audits/smap_directory_manifest.json
reports/audits/smap_directory_manifest.md
artifacts/geolocation/geolocation_candidates_US.json
reports/audits/geolocation_lookup_US.md
```

如果成功：

```text
artifacts/geolocation/US_latlon.nc
artifacts/geolocation/US_grid_mapping.json  # 如果需要
```

测试：

```text
python -m pytest tests/test_dataset_contract_from_audit.py tests/test_geolocation_candidate_scan.py -v
```

---

## 禁止事项

```text
不要训练模型
不要构建正式 US-R1..US-R6 region masks，除非 lat/lon 已恢复并验证
不要用 analysis increments 或 target query labels 选择/验证 geolocation
不要把 grid-only tile 命名为真实 hydroclimatic region
不要把 base_valid_mask 改名为 obs_mask / region_mask / static_land_mask
```

---

## 成功标准

至少满足以下之一。

### A. 完全成功

```text
US_latlon.nc generated
lat/lon shape compatible with [256, 640]
range sanity checks pass
Phase 2 scientific regions: UNBLOCKED
```

### B. 部分成功

```text
candidate files found
projection/transform or x/y recovered
remaining conversion steps documented
Phase 2 scientific regions: BLOCKED pending conversion
```

### C. 可解释失败

```text
directory manifest generated
all candidate classes checked
no reliable lat/lon found
Phase 2 scientific regions: BLOCKED
development-only grid split: ALLOWED
```

---

## 完成汇报格式

用中文汇报：

1. 新增文件；
2. 修改文件；
3. 运行命令；
4. 生成 artifacts；
5. 候选 geolocation sources；
6. 是否成功恢复 lat/lon；
7. 若成功，lat/lon 的 shape/range/coverage；
8. 若失败，失败证据；
9. Phase 1 / Phase 2 状态；
10. 泄漏风险检查。

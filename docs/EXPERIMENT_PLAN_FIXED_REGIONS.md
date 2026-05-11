# HydroDA-OOD Experiment Plan Update — Fixed Three-country Region Ranges

## 1. 为什么要固定三国区域范围

原研究计划已经固定了 US/CN/AU 三个国家、每国 6 个 regime slots 的设计，但区域只给出了名称和科学意义。为了让 Claude Code 能稳定执行 region masks、K-date splits、cross-continent transfer 和 same-regime transfer，本版本将 18 个区域全部固定为可执行 lat/lon bounding boxes。

区域范围固定后，后续所有实验必须复用同一份 spec：

```text
specs/regions_v2.yaml
region_protocol_version = fixed_bbox_v1
```

---

## 2. 固定区域表

### United States

| ID | Regime | Name | Latitude | Longitude |
|---|---|---|---:|---:|
| US-R1 | R1 dryland_sparse_vegetation | Southwest Desert / Four Corners | 31.5 to 37.0 | -115.5 to -109.0 |
| US-R2 | R2 semi_arid_transition | Southern Great Plains | 32.0 to 38.5 | -103.5 to -96.0 |
| US-R3 | R3 irrigated_managed_agriculture | California Central Valley | 35.0 to 40.5 | -122.5 to -118.5 |
| US-R4 | R4 rainfed_agriculture | Corn Belt | 39.0 to 45.0 | -96.0 to -84.0 |
| US-R5 | R5 humid_high_vegetation | Southeast US | 29.5 to 36.5 | -91.0 to -78.0 |
| US-R6 | R6 mountain_cold_terrain_stress | Central Rockies | 38.0 to 45.0 | -112.5 to -105.0 |

### China

| ID | Regime | Name | Latitude | Longitude |
|---|---|---|---:|---:|
| CN-R1 | R1 dryland_sparse_vegetation | Xinjiang / Gobi-margin dryland | 39.0 to 45.0 | 82.0 to 94.0 |
| CN-R2 | R2 semi_arid_transition | Inner Mongolia transition belt | 40.0 to 46.5 | 106.0 to 119.0 |
| CN-R3 | R3 irrigated_managed_agriculture | North China Plain | 34.0 to 40.0 | 113.0 to 119.5 |
| CN-R4 | R4 rainfed_agriculture | Northeast China Plain | 42.0 to 48.5 | 122.0 to 130.0 |
| CN-R5 | R5 humid_high_vegetation | South China monsoon region | 22.0 to 28.0 | 106.0 to 118.0 |
| CN-R6 | R6 mountain_cold_terrain_stress | Eastern Tibetan Plateau | 29.0 to 35.5 | 94.0 to 104.0 |

### Australia

| ID | Regime | Name | Latitude | Longitude |
|---|---|---|---:|---:|
| AU-R1 | R1 dryland_sparse_vegetation | Central Australia Desert | -27.5 to -21.0 | 128.0 to 137.5 |
| AU-R2 | R2 semi_arid_transition | Inland rangeland transition | -30.5 to -24.5 | 137.5 to 146.0 |
| AU-R3 | R3 irrigated_managed_agriculture | Murray-Darling Basin | -35.7 to -31.0 | 142.0 to 149.5 |
| AU-R4 | R4 rainfed_agriculture | Southwest Australia Wheatbelt | -35.0 to -29.0 | 115.5 to 121.5 |
| AU-R5 | R5 humid_high_vegetation | Wet Tropics / Queensland coast | -19.5 to -14.5 | 144.0 to 147.0 |
| AU-R6 | R6 mountain_cold_terrain_stress | Australian Alps / SE Highlands | -37.5 to -35.8 | 146.0 to 150.0 |

---

## 3. 实验协议更新

### US-only stage

当前 US geolocation 已恢复，因此 US-R1..US-R6 可立即用于 HydroDA-OOD-US leave-one-region-out：

```text
source regions = other five US regions
target region  = held-out US region
source fit     = 2015-04 to 2020-12
source val      = 2021
target context  = 2022
target query    = 2023-01 to 2025-05
```

### CN/AU stage

CN/AU 区域范围已经预注册，但必须等待各自数据和 geolocation artifact 后才能生成正式 masks：

```text
CN_latlon.nc or equivalent
AU_latlon.nc or equivalent
```

### Same-regime transfer

同 regime 跨国迁移固定为：

```text
R1: US-R1, CN-R1, AU-R1
R2: US-R2, CN-R2, AU-R2
R3: US-R3, CN-R3, AU-R3
R4: US-R4, CN-R4, AU-R4
R5: US-R5, CN-R5, AU-R5
R6: US-R6, CN-R6, AU-R6
```

---

## 4. 审稿人防御规则

1. 区域在模型训练前固定。
2. 所有范围写入 machine-readable spec。
3. 区域不能根据 analysis increment 或 model error 修改。
4. 如果 bbox 质量不足，只能创建新 protocol version，不能 post-hoc 调整。
5. 主文报告固定范围，附录报告 region quality statistics。

# 03 Region Protocol — Fixed Hydroclimatic Regions v2.2

本文件定义 HydroDA-OOD 的固定区域协议。区域定义必须在模型训练前冻结，且不得根据 model performance、analysis increment 或 target query labels 修改。

---

## 1. 核心原则

区域定义必须满足：

```text
1. 在训练前固定；
2. 使用 geography / hydroclimate / land-use priors；
3. 不使用 analysis increments；
4. 不使用 model errors；
5. 不使用 target query labels；
6. 不根据实验结果回头修改；
7. 所有范围写入 specs/regions_v2.yaml 并版本化。
```

本版本固定为：

```text
region_protocol_version = fixed_bbox_v1
```

如果后续要从 bbox 升级为 polygon，只能创建新版本，例如：

```text
fixed_polygon_v2
```

并且必须在任何模型训练和结果比较之前完成。

---

## 2. Geolocation gate

正式科学区域 masks 必须由 geolocation artifact 生成。

US 当前已恢复：

```text
artifacts/geolocation/US_latlon.nc
recovery_method = directory_latlon_vector_lookup
```

CN/AU 后续接入时必须各自执行 geolocation recovery。不能假设 US 的 grid 适用于 CN/AU。

没有 geolocation artifact 的国家不得生成正式 scientific region masks，只能处于 pending 状态。

---

## 3. 六类 regime slots

每个国家固定 6 个 regime slots：

| Slot | Regime | 含义 |
|---|---|---|
| R1 | dryland_sparse_vegetation | 干旱/低植被，rainfall pulse 和 drydown 明显 |
| R2 | semi_arid_transition | 半干旱草地或农牧过渡 |
| R3 | irrigated_managed_agriculture | 灌溉或强管理农业 |
| R4 | rainfed_agriculture | 雨养农业，作物季节性明显 |
| R5 | humid_high_vegetation | 湿润高植被，微波观测受植被影响强 |
| R6 | mountain_cold_terrain_stress | 山地/冷区/雪/冻融/复杂地形 stress |

---

## 4. 固定区域范围

完整机器可读定义见：

```text
specs/regions_v2.yaml
```

### United States

| ID | Regime | Name | Latitude | Longitude |
|---|---|---|---:|---:|
| US-R1 | R1 | Southwest Desert / Four Corners | 31.5 to 37.0 | -115.5 to -109.0 |
| US-R2 | R2 | Southern Great Plains | 32.0 to 38.5 | -103.5 to -96.0 |
| US-R3 | R3 | California Central Valley | 35.0 to 40.5 | -122.5 to -118.5 |
| US-R4 | R4 | Corn Belt | 39.0 to 45.0 | -96.0 to -84.0 |
| US-R5 | R5 | Southeast US | 29.5 to 36.5 | -91.0 to -78.0 |
| US-R6 | R6 | Central Rockies | 38.0 to 45.0 | -112.5 to -105.0 |

### China

| ID | Regime | Name | Latitude | Longitude |
|---|---|---|---:|---:|
| CN-R1 | R1 | Xinjiang / Gobi-margin dryland | 39.0 to 45.0 | 82.0 to 94.0 |
| CN-R2 | R2 | Inner Mongolia transition belt | 40.0 to 46.5 | 106.0 to 119.0 |
| CN-R3 | R3 | North China Plain | 34.0 to 40.0 | 113.0 to 119.5 |
| CN-R4 | R4 | Northeast China Plain | 42.0 to 48.5 | 122.0 to 130.0 |
| CN-R5 | R5 | South China monsoon region | 22.0 to 28.0 | 106.0 to 118.0 |
| CN-R6 | R6 | Eastern Tibetan Plateau | 29.0 to 35.5 | 94.0 to 104.0 |

### Australia

| ID | Regime | Name | Latitude | Longitude |
|---|---|---|---:|---:|
| AU-R1 | R1 | Central Australia Desert | -27.5 to -21.0 | 128.0 to 137.5 |
| AU-R2 | R2 | Inland rangeland transition | -30.5 to -24.5 | 137.5 to 146.0 |
| AU-R3 | R3 | Murray-Darling Basin | -35.7 to -31.0 | 142.0 to 149.5 |
| AU-R4 | R4 | Southwest Australia Wheatbelt | -35.0 to -29.0 | 115.5 to 121.5 |
| AU-R5 | R5 | Wet Tropics / Queensland coast | -19.5 to -14.5 | 144.0 to 147.0 |
| AU-R6 | R6 | Australian Alps / SE Highlands | -37.5 to -35.8 | 146.0 to 150.0 |

---

## 5. Mask construction rule

默认使用 half-open intervals：

```text
lat_min <= lat < lat_max
lon_min <= lon < lon_max
```

禁止默认重叠。如果同一国家内任意 region overlap，应让 tests fail，而不是静默使用 priority rule。

输出 masks：

```text
region_mask_integer[H,W]  # 0 background, 1..6 regions
region_mask_onehot[6,H,W]
lat[H,W]
lon[H,W]
```

---

## 6. Region quality report

每个国家每个 region 必须报告：

```text
pixel count
fraction of country grid
lat/lon actual min/max
mean base_valid_mask coverage
monthly base_valid_mask coverage
support/query date availability
overlap pixel count
quality flag
```

如果某一区域在目标数据网格中为空或 coverage 极低，不能根据模型结果修改范围；必须记录为 quality issue，并在 protocol version 层面讨论是否需要预注册修订。

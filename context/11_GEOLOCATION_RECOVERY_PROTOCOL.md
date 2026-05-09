# 11 Geolocation Recovery Protocol

本文件定义 HydroDA-OOD 的 geolocation recovery 规则。它不是 `CLAUDE.md` 的附录，而是 Phase 0.5 和 Phase 2 的正式上下文。

---

## 1. 背景

Phase 0/0.5 初步结果表明，`DA.nc` 本身没有 lat/lon coords、global attrs、grid_mapping 或 projection metadata。该失败结论只针对 `DA.nc` 文件本身，不代表整个 SMAP 数据目录没有坐标信息。

当前美国数据目录：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP
```

用户确认该目录下存在可用于恢复 lat/lon 或 geolocation 的信息。因此，Claude Code 在进入正式 scientific region masks 前，必须执行 directory-level geolocation lookup。

---

## 2. 核心原则

不要只审计 `DA.nc`。必须把整个 SMAP 数据目录视为 geolocation source space。

正式 US-R1 到 US-R6 科学区域 masks 依赖可靠的：

```text
DA.nc grid index [height, width] -> lat/lon [height, width]
```

或：

```text
DA.nc grid index [height, width] -> projected x/y -> lat/lon
```

没有 lat/lon 时，可以继续 Phase 1 dataset 和 temporal K-date split，但不能进入最终 scientific region masks。

---

## 3. 搜索范围

递归扫描：

```text
/fastersharefiles2/fenglonghan/dataset/SMAP
```

候选文件类型：

```text
*.nc, *.nc4, *.h5, *.hdf5, *.zarr,
*.npy, *.npz, *.pkl, *.pickle, *.mat,
*.csv, *.txt, *.json, *.yaml, *.yml,
*.tif, *.tiff, *.vrt, *.shp, *.gpkg,
*.parquet, *.feather,
*.png, *.jpg, *.pdf, *.md, *.py, *.ipynb
```

关键词：

```text
lat, latitude, lon, longitude,
coord, coords, grid, geolocation, geo,
projection, proj, crs, transform, affine,
EASE, SMAP, domain, mask, land,
CONUS, US, usa, north_america,
crop, resample, row, col, ij, xy
```

---

## 4. Candidate ranking

优先级从高到低：

1. 文件名包含 `lat`, `lon`, `coord`, `grid`, `geo`, `projection`, `domain`, `mask`。
2. NetCDF/HDF/Zarr 文件中含有变量名 `lat`, `latitude`, `lon`, `longitude`, `x`, `y`, `row`, `col`。
3. JSON/YAML/TXT 中含有 `epsg`, `proj4`, `crs`, `transform`, `affine`, `geotransform`。
4. Python/Jupyter 代码中出现 `lat`, `lon`, `meshgrid`, `EASE`, `pyproj`, `rasterio`, `cartopy`。
5. `.npy/.npz/.pkl/.mat` 中存在 `[256, 640]`、`[640, 256]`、`[256]`、`[640]` 形状数组。

---

## 5. 候选验证标准

每个候选必须通过以下检查：

```yaml
shape_compatible:
  accepted:
    - [256, 640]
    - [640, 256] with explicit transpose evidence
    - lat_vector[256] + lon_vector[640]
    - y_vector[256] + x_vector[640] + projection

value_range:
  lat:
    broad_range: [-90, 90]
    expected_us_like: [15, 75]
  lon:
    broad_range: [-180, 180] or [0, 360]
    expected_us_like: [-180, -50] or [180, 310]

spatial_monotonicity:
  required: warn_only
  reason: curvilinear grids may not be strictly monotonic

metadata_trace:
  required: true
  must_record:
    - source_file
    - variable_names
    - transformation_steps
    - confidence
```

不要用 analysis increment、model error 或 query label distribution 来验证 geolocation。

---

## 6. 必须输出的 artifacts

总是输出：

```text
artifacts/audits/smap_directory_manifest.json
reports/audits/smap_directory_manifest.md
artifacts/geolocation/geolocation_candidates_US.json
reports/audits/geolocation_lookup_US.md
reports/audits/geolocation_lookup_US.md (Phase 0.5 completion report)
```

如果成功恢复 lat/lon：

```text
artifacts/geolocation/US_latlon.nc
```

如果成功恢复 projection/transform：

```text
artifacts/geolocation/US_grid_mapping.json
```

---

## 7. Phase gate 判定

### 成功

```text
US_latlon.nc generated
lat/lon shape compatible with [256, 640]
range sanity checks pass
Phase 2 scientific regions: UNBLOCKED
```

### 部分成功

```text
candidate files found
projection/transform or x/y recovered
remaining conversion steps documented
Phase 2 scientific regions: BLOCKED pending conversion
```

### 可解释失败

```text
directory manifest generated
candidate classes checked
no reliable lat/lon found
Phase 2 scientific regions: BLOCKED
development-only grid split: ALLOWED
```

---

## 8.1 US Geolocation Recovery — COMPLETE

**Status:** COMPLETE (2026-05-06)
**Method:** directory_latlon_vector_lookup (Method 8)
**Source:** /fastersharefiles2/fenglonghan/dataset/SMAP/america_lat.npy + america_lon.npy
**Output:** artifacts/geolocation/US_latlon.nc

Recovery verification:
- lat shape: [256, 640] ✓
- lon shape: [256, 640] ✓
- lat range: ~25°N to ~51°N (conus) ✓
- lon range: ~-124°E to ~-65°E (conus) ✓
- Phase 2 scientific regions: UNBLOCKED

## 8. Mask 语义

`input` channel 11 暂定为：

```text
base_valid_mask
```

不得别名为：

```text
obs_mask
region_mask
static_land_mask
```

`region_mask` 必须来自 geolocation/region artifact，不得来自 input channel 11。
